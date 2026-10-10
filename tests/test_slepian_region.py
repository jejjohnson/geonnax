"""Tests for Slepian bases of general regions (GN6)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    fibonacci_sphere,
    gauss_legendre,
    slepian_cap_basis,
    slepian_polar_gap_basis,
    slepian_region_basis,
    spherical_polygon_mask,
)


def _band_quadrature(z_lo: float, z_hi: float, n_z: int, n_lon: int):
    """Exact quadrature over {z_lo < z < z_hi}: Gauss–Legendre in z × uniform lon."""
    nodes, weights = gauss_legendre(n_z)
    z = 0.5 * (z_hi - z_lo) * nodes + 0.5 * (z_hi + z_lo)
    w_z = 0.5 * (z_hi - z_lo) * weights
    lon = 2.0 * math.pi * np.arange(n_lon) / n_lon
    zz, ll = np.meshgrid(z, lon, indexing="ij")
    r = np.sqrt(1.0 - zz**2)
    points = np.stack([r * np.cos(ll), r * np.sin(ll), zz], axis=-1).reshape(-1, 3)
    weights = np.repeat(w_z * 2.0 * math.pi / n_lon, n_lon)
    return jnp.asarray(points), jnp.asarray(weights)


def test_region_basis_reproduces_cap_concentrations():
    l_max, cap = 6, 0.5
    with jax.enable_x64(True):
        points, weights = _band_quadrature(math.cos(cap), 1.0, l_max + 2, 4 * l_max)
        region = slepian_region_basis(l_max, points, weights, n_modes=49)
        reference = slepian_cap_basis(l_max, cap)
    np.testing.assert_allclose(
        np.asarray(region.eigenvalues), np.asarray(reference.eigenvalues), atol=1e-6
    )


def test_shannon_number_default_and_concentration_range():
    l_max, cap = 8, 0.6
    with jax.enable_x64(True):
        points, weights = _band_quadrature(math.cos(cap), 1.0, l_max + 2, 4 * l_max)
        full = slepian_region_basis(l_max, points, weights, n_modes=(l_max + 1) ** 2)
        default = slepian_region_basis(l_max, points, weights)
    shannon = (l_max + 1) ** 2 * (1.0 - math.cos(cap)) / 2.0
    assert abs(float(np.sum(np.asarray(full.eigenvalues))) - shannon) < 1e-8
    assert default.num_modes == round(shannon)
    vals = np.asarray(full.eigenvalues)
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert np.all(np.diff(vals) <= 1e-12)  # sorted by decreasing concentration


def test_polar_gap_matches_band_quadrature_and_is_symmetric():
    l_max, gap = 6, 0.3
    with jax.enable_x64(True):
        basis = slepian_polar_gap_basis(l_max, gap, n_modes=(l_max + 1) ** 2)
        c = math.cos(gap)
        points, weights = _band_quadrature(-c, c, l_max + 2, 4 * l_max)
        region = slepian_region_basis(l_max, points, weights, n_modes=(l_max + 1) ** 2)
        x = fibonacci_sphere(64)[0]
        flipped = x * jnp.array([1.0, 1.0, -1.0])
        g, g_flipped = basis.evaluate(x), basis.evaluate(flipped)
    np.testing.assert_allclose(
        np.asarray(basis.eigenvalues), np.asarray(region.eigenvalues), atol=1e-10
    )
    # Concentrations come in near-degenerate pairs, so compare per-mode
    # magnitudes only for modes whose parity is well defined.
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(g)), axis=0),
        np.sort(np.abs(np.asarray(g_flipped)), axis=0),
        atol=1e-6,  # eigenvectors of near-degenerate pairs mix at ~1e-8
    )


def _brute_force_inside(lonlat_deg, lon_range, lat_range):
    lon, lat = lonlat_deg[:, 0], lonlat_deg[:, 1]
    lon_lo, lon_hi = lon_range
    in_lon = (
        (lon >= lon_lo) & (lon <= lon_hi)
        if lon_lo < lon_hi
        else (lon >= lon_lo) | (lon <= lon_hi)
    )
    return in_lon & (lat >= lat_range[0]) & (lat <= lat_range[1])


@pytest.mark.parametrize(
    ("lon_range", "lat_range"),
    [((-20.0, 30.0), (-10.0, 25.0)), ((170.0, -165.0), (5.0, 20.0))],
)
def test_polygon_mask_on_rectangles(lon_range, lat_range):
    # Dense vertices along the parallels make the great-circle edges follow
    # them closely; test points well away from the boundary.
    lo, hi = lon_range
    span = (hi - lo) % 360.0
    south = [[lo + span * t, lat_range[0]] for t in np.linspace(0, 1, 200)]
    north = [[lo + span * t, lat_range[1]] for t in np.linspace(1, 0, 200)]
    polygon = np.array(south + north)
    polygon[:, 0] = (polygon[:, 0] + 180.0) % 360.0 - 180.0
    points_lonlat, _ = fibonacci_sphere(4000, output="lonlat", output_unit="degrees")
    points_xyz, _ = fibonacci_sphere(4000)
    mask = np.asarray(spherical_polygon_mask(points_xyz, jnp.asarray(polygon)))
    expected = _brute_force_inside(np.asarray(points_lonlat), lon_range, lat_range)
    lat = np.asarray(points_lonlat)[:, 1]
    lon = np.asarray(points_lonlat)[:, 0]
    margin = 1.5
    near_edge = (
        (np.abs(lat - lat_range[0]) < margin)
        | (np.abs(lat - lat_range[1]) < margin)
        | (np.abs(((lon - lo + 180) % 360) - 180) < margin)
        | (np.abs(((lon - hi + 180) % 360) - 180) < margin)
    )
    assert expected.sum() > 20
    np.testing.assert_array_equal(mask[~near_edge], expected[~near_edge])


def test_validation():
    with pytest.raises(ValueError, match="gap_colatitude"):
        slepian_polar_gap_basis(3, 2.0)
    with pytest.raises(ValueError, match="polygon"):
        spherical_polygon_mask(jnp.zeros((1, 3)), jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="region_weights"):
        slepian_region_basis(2, jnp.zeros((3, 3)), jnp.zeros(2))
