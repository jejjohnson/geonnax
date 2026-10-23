"""Tests for the sphere point sets and quadrature grids (GN2)."""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    fibonacci_sphere,
    gauss_legendre_grid,
    healpix_grid,
    latlon_grid,
    real_spherical_harmonics,
)
from geonnax.geo import lonlat_to_cartesian3d


GRIDS = [
    lambda **kw: fibonacci_sphere(500, **kw),
    lambda **kw: gauss_legendre_grid(12, **kw),
    lambda **kw: latlon_grid(18, 36, **kw),
    lambda **kw: healpix_grid(4, **kw),
]


@pytest.mark.parametrize("grid", GRIDS)
def test_weights_sum_to_four_pi_and_unit_norm(grid):
    with jax.enable_x64(True):
        points, weights = grid()
        assert abs(float(weights.sum()) - 4.0 * math.pi) < 1e-10
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("grid", GRIDS)
def test_lonlat_and_xyz_agree(grid):
    with jax.enable_x64(True):
        xyz, _ = grid()
        lonlat, _ = grid(output="lonlat")
        np.testing.assert_allclose(lonlat_to_cartesian3d(lonlat), xyz, atol=1e-12)
        degrees, _ = grid(output="lonlat", output_unit="degrees")
        np.testing.assert_allclose(np.radians(degrees), lonlat, atol=1e-12)


def test_gauss_legendre_grid_is_exact_for_harmonic_products():
    n_lat = 9  # exact for l + l' <= 17 with n_lon = 18 > l + l'
    l_max = 8
    with jax.enable_x64(True):
        points, weights = gauss_legendre_grid(n_lat)
        Y = np.asarray(real_spherical_harmonics(points, l_max))
    gram = (Y * np.asarray(weights)[:, None]).T @ Y
    np.testing.assert_allclose(gram, np.eye(Y.shape[1]), atol=1e-12)


def test_fibonacci_quadrature_error_is_small():
    # Fibonacci quadrature of a smooth, low-degree integrand has error
    # roughly O(n^-1) (Hardin et al. 2016); 5e-3 at n = 10^4 is a loose bound.
    with jax.enable_x64(True):
        points, weights = fibonacci_sphere(10_000)
        Y = np.asarray(real_spherical_harmonics(points, 4))
    means = (np.asarray(weights)[:, None] * Y).sum(axis=0)[1:]  # ∫ Y_lm, l >= 1
    assert np.max(np.abs(means)) < 5e-3


def test_latlon_first_cell_and_bad_arguments():
    lonlat, _ = latlon_grid(18, 36, output="lonlat", output_unit="degrees")
    np.testing.assert_allclose(lonlat[0], [-175.0, -85.0], atol=1e-4)
    with pytest.raises(ValueError):
        fibonacci_sphere(0)
    with pytest.raises(ValueError):
        gauss_legendre_grid(4, output="polar")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        latlon_grid(0, 4)


def test_outputs_are_jax_arrays():
    points, weights = fibonacci_sphere(10)
    assert isinstance(points, jnp.ndarray) and isinstance(weights, jnp.ndarray)


# --- healpix_grid ----------------------------------------------------------

# Reference values from healpy 1.20.1 (``healpy.pix2ang``, ``healpy.nest2ring``).
# Ring order at nside = 2: cos(colatitude) per ring with its pixel count, and
# longitudes in units of π/8.
Z2_RINGS = [(11 / 12, 4), (2 / 3, 8), (1 / 3, 8), (0.0, 8), (-1 / 3, 8),
            (-2 / 3, 8), (-11 / 12, 4)]  # fmt: skip
PHI2_RING_EIGHTHS = [2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10,
                     12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14,
                     1, 3, 5, 7, 9, 11, 13, 15, 2, 6, 10, 14]  # fmt: skip
# healpy.nest2ring(4, arange(192)): ring index of each nested pixel.
NEST2RING_4 = [
    74, 58, 57, 42, 43, 27, 26, 14, 41, 25, 24, 12, 13, 5, 4, 0,
    78, 62, 61, 46, 47, 31, 30, 17, 45, 29, 28, 15, 16, 7, 6, 1,
    82, 66, 65, 50, 51, 35, 34, 20, 49, 33, 32, 18, 19, 9, 8, 2,
    86, 70, 69, 54, 55, 39, 38, 23, 53, 37, 36, 21, 22, 11, 10, 3,
    136, 120, 135, 104, 105, 89, 88, 73, 119, 103, 102, 87, 72, 56, 71, 40,
    140, 124, 123, 108, 109, 93, 92, 77, 107, 91, 90, 75, 76, 60, 59, 44,
    144, 128, 127, 112, 113, 97, 96, 81, 111, 95, 94, 79, 80, 64, 63, 48,
    148, 132, 131, 116, 117, 101, 100, 85, 115, 99, 98, 83, 84, 68, 67, 52,
    188, 181, 180, 169, 170, 155, 154, 139, 168, 153, 152, 137, 138, 122, 121, 106,
    189, 183, 182, 172, 173, 159, 158, 143, 171, 157, 156, 141, 142, 126, 125, 110,
    190, 185, 184, 175, 176, 163, 162, 147, 174, 161, 160, 145, 146, 130, 129, 114,
    191, 187, 186, 178, 179, 167, 166, 151, 177, 165, 164, 149, 150, 134, 133, 118,
]  # fmt: skip


def _wrap(lon):
    return np.mod(lon + math.pi, 2.0 * math.pi) - math.pi


def test_healpix_ring_centres_match_healpy():
    with jax.enable_x64(True):
        lonlat, w = healpix_grid(2, nest=False, output="lonlat")
    z = np.concatenate([[zz] * k for zz, k in Z2_RINGS])
    lon = _wrap(np.asarray(PHI2_RING_EIGHTHS) * math.pi / 8.0)
    np.testing.assert_allclose(np.sin(lonlat[:, 1]), z, atol=1e-12)
    np.testing.assert_allclose(lonlat[:, 0], lon, atol=1e-12)
    np.testing.assert_allclose(w, 4.0 * math.pi / 48, rtol=1e-14)


def test_healpix_nested_is_ring_permuted_like_healpy():
    with jax.enable_x64(True):
        ring, _ = healpix_grid(4, nest=False)
        nested, _ = healpix_grid(4, nest=True)
    np.testing.assert_allclose(nested, ring[np.asarray(NEST2RING_4)], atol=1e-12)


@pytest.mark.parametrize("nside", [1, 2, 4, 8])
def test_healpix_nested_hierarchy(nside):
    # Nested pixel p at nside contains pixels 4p .. 4p+3 at 2·nside, so the
    # mean of the four children points close to the parent centre (observed
    # within 0.06 of a parent pixel width for nside 1..8). The order within
    # each group of four is pinned by the NEST2RING_4 test.
    with jax.enable_x64(True):
        parent, _ = healpix_grid(nside)
        child, _ = healpix_grid(2 * nside)
    mean = np.asarray(child).reshape(-1, 4, 3).mean(1)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    # A quarter of the parent pixel width sqrt(4π/N): misgrouped children miss.
    cos_sep = np.sum(mean * np.asarray(parent), axis=1)
    assert np.all(cos_sep > np.cos(0.25 * math.sqrt(4 * math.pi / parent.shape[0])))


@pytest.mark.parametrize("nside", [1, 2, 3, 4, 5])
def test_healpix_count_area_unit_norm(nside):
    with jax.enable_x64(True):
        points, weights = healpix_grid(nside, nest=False)
        assert points.shape == (12 * nside**2, 3)
        assert abs(float(weights.sum()) - 4.0 * math.pi) < 1e-12
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
        # Distinct pixel centres.
        assert np.unique(np.round(np.asarray(points), 9), axis=0).shape[0] == len(
            points
        )


def _healpix_harmonic_integrals(nside, l_max=6):
    ls = np.concatenate([[l] * (2 * l + 1) for l in range(l_max + 1)])
    ms = np.concatenate([np.arange(-l, l + 1) for l in range(l_max + 1)])
    # 4-fold azimuthal and north-south symmetry of the HEALPix centres.
    forbidden = ((ls + ms) % 2 == 1) | (ms % 4 != 0)
    with jax.enable_x64(True):
        points, w = healpix_grid(nside)
        integral = np.asarray(real_spherical_harmonics(points, l_max).T @ w)
    return integral, forbidden


def test_healpix_harmonic_quadrature_symmetry():
    # HEALPix is not an exact quadrature rule, but its symmetry makes ∫ Y_lm
    # vanish exactly when m ≢ 0 (mod 4) or l + m is odd.
    integral, forbidden = _healpix_harmonic_integrals(4)
    # x64 round-off on a sum of 192 terms of size O(1).
    np.testing.assert_allclose(integral[forbidden], 0.0, atol=1e-12)
    np.testing.assert_allclose(integral[0], math.sqrt(4 * math.pi), rtol=1e-12)


@pytest.mark.slow
def test_healpix_harmonic_quadrature_converges():
    # The remaining low-degree terms decay as O(nside⁻²): a ratio of 4 per
    # doubling (observed 4.0-4.6 for nside 2..32).
    errs = []
    for nside in (2, 4, 8, 16):
        integral, forbidden = _healpix_harmonic_integrals(nside)
        errs.append(np.abs(integral[~forbidden][1:]).max())
    assert all(a / b > 3.5 for a, b in itertools.pairwise(errs)), errs


def test_healpix_validation():
    with pytest.raises(ValueError, match="nside"):
        healpix_grid(0)
    with pytest.raises(ValueError, match="power of two"):
        healpix_grid(3)
    with pytest.raises(ValueError, match="integer"):
        healpix_grid(2.0)
    with pytest.raises(ValueError, match="integer"):
        healpix_grid(True)
    healpix_grid(3, nest=False)
    assert healpix_grid(np.int64(4))[0].shape == (192, 3)
