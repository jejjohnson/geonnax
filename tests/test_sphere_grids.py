"""Tests for the sphere point sets and quadrature grids (GN2)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    fibonacci_sphere,
    gauss_legendre_grid,
    latlon_grid,
    real_spherical_harmonics,
)
from geonnax.geo import lonlat_to_cartesian3d


GRIDS = [
    lambda **kw: fibonacci_sphere(500, **kw),
    lambda **kw: gauss_legendre_grid(12, **kw),
    lambda **kw: latlon_grid(18, 36, **kw),
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
