"""Tests for the spherical-shell basis (GN9)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    gauss_legendre,
    gauss_legendre_grid,
    real_spherical_harmonics,
    shell_basis,
)


R0, R1 = 1.0, 1.8


def test_orthonormal_in_area_times_radius_measure():
    l_max, n_radial = 4, 5
    with jax.enable_x64(True):
        directions, w_dir = gauss_legendre_grid(l_max + 2)
        nodes, w_r = gauss_legendre(n_radial + 2)
        radii = 0.5 * (R1 - R0) * nodes + 0.5 * (R1 + R0)
        points = jnp.concatenate([directions * r for r in radii])
        weights = jnp.concatenate([w_dir * (0.5 * (R1 - R0) * w) for w in w_r])
        values, _ = shell_basis(points, l_max, n_radial, r_inner=R0, r_outer=R1)
    values = np.asarray(values)
    gram = (values * np.asarray(weights)[:, None]).T @ values
    np.testing.assert_allclose(gram, np.eye(values.shape[1]), atol=1e-10)


def test_single_radial_mode_is_scaled_harmonics_and_indices():
    with jax.enable_x64(True):
        directions, _ = gauss_legendre_grid(4)
        points = 1.3 * directions
        values, indices = shell_basis(points, 3, 1, r_inner=R0, r_outer=R1)
        expected = real_spherical_harmonics(directions, 3) / math.sqrt(R1 - R0)
    np.testing.assert_allclose(np.asarray(values), np.asarray(expected), atol=1e-12)
    assert indices.shape == (16, 2) and indices[-1].tolist() == [3, 0]


def test_chebyshev_radial_and_validation():
    points = jnp.array([[0.0, 0.0, R0], [0.0, 0.0, R1]])
    values, _ = shell_basis(points, 0, 3, r_inner=R0, r_outer=R1, radial="chebyshev")
    y00 = 1.0 / math.sqrt(4.0 * math.pi)
    np.testing.assert_allclose(
        np.asarray(values), y00 * np.array([[1, -1, 1], [1, 1, 1]]), atol=1e-6
    )
    with pytest.raises(ValueError, match="r_inner"):
        shell_basis(points, 1, 2, r_inner=2.0, r_outer=1.0)
    with pytest.raises(ValueError, match="n_radial"):
        shell_basis(points, 1, 0, r_inner=R0, r_outer=R1)
