"""Tests for spherical needlets (GN7)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    fibonacci_sphere,
    gauss_legendre_grid,
    needlet_basis,
    needlet_window,
    real_spherical_harmonics,
)


@pytest.mark.parametrize("B", [1.5, 2.0, 3.0])
def test_window_partition_of_unity_and_support(B):
    b = needlet_window(B)
    xi = jnp.linspace(1.0, 10.0**3, 5001)
    with jax.enable_x64(True):
        xi = jnp.asarray(np.linspace(1.0, 1e3, 5001))
        j_max = int(np.ceil(np.log(1e3) / np.log(B))) + 1
        total = sum(b(xi / B**j) ** 2 for j in range(j_max + 1))
        np.testing.assert_allclose(np.asarray(total), 1.0, atol=1e-12)
        outside = b(jnp.asarray([0.0, 0.99 / B, 1.01 * B, 50.0]))
    assert np.all(np.asarray(outside) == 0.0)


def test_tight_frame_reconstruction():
    B, j_max, degree = 2.0, 3, 8  # B**j_max = 8: every degree <= 8 is covered
    with jax.enable_x64(True):
        rng = np.random.default_rng(0)
        coeffs = jnp.asarray(rng.normal(size=(degree + 1) ** 2))
        quad_points, quad_weights = gauss_legendre_grid(14, 28)  # exact to degree 26
        test_points, _ = fibonacci_sphere(50)

        def field(x):
            return real_spherical_harmonics(x, degree) @ coeffs

        psi_quad, _, _ = needlet_basis(quad_points, B=B, j_max=j_max)
        psi_test, _, _ = needlet_basis(test_points, B=B, j_max=j_max)
        inner = (quad_weights * field(quad_points)) @ psi_quad  # ⟨f, ψ_m⟩
        reconstruction = psi_test @ inner
        expected = field(test_points)
    np.testing.assert_allclose(
        np.asarray(reconstruction), np.asarray(expected), atol=1e-8
    )


def test_localisation():
    with jax.enable_x64(True):
        points, _ = fibonacci_sphere(3000)
        values, levels, centres = needlet_basis(points, j_max=4)
        column = int(np.flatnonzero(np.asarray(levels) == 4)[0])
        centre = np.asarray(centres[column])
        angle = np.arccos(np.clip(np.asarray(points) @ centre, -1.0, 1.0))
        magnitude = np.abs(np.asarray(values[:, column]))
    near = magnitude[angle < 0.05].max()
    for lo, hi in [(0.5, 1.0), (1.0, 2.0), (2.0, np.pi)]:
        band = magnitude[(angle >= lo) & (angle < hi)].max()
        assert band < 0.05 * near  # needlet decay away from its centre


def test_shapes_and_validation():
    values, levels, centres = needlet_basis(jnp.array([[0.0, 0.0, 1.0]]), j_max=1)
    assert (
        values.shape == (1, 69) and levels.shape == (69,) and centres.shape == (69, 3)
    )
    with pytest.raises(ValueError, match="B must"):
        needlet_window(1.0)
    with pytest.raises(ValueError, match="j_max"):
        needlet_basis(jnp.zeros((1, 3)), j_max=-1)
