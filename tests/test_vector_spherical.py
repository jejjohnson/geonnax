"""Tests for the vector spherical harmonics (GN5)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    fibonacci_sphere,
    gauss_legendre_grid,
    harmonic_degrees,
    real_spherical_harmonics,
    vector_spherical_harmonics,
)
from geonnax.geo import enu_basis


def test_orthonormal_on_gauss_legendre_grid():
    l_max = 8
    with jax.enable_x64(True):
        points, weights = gauss_legendre_grid(l_max + 2)
        psi, phi, _ = vector_spherical_harmonics(points, l_max)
    fields = np.concatenate([np.asarray(psi), np.asarray(phi)], axis=1)  # (N, 2M, 3)
    gram = np.einsum("n,nid,njd->ij", np.asarray(weights), fields, fields)
    np.testing.assert_allclose(gram, np.eye(fields.shape[1]), atol=1e-10)


def test_tangent_and_finite_at_poles():
    poles = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    points = jnp.concatenate([fibonacci_sphere(50)[0], poles])
    psi, phi, _ = vector_spherical_harmonics(points, 5)
    assert jnp.isfinite(psi).all() and jnp.isfinite(phi).all()
    for field in (psi, phi):
        radial = jnp.einsum("nmd,nd->nm", field, points)
        assert float(jnp.max(jnp.abs(radial))) < 1e-5


def _surface_divergence(field_fn, u):
    """∇_S · F at u, for F a tangent field given on R³ by `field_fn`."""
    jac = jax.jacfwd(field_fn)(u)  # (3, 3): ∂F_i/∂x_j
    projector = jnp.eye(3) - jnp.outer(u, u)
    return jnp.trace(projector @ jac)


@pytest.mark.parametrize("column", [0, 3, 7])
def test_divergence_identities(column):
    l_max = 2
    degree = harmonic_degrees(l_max)[1:][column]

    def field(kind):
        def fn(x):
            u = x / jnp.linalg.norm(x)
            psi, phi, _ = vector_spherical_harmonics(u[None], l_max, normalized=False)
            return (psi if kind == "psi" else phi)[0, column]

        return fn

    with jax.enable_x64(True):
        points = fibonacci_sphere(20)[0]
        Y = real_spherical_harmonics(points, l_max)[:, 1 + column]
        for k, u in enumerate(points):
            div_psi = _surface_divergence(field("psi"), u)
            div_phi = _surface_divergence(field("phi"), u)
            assert abs(float(div_psi) + degree * (degree + 1) * float(Y[k])) < 1e-8
            assert abs(float(div_phi)) < 1e-8


def test_enu_output_matches_projection_and_validation():
    points, _ = fibonacci_sphere(30)
    psi, _, _ = vector_spherical_harmonics(points, 3)
    psi_en, _, degrees = vector_spherical_harmonics(points, 3, output="enu")
    lonlat = jnp.stack(
        [jnp.arctan2(points[:, 1], points[:, 0]), jnp.arcsin(points[:, 2])], axis=-1
    )
    frame = enu_basis(lonlat, input_unit="radians")[:, :2, :]
    np.testing.assert_allclose(
        np.asarray(psi_en), np.einsum("nmd,ncd->nmc", psi, frame), atol=1e-5
    )
    assert psi_en.shape == (30, 15, 2) and degrees.shape == (15,)
    with pytest.raises(ValueError, match="l_max"):
        vector_spherical_harmonics(points, 0)
