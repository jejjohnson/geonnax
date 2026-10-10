"""Tests for the Neumann/periodic box bases and curlfree_basis (GN8)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    curlfree_basis,
    divfree_basis,
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
    gauss_legendre,
)


BOUNDARIES = ["dirichlet", "neumann", "periodic"]
L = 1.5


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_orthonormal_on_interval(boundary):
    nodes, weights = gauss_legendre(80)
    with jax.enable_x64(True):
        phi = np.asarray(
            fourier_basis_1d(jnp.asarray(L * nodes), 9, L, boundary=boundary)
        )
    gram = (phi * (L * weights)[:, None]).T @ phi
    np.testing.assert_allclose(gram, np.eye(9), atol=1e-12)


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_eigen_equation_and_sorted_eigenvalues(boundary):
    with jax.enable_x64(True):
        lam = fourier_eigenvalues_1d(7, L, boundary=boundary, dtype=jnp.float64)
        x = jnp.linspace(-L, L, 11)

        def column(t, m):
            return fourier_basis_1d(t[None], 7, L, boundary=boundary)[0, m]

        for m in range(7):
            second = jax.vmap(jax.grad(jax.grad(column), argnums=0), (0, None))(x, m)
            value = jax.vmap(column, (0, None))(x, m)
            np.testing.assert_allclose(-second, lam[m] * value, atol=1e-9)
    assert np.all(np.diff(np.asarray(lam)) >= 0)


def test_boundary_conditions_hold():
    with jax.enable_x64(True):
        ends = jnp.array([-L, L])

        def basis(t, boundary):
            return fourier_basis_1d(t, 6, L, boundary=boundary)

        np.testing.assert_allclose(basis(ends, "dirichlet"), 0.0, atol=1e-12)
        neumann = jax.vmap(
            lambda e: jax.jacfwd(lambda t: basis(t[None], "neumann")[0])(e)
        )(ends)
        np.testing.assert_allclose(neumann, 0.0, atol=1e-12)
        periodic = basis(ends, "periodic")
        np.testing.assert_allclose(periodic[0], periodic[1], atol=1e-12)
        grads = jax.vmap(
            lambda e: jax.jacfwd(lambda t: basis(t[None], "periodic")[0])(e)
        )(ends)
        np.testing.assert_allclose(grads[0], grads[1], atol=1e-12)


def test_dirichlet_default_unchanged():
    x = jnp.linspace(-L, L, 7)
    j = np.arange(1, 5)
    expected = np.sin(np.outer(np.asarray(x) + L, j) * math.pi / (2 * L)) / math.sqrt(L)
    np.testing.assert_allclose(
        np.asarray(fourier_basis_1d(x, 4, L)), expected, atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(fourier_basis_1d(x, 4, L)),
        np.asarray(fourier_basis_1d(x, 4, L, boundary="dirichlet")),
    )


def test_mixed_boundaries_in_fourier_basis():
    xy = jax.random.uniform(jax.random.key(0), (20, 2), minval=-L, maxval=L)
    phi, lam = fourier_basis(xy, (4, 3), L, boundary=("periodic", "dirichlet"))
    px = fourier_basis_1d(xy[:, 0], 4, L, boundary="periodic")
    py = fourier_basis_1d(xy[:, 1], 3, L)
    np.testing.assert_allclose(
        phi, np.einsum("na,nb->nab", px, py).reshape(20, 12), atol=1e-6
    )
    np.testing.assert_allclose(
        lam,
        fourier_eigenvalues((4, 3), L, 2, boundary=("periodic", "dirichlet")),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="boundary"):
        fourier_basis(xy, 3, L, boundary="robin")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_curlfree_is_gradient_and_curl_free(boundary):
    with jax.enable_x64(True):
        xy = jax.random.uniform(
            jax.random.key(1), (8, 2), minval=-L, maxval=L, dtype=jnp.float64
        )
        phi, lam = curlfree_basis(xy, 3, L, boundary=boundary)

        def potential(p):
            return fourier_basis(p[None], 3, L, boundary=boundary)[0][0]

        grads = jax.vmap(jax.jacfwd(potential))(xy)  # (N, M, 2)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(grads), atol=1e-10)

        def field(p, m):
            return curlfree_basis(p[None], 3, L, boundary=boundary)[0][0, m]

        for m in range(9):
            jac = jax.vmap(jax.jacfwd(field), (0, None))(xy, m)  # (N, 2, 2)
            np.testing.assert_allclose(jac[:, 1, 0] - jac[:, 0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(
            lam, fourier_eigenvalues(3, L, 2, boundary=boundary, dtype=jnp.float64)
        )


@pytest.mark.parametrize("boundary", ["neumann", "periodic"])
def test_divfree_boundary_option_is_divergence_free(boundary):
    with jax.enable_x64(True):
        xy = jax.random.uniform(
            jax.random.key(2), (6, 2), minval=-L, maxval=L, dtype=jnp.float64
        )

        def field(p, m):
            return divfree_basis(p[None], 3, L, boundary=boundary)[0][0, m]

        for m in range(9):
            jac = jax.vmap(jax.jacfwd(field), (0, None))(xy, m)
            np.testing.assert_allclose(jac[:, 0, 0] + jac[:, 1, 1], 0.0, atol=1e-10)
