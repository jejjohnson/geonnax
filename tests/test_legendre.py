"""Tests for the Legendre primitives and Gauss–Legendre quadrature (GN1)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from geonnax.basis import (
    associated_legendre,
    associated_legendre_indices,
    gauss_legendre,
    legendre_polynomials,
)


T = np.linspace(-1.0, 1.0, 41)


def test_legendre_polynomials_match_scipy():
    with jax.enable_x64(True):
        P = np.asarray(legendre_polynomials(jnp.asarray(T), 200))
    ref = np.stack([sp.eval_legendre(ell, T) for ell in range(201)], axis=-1)
    assert P.shape == (41, 201)
    np.testing.assert_allclose(P, ref, atol=1e-12)


def test_legendre_polynomials_shape_and_l_max_zero():
    t = jnp.zeros((2, 3))
    assert legendre_polynomials(t, 4).shape == (2, 3, 5)
    assert legendre_polynomials(t, 0).shape == (2, 3, 1)
    with pytest.raises(ValueError, match="l_max"):
        legendre_polynomials(t, -1)


def test_indices_order_and_count():
    idx = associated_legendre_indices(3)
    assert len(idx) == 10
    assert idx[:4] == ((0, 0), (1, 0), (1, 1), (2, 0))


def _column_scale(ref):
    return np.maximum(np.max(np.abs(ref), axis=0, keepdims=True), 1.0)


def test_classical_matches_scipy_lpmv():
    l_max = 20
    with jax.enable_x64(True):
        A = np.asarray(associated_legendre(jnp.asarray(T), l_max, normalization="none"))
    idx = associated_legendre_indices(l_max)
    ref = np.stack([sp.lpmv(m, ell, T) for ell, m in idx], axis=-1)
    # P_l^m spans many orders of magnitude, and some values are near-roots of a
    # large polynomial, so compare relative to each column's scale.
    np.testing.assert_allclose(
        A / _column_scale(ref), ref / _column_scale(ref), atol=1e-12
    )


def test_schmidt_matches_definition():
    l_max = 20
    with jax.enable_x64(True):
        S = np.asarray(
            associated_legendre(jnp.asarray(T), l_max, normalization="schmidt")
        )
    ref = np.stack(
        [
            math.sqrt(
                (2 - (m == 0)) * math.factorial(ell - m) / math.factorial(ell + m)
            )
            * (-1.0) ** m
            * sp.lpmv(m, ell, T)
            for ell, m in associated_legendre_indices(l_max)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(S, ref, atol=1e-12)


@pytest.mark.parametrize("m", [0, 3, 50])
def test_orthonormal_by_quadrature(m):
    l_max = 200
    nodes, weights = gauss_legendre(220)
    with jax.enable_x64(True):
        P = np.asarray(associated_legendre(jnp.asarray(nodes), l_max))
    cols = [
        i for i, (_, mm) in enumerate(associated_legendre_indices(l_max)) if mm == m
    ]
    gram = (P[:, cols] * weights[:, None]).T @ P[:, cols]
    np.testing.assert_allclose(gram, np.eye(len(cols)), atol=1e-11)
    assert np.isfinite(P).all()


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        associated_legendre(jnp.zeros(3), 2, normalization="bad")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("n", [1, 4, 9])
def test_gauss_legendre_exactness(n):
    nodes, weights = gauss_legendre(n)
    for degree in (2 * n - 2, 2 * n - 1):
        exact = (1.0 - (-1.0) ** (degree + 1)) / (degree + 1)  # ∫_{-1}^{1} t^d dt
        assert abs(np.sum(weights * nodes**degree) - exact) < 1e-13
    with pytest.raises(ValueError, match="n must"):
        gauss_legendre(0)


def test_gradient_finite_inside_interval():
    t = jnp.linspace(-0.95, 0.95, 7)

    def f(t):
        return jnp.sum(associated_legendre(t, 8)) + jnp.sum(legendre_polynomials(t, 8))

    assert jnp.isfinite(jax.grad(f)(t)).all()
