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


def _reference_orthonormal(t, l_max):
    """P̄ₗᵐ(t) by the same recurrence in NumPy float64, with no exponent carry.

    Column order of `associated_legendre`. Exact (to rounding) wherever the
    sectoral seed stays in float64 range, which covers l_max <= 1000.
    """
    s = np.sqrt(np.clip(1.0 - t**2, 0.0, None))
    out = np.zeros((t.size, (l_max + 1) * (l_max + 2) // 2))
    col = {lm: i for i, lm in enumerate(associated_legendre_indices(l_max))}
    pmm = np.full(t.size, 1.0 / math.sqrt(2.0))
    for m in range(l_max + 1):
        if m > 0:
            pmm = -math.sqrt((2 * m + 1) / (2 * m)) * s * pmm
        p_prev, p_curr = np.zeros_like(t), pmm
        out[:, col[(m, m)]] = pmm
        for ell in range(m + 1, l_max + 1):
            if ell == m + 1:
                p_next = math.sqrt(2 * m + 3) * t * p_curr
            else:
                a = math.sqrt((4 * ell**2 - 1) / (ell**2 - m**2))
                b = math.sqrt(((ell - 1) ** 2 - m**2) / (4 * (ell - 1) ** 2 - 1))
                p_next = a * (t * p_curr - b * p_prev)
            out[:, col[(ell, m)]] = p_next
            p_prev, p_curr = p_curr, p_next
    return out


def test_float64_unchanged_by_exponent_carry():
    # #122: the carry rescales by exact powers of two, so float64 values equal the
    # plain recurrence to rounding: measured 4.7e-13 at l_max = 127 with
    # max |P̄| ≈ 11, the same as before the carry; 1e-12 leaves margin.
    t = np.concatenate([np.linspace(-1.0, 1.0, 21), [0.9999, -0.99999]])
    with jax.enable_x64(True):
        P = np.asarray(associated_legendre(jnp.asarray(t), 127))
    np.testing.assert_allclose(P, _reference_orthonormal(t, 127), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [256, pytest.param(512, marks=pytest.mark.slow)])
def test_orthonormal_float32_high_degree(n):
    # #122: without the exponent carry the sectoral seed (1 - t²)^{m/2}
    # underflowed float32 near the poles, leaving all-zero rows and a Gram error
    # of 1.2e-1 (n = 256) / 4.0e-1 (n = 512). With it: 1.2e-5 / 2.7e-5 measured;
    # 5e-5 leaves margin (float32 rounding accumulates over ~n recurrence steps).
    nodes, weights = gauss_legendre(n)
    P = np.asarray(associated_legendre(jnp.asarray(nodes, jnp.float32), n - 1))
    assert P.dtype == np.float32
    orders = np.array([m for _, m in associated_legendre_indices(n - 1)])
    err = 0.0
    for m in range(n):
        rows = P[:, orders == m].astype(np.float64).T  # (n − m, n)
        assert (np.abs(rows).max(axis=1) > 0).all(), f"zero row at m = {m}"
        gram = (rows * weights) @ rows.T
        err = max(err, float(np.abs(gram - np.eye(len(rows))).max()))
    assert err < 5e-5


def test_gradient_matches_finite_difference_at_high_degree():
    # d P̄₂₀₀¹⁵⁰ / dt through the exponent carry, against a central difference
    # (truncation ~h² · P''' ≈ 1e-6 relative here; measured agreement 3e-9).
    col = associated_legendre_indices(255).index((200, 150))

    def f(t):
        return associated_legendre(t[None], 255)[0, col]

    with jax.enable_x64(True):
        t0, h = jnp.float64(0.3), 1e-6
        grad = float(jax.grad(f)(t0))
        fd = float((f(t0 + h) - f(t0 - h)) / (2 * h))
    assert abs(grad - fd) <= 1e-6 * max(abs(fd), 1.0)


@pytest.mark.parametrize("t0", [0.0, 0.3, -0.9])
def test_gradient_matches_finite_difference_where_values_vanish(t0):
    # At t = 0 every P̄ₗᵐ with odd l − m is exactly zero but has a non-zero
    # derivative; the exponent carry must still give d/dt exactly (#122).
    # Central-difference truncation ~1e-10 at l_max = 8; measured 2e-10.
    def f(t):
        return associated_legendre(t[None], 8)[0]

    with jax.enable_x64(True):
        t, h = jnp.float64(t0), 1e-6
        jac = np.asarray(jax.jacfwd(f)(t))
        fd = np.asarray((f(t + h) - f(t - h)) / (2 * h))
    np.testing.assert_allclose(jac, fd, rtol=0, atol=1e-7)


def test_ldexp_helper_exact_at_zero_and_out_of_range():
    # #122: the exponent carry scales by powers of two with `_ldexp`, which
    # must have derivative 2^e at x = 0 (jnp.ldexp gives 1) and must not form
    # 0 · inf when e exceeds float32's range (exact poles at l_max ≈ 370).
    from geonnax._basis._legendre import _ldexp

    zero = jnp.float32(0.0)
    assert float(_ldexp(zero, jnp.int32(400))) == 0.0
    assert float(_ldexp(jnp.float32(0.75), jnp.int32(-400))) == 0.0
    assert float(_ldexp(jnp.float32(0.75), jnp.int32(5))) == 24.0
    grad = jax.grad(lambda x: _ldexp(x, jnp.int32(3)))(zero)
    assert float(grad) == 8.0


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
