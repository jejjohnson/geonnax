"""Tests for `real_spherical_harmonics`: values, orthonormality, high degree."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    gauss_legendre,
    harmonic_degrees,
    legendre_polynomials,
    real_spherical_harmonics,
)


def _unit_points(n: int, seed: int = 0) -> np.ndarray:
    """``n`` random unit vectors plus both poles and an equator point."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    special = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    return np.concatenate([pts, special], axis=0)


def _reference_harmonics(unit_xyz: np.ndarray, l_max: int) -> np.ndarray:
    r"""The pre-#73 implementation, in NumPy float64.

    Unnormalised $Q_l^m = P_l^m / (1-z^2)^{m/2}$ seeded from
    $(-1)^m (2m-1)!!$, scaled by the factorial ratio per column and paired
    with $\mathrm{Re}/\mathrm{Im}\,(x+iy)^m$; columns stacked one by one. It
    overflows in float64 near $l_{\max} \approx 85$, so it is used for small
    degrees only.
    """
    x, y, z = unit_xyz[:, 0], unit_xyz[:, 1], unit_xyz[:, 2]
    Q: dict[tuple[int, int], np.ndarray] = {}
    for m in range(l_max + 1):
        dfac = 1.0
        for k in range(1, 2 * m, 2):
            dfac *= float(k)
        Q[(m, m)] = np.full_like(z, (-1.0) ** m * dfac)
        if m + 1 <= l_max:
            Q[(m + 1, m)] = (2.0 * m + 1.0) * z * Q[(m, m)]
        for ell in range(m + 2, l_max + 1):
            num = (2.0 * ell - 1.0) * z * Q[(ell - 1, m)] - (ell + m - 1.0) * Q[
                (ell - 2, m)
            ]
            Q[(ell, m)] = num / (ell - m)
    re, im = [np.ones_like(x)], [np.zeros_like(x)]
    for _ in range(l_max):
        re, im = [*re, x * re[-1] - y * im[-1]], [*im, x * im[-1] + y * re[-1]]

    inv_sqrt_4pi = 1.0 / math.sqrt(4.0 * math.pi)
    columns = []
    for ell in range(l_max + 1):
        for m in range(-ell, ell + 1):
            am = abs(m)
            if am == 0:
                columns.append(math.sqrt(2.0 * ell + 1.0) * inv_sqrt_4pi * Q[(ell, 0)])
            else:
                ratio = math.factorial(ell - am) / math.factorial(ell + am)
                norm = math.sqrt(2.0 * (2.0 * ell + 1.0) * ratio) * inv_sqrt_4pi
                trig = re[am] if m > 0 else im[am]
                columns.append(norm * Q[(ell, am)] * trig)
    return np.stack(columns, axis=-1)


def _sphere_quadrature(l_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre × uniform-longitude grid, exact for degree ``2 l_max``."""
    nodes, weights = gauss_legendre(l_max + 1)
    n_lon = 2 * l_max + 1
    lon = 2.0 * np.pi * np.arange(n_lon) / n_lon
    z, phi = np.meshgrid(nodes, lon, indexing="ij")
    s = np.sqrt(1.0 - z**2)
    xyz = np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=-1).reshape(-1, 3)
    w = np.repeat(weights * 2.0 * np.pi / n_lon, n_lon)
    return xyz, w


@pytest.mark.parametrize("l_max", [0, 1, 2, 5, 12, 20])
def test_matches_previous_implementation_float64(l_max):
    pts = _unit_points(40)
    ref = _reference_harmonics(pts, l_max)
    with jax.enable_x64(True):
        Y = np.asarray(real_spherical_harmonics(jnp.asarray(pts), l_max))
    assert Y.shape == (pts.shape[0], (l_max + 1) ** 2)
    np.testing.assert_allclose(Y, ref, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("l_max", [8, 30])
def test_matches_previous_implementation_float32(l_max):
    pts = _unit_points(40, seed=1)
    ref = _reference_harmonics(pts, l_max)
    Y = real_spherical_harmonics(jnp.asarray(pts, dtype=jnp.float32), l_max)
    assert Y.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(Y), ref, rtol=0.0, atol=5e-5)


@pytest.mark.parametrize("l_max", [4, 16])
def test_orthonormal_on_quadrature_grid(l_max):
    xyz, w = _sphere_quadrature(l_max)
    with jax.enable_x64(True):
        Y = np.asarray(real_spherical_harmonics(jnp.asarray(xyz), l_max))
    gram = einx.dot("n a, n b -> a b", einx.multiply("n a, n -> n a", Y, w), Y)
    np.testing.assert_allclose(gram, np.eye((l_max + 1) ** 2), atol=1e-12)


@pytest.mark.parametrize(("l_max", "x64"), [(12, False), (100, True)])
def test_addition_theorem(l_max, x64):
    a, b = _unit_points(6, seed=2), _unit_points(6, seed=3)
    with jax.enable_x64(x64):
        Ya = np.asarray(real_spherical_harmonics(jnp.asarray(a), l_max))
        Yb = np.asarray(real_spherical_harmonics(jnp.asarray(b), l_max))
        cos = np.clip(einx.dot("n d, n d -> n", a, b), -1.0, 1.0)
        P = np.asarray(legendre_polynomials(jnp.asarray(cos), l_max))
    degrees = np.asarray(harmonic_degrees(l_max))
    onehot = (degrees[:, None] == np.arange(l_max + 1)[None, :]).astype(float)
    lhs = einx.dot("n f, f l -> n l", Ya * Yb, onehot)
    rhs = einx.multiply(
        "n l, l -> n l", P, (2 * np.arange(l_max + 1) + 1) / (4 * np.pi)
    )
    atol = 1e-9 if x64 else 1e-4
    np.testing.assert_allclose(lhs, rhs, atol=atol)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_finite_at_high_degree(dtype):
    pts = _unit_points(50, seed=4)
    with jax.enable_x64(dtype == jnp.float64):
        Y = real_spherical_harmonics(jnp.asarray(pts, dtype=dtype), 150)
        assert Y.dtype == dtype
        Y = np.asarray(Y)
    assert Y.shape == (pts.shape[0], 151**2)
    assert np.isfinite(Y).all()
    # Each degree block has sum_m Y_lm(x)^2 = (2l + 1) / (4 pi).
    degrees = np.asarray(harmonic_degrees(150))
    onehot = (degrees[:, None] == np.arange(151)[None, :]).astype(float)
    sq = einx.dot("n f, f l -> n l", Y**2, onehot)
    expected = np.broadcast_to((2 * np.arange(151) + 1) / (4 * np.pi), sq.shape)
    np.testing.assert_allclose(
        sq, expected, rtol=1e-3 if dtype == jnp.float32 else 1e-9
    )


@pytest.mark.parametrize("l_max", [191, pytest.param(255, marks=pytest.mark.slow)])
def test_float32_matches_float64_beyond_old_limit(l_max):
    # #122: near the poles Q̄ grows like sin^{-m}θ and (x + iy)^m shrinks like
    # sin^m θ; before the exponent carry float32 overflowed to NaN from
    # l_max ≈ 180. Half the points lie within 0.15 rad of a pole, plus the poles.
    # Measured max |Y32 − Y64| = 2.1e-3 (191) and 3.8e-3 (255) with max |Y| ≈ 6:
    # float32 rounding accumulated over ~l_max recurrence steps; 6e-3 leaves margin.
    rng = np.random.default_rng(0)
    theta = np.r_[rng.uniform(0, np.pi, 30), rng.uniform(0, 0.15, 30)]
    phi = rng.uniform(0, 2 * np.pi, 60)
    pts = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1
    )
    pts = np.concatenate([pts, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    with jax.enable_x64(True):
        y64 = np.asarray(real_spherical_harmonics(jnp.asarray(pts, jnp.float64), l_max))
    y32 = np.asarray(real_spherical_harmonics(jnp.asarray(pts, jnp.float32), l_max))
    assert np.isfinite(y32).all()
    np.testing.assert_allclose(y32, y64, rtol=0, atol=6e-3)


def test_float32_finite_at_exact_poles_at_very_high_degree():
    # #122: at the exact poles (x + iy)^m is exactly zero while Q̄'s exponent
    # passes float32's range near l_max ≈ 370; the power-of-two scaling must
    # give 0, not 0 · inf.
    pts = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=jnp.float32)
    assert bool(jnp.isfinite(real_spherical_harmonics(pts, 400)).all())


def test_trace_size_independent_of_degree():
    x = jnp.asarray(_unit_points(4), dtype=jnp.float32)

    def n_eqns(l_max):
        return len(jax.make_jaxpr(lambda u: real_spherical_harmonics(u, l_max))(x).eqns)

    assert n_eqns(10) == n_eqns(60)


@pytest.mark.parametrize("l_max", [20, pytest.param(200, marks=pytest.mark.slow)])
def test_gradient_finite_at_poles(l_max):
    poles = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    jac = jax.vmap(jax.jacfwd(lambda p: real_spherical_harmonics(p[None], l_max)[0]))(
        poles
    )
    assert jac.shape == (2, (l_max + 1) ** 2, 3)
    assert bool(jnp.isfinite(jac).all())


def test_input_validation():
    with pytest.raises(ValueError, match="l_max"):
        real_spherical_harmonics(jnp.zeros((2, 3)), -1)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        real_spherical_harmonics(jnp.zeros((2, 2)), 2)
