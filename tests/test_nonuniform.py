"""Tests for the weighted non-uniform Fourier transforms (`layers._nonuniform`)."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from geonnax.layers import nonuniform_irfft, nonuniform_rfft
from geonnax.layers._utils import _corner_freqs, _corner_slices


KEY = jr.PRNGKey(0)

# Per-dimension (grid size, retained modes) used by the grid-bridge tests.
GRIDS = {1: ((32,), (5,)), 2: ((16, 12), (4, 3)), 3: ((8, 8, 6), (3, 2, 2))}


def _nodes(grid: tuple[int, ...]) -> jax.Array:
    """Uniform grid nodes ``j / s`` in C order, shape ``(S, d)``."""
    axes = [np.arange(s) / s for s in grid]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return jnp.asarray(mesh.reshape(-1, len(grid)), dtype=jnp.float32)


def _flatten(u: jax.Array) -> jax.Array:
    """``(C, *grid)`` → ``(C, S)`` in the same C order as `_nodes`."""
    spatial = " ".join(f"s{a}" for a in range(u.ndim - 1))
    return einx.id(f"c {spatial} -> c ({spatial})", u)


def _rfft_corners(u: jax.Array, n_modes: tuple[int, ...]) -> list[jax.Array]:
    """``rfftn(u) / S`` restricted to the corner blocks."""
    axes = tuple(range(1, u.ndim))
    u_ft = jnp.fft.rfftn(u, axes=axes) / math.prod(u.shape[1:])
    return [u_ft[(slice(None), *sl)] for sl in _corner_slices(n_modes)]


# ---------------------------------------------------------------------------
# Corner layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_modes", [(4,), (3, 2), (2, 3, 2)])
def test_corner_freqs_match_corner_slices(n_modes):
    """`_corner_freqs[c]` are exactly the wavenumbers `_corner_slices[c]` selects."""
    shape = (*(2 * m + 2 for m in n_modes[:-1]), n_modes[-1] + 1)
    per_axis = [np.fft.fftfreq(s, 1.0 / s).astype(int) for s in shape[:-1]]
    per_axis.append(np.arange(shape[-1]))
    wavenumbers = np.stack(np.meshgrid(*per_axis, indexing="ij"), axis=-1)
    slices = _corner_slices(n_modes)
    freqs = _corner_freqs(n_modes)
    assert len(freqs) == len(slices) == 2 ** (len(n_modes) - 1)
    for sl, fr in zip(slices, freqs, strict=True):
        expected = np.stack(np.meshgrid(*fr, indexing="ij"), axis=-1)
        np.testing.assert_array_equal(wavenumbers[sl], expected)


# ---------------------------------------------------------------------------
# Grid bridge, round trip, padding, linearity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_nonuniform_rfft_grid_bridge(nd):
    """On grid nodes with w = 1/S the transform is ``rfftn(u) / S`` on the corners."""
    grid, n_modes = GRIDS[nd]
    s = math.prod(grid)
    u = jr.normal(KEY, (2, *grid))
    blocks = nonuniform_rfft(
        _flatten(u), _nodes(grid), jnp.full((s,), 1.0 / s), n_modes
    )
    for got, ref in zip(blocks, _rfft_corners(u, n_modes), strict=True):
        assert got.shape == (2, *n_modes)
        np.testing.assert_allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_nonuniform_irfft_matches_irfftn_on_grid(nd):
    """At grid nodes the inverse equals ``irfftn`` of the scattered corners × S."""
    grid, n_modes = GRIDS[nd]
    s = math.prod(grid)
    keys = jr.split(KEY, 2 ** (nd - 1))
    # Random, not Hermitian-consistent (irfftn's handling must match anyway),
    # scaled so the field is O(1) and atol=1e-5 is a float32 tolerance.
    scale = 1.0 / math.sqrt(2 * len(keys) * math.prod(n_modes))
    blocks = [scale * jax.lax.complex(*jr.normal(k, (2, 3, *n_modes))) for k in keys]
    spectrum = jnp.zeros((3, *grid[:-1], grid[-1] // 2 + 1), dtype=jnp.complex64)
    for sl, b in zip(_corner_slices(n_modes), blocks, strict=True):
        spectrum = spectrum.at[(slice(None), *sl)].set(s * b)
    ref = jnp.fft.irfftn(spectrum, s=grid, axes=tuple(range(1, nd + 1)))
    got = nonuniform_irfft(blocks, _nodes(grid), n_modes)
    assert got.shape == (3, s)
    np.testing.assert_allclose(got, _flatten(ref), atol=1e-5)


def _trig_poly(xi: jax.Array) -> jax.Array:
    """A real band-limited field with modes inside n_modes = (4, 4); shape (2, N)."""
    k = jnp.asarray([[0, 0], [1, 2], [-3, 1], [2, 0], [3, 3], [-1, 3]], jnp.float32)
    amp = jnp.asarray([0.7, 1.0, -0.5, 0.8, 0.3, 0.6])
    phi = jnp.asarray([0.0, 0.3, 1.1, -0.4, 2.0, 0.9])
    arg = 2 * jnp.pi * einx.dot("n d, j d -> n j", xi, k) + phi
    first = einx.dot("n j, j -> n", jnp.cos(arg), amp)
    second = einx.dot("n j, j -> n", jnp.sin(arg), amp[::-1])
    return jnp.stack([first, second])


def test_round_trip_band_limited():
    """rfft on a grid then irfft at random points recovers a band-limited field."""
    grid, n_modes = (16, 16), (4, 4)
    nodes = _nodes(grid)
    s = math.prod(grid)
    blocks = nonuniform_rfft(_trig_poly(nodes), nodes, jnp.full((s,), 1.0 / s), n_modes)
    query = jr.uniform(KEY, (200, 2))
    got = nonuniform_irfft(blocks, query, n_modes)
    np.testing.assert_allclose(got, _trig_poly(query), atol=1e-5)


def test_zero_weight_padding_is_bit_identical():
    """Points with w = 0 contribute nothing: 300 points padded to 500 match exactly."""
    k1, k2, k3 = jr.split(KEY, 3)
    xi = jr.uniform(k1, (300, 2))
    u = jr.normal(k2, (3, 300))
    w = jr.uniform(k3, (300,)) / 300
    pad = 200
    xi_p = jnp.concatenate([xi, jnp.zeros((pad, 2))])
    u_p = jnp.concatenate([u, jnp.ones((3, pad))], axis=-1)
    w_p = jnp.concatenate([w, jnp.zeros((pad,))])
    ref = nonuniform_rfft(u, xi, w, (4, 4))
    got = nonuniform_rfft(u_p, xi_p, w_p, (4, 4))
    for a, b in zip(got, ref, strict=True):
        assert jnp.array_equal(a, b)


def test_weights_are_not_normalised():
    """Scaling every weight by α scales the coefficients by α."""
    k1, k2 = jr.split(KEY)
    xi = jr.uniform(k1, (64, 2))
    u = jr.normal(k2, (1, 64))
    w = jnp.full((64,), 1.0 / 64)
    ref = nonuniform_rfft(u, xi, w, (3, 3))
    got = nonuniform_rfft(u, xi, 3.0 * w, (3, 3))
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_allclose(a, 3.0 * b, rtol=1e-5, atol=1e-6)


def _jittered(key: jax.Array, s: int) -> jax.Array:
    """One uniform point per cell of an ``s × s`` grid, shape ``(s², 2)``."""
    return _nodes((s, s)) + jr.uniform(key, (s * s, 2)) / s


def test_convergence_with_n_for_jittered_points():
    """Stratified (jittered) quadrature with w = 1/N converges as N grows.

    The field is smooth but not band-limited, so the coefficients are a true
    quadrature. Jittered sampling converges ~N^-1 (the #75 prototype measured
    0.39 → 0.024 over N ≈ 500 → 8000); a 16× increase in N must cut the error
    by at least 4×, a loose bound well inside that rate.
    """
    n_modes = (4, 4)

    def field(xi):
        return jnp.exp(jnp.sin(2 * jnp.pi * xi[:, 0]) * jnp.cos(2 * jnp.pi * xi[:, 1]))[
            None
        ]

    fine = _nodes((128, 128))
    ref = nonuniform_rfft(field(fine), fine, jnp.full((128 * 128,), 128.0**-2), n_modes)
    ref_norm = math.sqrt(sum(float(jnp.sum(jnp.abs(r) ** 2)) for r in ref))

    def rel_err(s, key):
        xi = _jittered(key, s)
        got = nonuniform_rfft(field(xi), xi, jnp.full((s * s,), 1.0 / s**2), n_modes)
        sq = sum(
            float(jnp.sum(jnp.abs(g - r) ** 2)) for g, r in zip(got, ref, strict=True)
        )
        return math.sqrt(sq) / ref_norm

    k1, k2 = jr.split(KEY)
    coarse, finer = rel_err(16, k1), rel_err(64, k2)
    assert finer < 0.25 * coarse
    assert finer < 0.05


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_nonuniform_rfft_validation():
    xi = jnp.zeros((10, 2))
    u = jnp.zeros((1, 10))
    w = jnp.zeros((10,))
    with pytest.raises(ValueError, match="n_modes must be non-empty"):
        nonuniform_rfft(u, xi, w, ())
    with pytest.raises(ValueError, match=">= 1"):
        nonuniform_rfft(u, xi, w, (0, 2))
    with pytest.raises(ValueError, match="xi must have shape"):
        nonuniform_rfft(u, xi, w, (2, 2, 2))
    with pytest.raises(ValueError, match="u must have shape"):
        nonuniform_rfft(jnp.zeros((1, 9)), xi, w, (2, 2))
    with pytest.raises(ValueError, match="weights must have shape"):
        nonuniform_rfft(u, xi, jnp.zeros((9,)), (2, 2))


def test_nonuniform_irfft_validation():
    xi = jnp.zeros((10, 2))
    block = jnp.zeros((1, 2, 2), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="2\\*\\*\\(d-1\\)"):
        nonuniform_irfft([block], xi, (2, 2))
    with pytest.raises(ValueError, match="corner block must have shape"):
        nonuniform_irfft([block, jnp.zeros((1, 3, 2), jnp.complex64)], xi, (2, 2))
    with pytest.raises(ValueError, match="xi must have shape"):
        nonuniform_irfft([block, block], jnp.zeros((10, 3)), (2, 2))


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_nonuniform_jit_vmap_grad():
    """jit, vmap over a batch of padded clouds, and finite gradients w.r.t. xi."""
    n_modes = (3, 3)
    b, n = 4, 50
    k1, k2, k3 = jr.split(KEY, 3)
    xi = jr.uniform(k1, (b, n, 2))
    u = jr.normal(k2, (b, 2, n))
    w = jnp.where(jnp.arange(n) < 40, 1.0 / 40, 0.0)
    w = einx.multiply("n, b -> b n", w, jnp.ones((b,)))
    query = jr.uniform(k3, (b, 30, 2))

    @jax.jit
    def roundtrip(u, xi, w, q):
        return nonuniform_irfft(nonuniform_rfft(u, xi, w, n_modes), q, n_modes)

    out = jax.vmap(roundtrip)(u, xi, w, query)
    assert out.shape == (b, 2, 30)
    eager = nonuniform_irfft(
        nonuniform_rfft(u[0], xi[0], w[0], n_modes), query[0], n_modes
    )
    np.testing.assert_allclose(out[0], eager, atol=1e-5)

    def loss(xi_, q_):
        return jnp.sum(roundtrip(u[0], xi_, w[0], q_) ** 2)

    g_xi, g_q = jax.grad(loss, argnums=(0, 1))(xi[0], query[0])
    assert bool(jnp.all(jnp.isfinite(g_xi))) and bool(jnp.all(jnp.isfinite(g_q)))
    assert float(jnp.abs(g_xi).max()) > 0.0
