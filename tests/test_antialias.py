"""Tests for band-limited resampling and the anti-aliased activation."""

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.layers import AntiAliasedActivation, band_limited_resize


KEY = jr.PRNGKey(0)
ND = [1, 2, pytest.param(3, marks=pytest.mark.slow)]


def _axes(nd):
    return " ".join(f"s{a}" for a in range(nd))


def _trig_field(key, grid, max_k, *, channels=2, n_terms=6):
    """A random real trigonometric polynomial with ``|k_a| <= max_k``, sampled on
    the uniform grid ``j / grid`` of ``[0, 1)^d``; the same ``key`` gives the same
    function on any grid, so it doubles as the analytic reference."""
    nd = len(grid)
    k_freq, k_cos, k_sin = jr.split(key, 3)
    freqs = jr.randint(k_freq, (n_terms, nd), -max_k, max_k + 1)  # (T, d)
    a = jr.normal(k_cos, (channels, n_terms))  # (C, T)
    b = jr.normal(k_sin, (channels, n_terms))  # (C, T)
    mesh = jnp.stack(jnp.meshgrid(*[jnp.arange(s) / s for s in grid], indexing="ij"))
    pts = einx.id(f"d {_axes(nd)} -> ({_axes(nd)}) d", mesh)  # (P, d)
    phase = 2 * jnp.pi * einx.dot("p d, t d -> p t", pts, freqs.astype(pts.dtype))
    vals = einx.dot("p t, c t -> c p", jnp.cos(phase), a) + einx.dot(
        "p t, c t -> c p", jnp.sin(phase), b
    )
    sizes = {f"s{i}": s for i, s in enumerate(grid)}
    return einx.id(f"c ({_axes(nd)}) -> c {_axes(nd)}", vals, **sizes)


def _rel_err(y, ref):
    return float(jnp.linalg.norm(y - ref) / jnp.linalg.norm(ref))


# ---- band_limited_resize ----------------------------------------------------


@pytest.mark.parametrize("nd", ND)
def test_resize_shapes(nd):
    x = jnp.ones((3,) + (8,) * nd)
    out = tuple(range(5, 5 + nd))
    assert band_limited_resize(x, out).shape == (3, *out)


@pytest.mark.parametrize("nd", ND)
def test_resize_up_then_down_is_identity(nd):
    # |k| <= 3 is below the Nyquist of every grid here, so nothing is lost.
    grid = (12, 9, 10)[:nd]
    x = _trig_field(KEY, grid, 3)
    up = band_limited_resize(x, tuple(2 * s + 1 for s in grid))
    assert jnp.allclose(band_limited_resize(up, grid), x, atol=1e-5)


@pytest.mark.parametrize("nd", ND)
def test_resize_matches_analytic_field(nd):
    # Mixed up / down / odd extents; |k| <= 4 < min(N, M) / 2 on every axis.
    grid_in = (16, 20, 11)[:nd]
    grid_out = (40, 10, 18)[:nd]
    x = _trig_field(KEY, grid_in, 4)
    y = band_limited_resize(x, grid_out)
    assert jnp.allclose(y, _trig_field(KEY, grid_out, 4), atol=2e-5)


def test_resize_same_grid_is_identity():
    x = jr.normal(KEY, (2, 8, 6))
    assert band_limited_resize(x, (8, 6)) is x


def test_resize_down_then_up_is_a_low_pass():
    # White noise loses its high modes: the round trip is a projection
    # (idempotent, strictly smaller norm), not the identity.
    x = jr.normal(KEY, (1, 64))
    y = band_limited_resize(band_limited_resize(x, (16,)), (64,))
    assert float(jnp.max(jnp.abs(y - x))) > 0.5
    assert float(jnp.linalg.norm(y)) < 0.6 * float(jnp.linalg.norm(x))
    z = band_limited_resize(band_limited_resize(y, (16,)), (64,))
    assert jnp.allclose(z, y, atol=1e-5)
    # ...and keeps exactly the modes |k| < 8.
    spec = jnp.abs(jnp.fft.rfft(y[0]))
    assert jnp.allclose(spec[:8], jnp.abs(jnp.fft.rfft(x[0]))[:8], atol=1e-4)
    assert float(jnp.max(spec[8:])) < 1e-4


def test_resize_drops_the_nyquist_bin():
    # cos(pi j) is the Nyquist mode of an even grid; it has no partner on a
    # finer grid, so a band-limited resize removes it.
    x = jnp.cos(jnp.pi * jnp.arange(8.0))[None]
    assert jnp.allclose(band_limited_resize(x, (16,)), 0.0, atol=1e-6)


def test_resize_validation():
    with pytest.raises(ValueError, match="out_spatial must have 2"):
        band_limited_resize(jnp.ones((1, 8, 8)), (16,))
    with pytest.raises(ValueError, match=">= 1"):
        band_limited_resize(jnp.ones((1, 8)), (0,))
    with pytest.raises(ValueError, match="shape"):
        band_limited_resize(jnp.ones((4,)), ())


# ---- AntiAliasedActivation --------------------------------------------------


@pytest.mark.parametrize("nd", ND)
def test_activation_shapes(nd):
    act = AntiAliasedActivation.init(nd)
    x = jr.normal(KEY, (3,) + (8,) * nd)
    assert act(x).shape == x.shape
    assert act(x).dtype == x.dtype


def test_activation_up_one_is_pointwise():
    x = jr.normal(KEY, (2, 16, 12))
    act = AntiAliasedActivation.init(2, activation=jnp.tanh, up=1)
    assert jnp.allclose(act(x), jnp.tanh(x), atol=1e-6)


@pytest.mark.parametrize(
    ("activation", "nd", "n", "max_k", "factor"),
    [
        # Measured in float32: gelu 1-D 2.5e-2 -> 3.8e-4 (65x), 2-D 0.10 ->
        # 4.2e-3 (25x); relu 1-D 2.7e-2 -> 5.6e-3 (5x, its kink decays slowly).
        (jax.nn.gelu, 1, 64, 16, 20.0),
        (jax.nn.gelu, 2, 32, 8, 10.0),
        (jax.nn.relu, 1, 64, 16, 3.0),
    ],
)
def test_activation_reduces_aliasing(activation, nd, n, max_k, factor):
    """The output is much closer to the band-limited projection of the
    continuous sigma(u) than the naive pointwise sigma on the grid.

    The reference is P_{n/2} sigma(u), with sigma evaluated on a 16x finer grid
    (where the aliased harmonics are negligible) and low-passed back.
    """
    grid = (n,) * nd
    u = _trig_field(KEY, grid, max_k, channels=1, n_terms=8)
    u = 2.0 * u / jnp.std(u)  # O(1) amplitude so the nonlinearity bites
    fine = tuple(16 * s for s in grid)
    ideal = band_limited_resize(activation(band_limited_resize(u, fine)), grid)
    err_naive = _rel_err(activation(u), ideal)
    err_up2 = _rel_err(AntiAliasedActivation.init(nd, activation=activation)(u), ideal)
    err_up4 = _rel_err(
        AntiAliasedActivation.init(nd, activation=activation, up=4)(u), ideal
    )
    assert err_naive > 1e-2
    assert err_up2 < err_naive / factor
    assert err_up4 < err_up2


def test_activation_output_has_no_nyquist_energy():
    # The output spectrum lies strictly inside |k| < N/2; the naive one does not.
    u = _trig_field(KEY, (32,), 12, channels=1)
    out = AntiAliasedActivation.init(1)(u)
    assert float(jnp.abs(jnp.fft.rfft(out[0])[16])) < 1e-4
    assert float(jnp.abs(jnp.fft.rfft(jax.nn.gelu(u[0]))[16])) > 1e-3


def test_activation_validation():
    with pytest.raises(ValueError, match="num_spatial_dims"):
        AntiAliasedActivation.init(0)
    with pytest.raises(ValueError, match="up must be"):
        AntiAliasedActivation.init(2, up=0)
    with pytest.raises(ValueError, match="2 spatial axes"):
        AntiAliasedActivation.init(2)(jnp.ones((3, 8)))


def test_top_level_exports():
    assert geonnax.AntiAliasedActivation is AntiAliasedActivation
    assert geonnax.band_limited_resize is band_limited_resize


# ---- jit / vmap / grad ------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("nd", ND)
def test_jit_vmap_grad(nd):
    act = AntiAliasedActivation.init(nd)
    xs = jr.normal(KEY, (4, 2) + (8,) * nd)
    out = eqx.filter_jit(jax.vmap(act))(xs)
    assert out.shape == xs.shape
    assert jnp.allclose(out[1], act(xs[1]), atol=1e-5)

    up = (12,) * nd
    resized = jax.jit(jax.vmap(lambda x: band_limited_resize(x, up)))(xs)
    assert resized.shape == (4, 2, *up)

    def loss(x):
        return jnp.mean(band_limited_resize(act(x), up) ** 2)

    g = eqx.filter_grad(loss)(xs[0])
    assert g.shape == xs[0].shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0
