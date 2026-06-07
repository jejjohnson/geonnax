"""Tests for WaveletAttention and the Swin-style WindowedAttention."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.layers import Attention, WaveletAttention, WindowedAttention


KEY = jr.PRNGKey(0)
WAVELETS = ["haar", "db2", "db4", "sym6", "coif1"]


def _grads_finite(model, x):
    # Random (non-constant) input: a constant field has zero GroupNorm variance.
    grads = eqx.filter_grad(lambda m, x: jnp.mean(m(x) ** 2))(model, x)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    return bool(leaves) and all(jnp.isfinite(g).all() for g in leaves)


# ---- WindowedAttention ------------------------------------------------------


@pytest.mark.parametrize(
    "nd,shape,window",
    [(1, (16, 16), 4), (2, (16, 16, 16), 4), (2, (16, 12, 12), 4)],
)
def test_windowed_attention_shape(nd, shape, window):
    attn = WindowedAttention.init(
        shape[0], nd, window=window, key=KEY, heads=2, dim_head=8
    )
    assert attn(jnp.ones(shape)).shape == shape  # incl. 12 not a multiple of 4


def test_windowed_attention_differs_from_global():
    x = jr.normal(KEY, (16, 16, 16))
    g = Attention.init(16, 2, key=jr.PRNGKey(1), heads=2, dim_head=8)
    w = WindowedAttention.init(16, 2, window=4, key=jr.PRNGKey(1), heads=2, dim_head=8)
    assert not jnp.allclose(g(x), w(x), atol=1e-4)


def test_windowed_attention_rejects_bad_window():
    with pytest.raises(ValueError, match="window must be"):
        WindowedAttention.init(8, 2, window=0, key=KEY)


# ---- WaveletAttention -------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", WAVELETS)
def test_wavelet_attention_shape_all_wavelets(wavelet):
    wa = WaveletAttention.init(16, 2, key=KEY, wavelet=wavelet)
    assert wa(jnp.ones((16, 32, 32))).shape == (16, 32, 32)


def test_wavelet_attention_resolution_flexible():
    wa = WaveletAttention.init(16, 2, key=KEY)
    assert wa(jnp.ones((16, 32, 32))).shape == (16, 32, 32)
    assert wa(jnp.ones((16, 64, 64))).shape == (16, 64, 64)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_wavelet_attention_dimension_flexible(nd):
    dim = 2 ** (nd + 2)  # divisible by 2**nd
    wa = WaveletAttention.init(dim, nd, key=KEY, dim_head=8, heads=2)
    x = jnp.ones((dim, *([16] * nd)))
    assert wa(x).shape == x.shape


def test_wavelet_attention_windowed():
    wa = WaveletAttention.init(16, 2, key=KEY, window=4)
    assert wa(jnp.ones((16, 64, 64))).shape == (16, 64, 64)


def test_wavelet_attention_even_extent_guard():
    wa = WaveletAttention.init(16, 2, key=KEY)
    with pytest.raises(ValueError, match="even spatial extents"):
        wa(jnp.ones((16, 31, 32)))


def test_wavelet_attention_rejects_bad_args():
    with pytest.raises(ValueError, match="divisible by"):
        WaveletAttention.init(18, 2, key=KEY)  # 18 % 4 != 0
    with pytest.raises(ValueError, match="num_spatial_dims must be"):
        WaveletAttention.init(16, 0, key=KEY)


@pytest.mark.integration
def test_wavelet_attention_vmap_jit_grad():
    wa = WaveletAttention.init(16, 2, key=KEY)
    x = jr.normal(jr.PRNGKey(3), (16, 32, 32))
    assert jax.vmap(wa)(jr.normal(KEY, (3, 16, 32, 32))).shape == (3, 16, 32, 32)
    assert eqx.filter_jit(wa)(x).shape == (16, 32, 32)
    assert _grads_finite(wa, x)


def test_exports():
    for name in ["WaveletAttention", "WindowedAttention"]:
        assert hasattr(geonnax, name), name
        assert name in geonnax.layers.__all__
