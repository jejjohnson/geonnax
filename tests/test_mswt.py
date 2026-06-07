"""Tests for the Multi-Scale Wavelet Transformer (MSWT) operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.layers._wavelet import _flatten, dwt
from geonnax.mswt import (
    MSWT,
    MSWTBlock,
    MultiScaleWaveletTransformer,
    WaveletDownsample,
    WaveletUpsample,
    _level1_layout,
)


KEY = jr.PRNGKey(0)
WAVELETS = ["haar", "db2", "db4", "sym6"]


def _grads_finite(model, x):
    grads = eqx.filter_grad(lambda m, x: jnp.mean(m(x) ** 2))(model, x)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    return bool(leaves) and all(jnp.isfinite(g).all() for g in leaves)


# ---- sub-band plumbing ------------------------------------------------------


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_level1_layout_matches_flatten(nd):
    # The hand-built layout must match what _flatten produces, or up-sampling
    # would reassemble the sub-bands in the wrong order.
    x = jr.normal(KEY, (2, *([4] * nd)))
    _, layout = _flatten(dwt(x, "haar", 1, tuple(range(1, nd + 1))))
    assert layout == _level1_layout(nd)


@pytest.mark.parametrize("nd,shape", [(1, (8, 16)), (2, (8, 16, 16))])
def test_wavelet_down_up_shapes(nd, shape):
    down = WaveletDownsample.init(8, nd, key=KEY)
    up = WaveletUpsample.init(8, nd, key=jr.PRNGKey(1))
    half = down(jnp.ones(shape))
    assert half.shape[1:] == tuple(s // 2 for s in shape[1:])
    assert up(half).shape == shape


# ---- MSWT model -------------------------------------------------------------


def test_mswt_forward_changes_channels():
    op = MSWT.init(3, 1, 2, key=KEY, hidden_channels=16, depth=2, n_blocks=1)
    assert op(jnp.ones((3, 32, 32))).shape == (1, 32, 32)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", WAVELETS)
def test_mswt_wavelets(wavelet):
    op = MSWT.init(
        2, 2, 2, key=KEY, hidden_channels=16, depth=1, n_blocks=1, wavelet=wavelet
    )
    assert op(jnp.ones((2, 32, 32))).shape == (2, 32, 32)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_mswt_dimension_flexible(nd):
    op = MSWT.init(2, 3, nd, key=KEY, hidden_channels=8, depth=1, n_blocks=1)
    x = jnp.ones((2, *([16] * nd)))
    assert op(x).shape == (3, *([16] * nd))


def test_mswt_windowed():
    op = MSWT.init(3, 1, 2, key=KEY, hidden_channels=16, depth=1, window=4)
    assert op(jnp.ones((3, 32, 32))).shape == (1, 32, 32)


def test_mswt_patch_size():
    op = MSWT.init(3, 1, 2, key=KEY, hidden_channels=16, depth=1, patch_size=2)
    assert op(jnp.ones((3, 32, 32))).shape == (1, 32, 32)


def test_mswt_depth_zero_is_flat():
    op = MSWT.init(2, 2, 2, key=KEY, hidden_channels=16, depth=0, n_blocks=2)
    assert op(jnp.ones((2, 16, 16))).shape == (2, 16, 16)


@pytest.mark.integration
def test_mswt_vmap_jit_grad():
    op = MSWT.init(2, 2, 2, key=KEY, hidden_channels=16, depth=1, n_blocks=1)
    x = jr.normal(jr.PRNGKey(4), (2, 32, 32))
    assert jax.vmap(op)(jr.normal(KEY, (3, 2, 32, 32))).shape == (3, 2, 32, 32)
    assert eqx.filter_jit(op)(x).shape == (2, 32, 32)
    assert _grads_finite(op, x)


def test_mswt_divisibility_guard():
    op = MSWT.init(3, 1, 2, key=KEY, hidden_channels=16, depth=2)
    # depth=2 needs extents divisible by 2**3 = 8; 12 is not.
    with pytest.raises(ValueError, match="divisible by"):
        op(jnp.ones((3, 12, 12)))


def test_mswt_rejects_bad_args():
    with pytest.raises(ValueError, match="hidden_channels must be divisible"):
        MSWT.init(3, 1, 2, key=KEY, hidden_channels=18)  # 18 % 4 != 0
    with pytest.raises(ValueError, match="n_blocks must be"):
        MSWT.init(3, 1, 2, key=KEY, n_blocks=0)
    with pytest.raises(ValueError, match="depth must be"):
        MSWT.init(3, 1, 2, key=KEY, depth=-1)
    with pytest.raises(ValueError, match="num_spatial_dims must be"):
        MSWT.init(3, 1, 0, key=KEY)


def test_mswt_block_shape():
    block = MSWTBlock.init(16, 2, key=KEY)
    assert block(jnp.ones((16, 16, 16))).shape == (16, 16, 16)


def test_mswt_alias_and_exports():
    assert MultiScaleWaveletTransformer is MSWT
    assert geonnax.MSWT is MSWT
    for name in ["MSWT", "MSWTBlock", "MultiScaleWaveletTransformer"]:
        assert hasattr(geonnax, name), name
    assert hasattr(geonnax, "mswt")
