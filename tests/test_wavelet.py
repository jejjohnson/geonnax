"""Tests for the discrete wavelet transform, WaveletConv, and the WNO."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.layers._wavelet import _filter_bank, dwt, idwt
from geonnax.wno import WNO, WaveletNeuralOperator, WNOBlock


KEY = jr.PRNGKey(0)
WAVELETS = [
    "haar",
    "db2",
    "db3",
    "db4",
    "db5",
    "sym4",
    "sym5",
    "sym6",
    "coif1",
    "coif2",
]


def _grads_finite(model, *args):
    def loss(m, *a):
        return jnp.mean(m(*a) ** 2)

    grads = eqx.filter_grad(loss)(model, *args)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    return bool(leaves) and all(jnp.isfinite(g).all() for g in leaves)


# ---- filters & transform ----------------------------------------------------


@pytest.mark.parametrize("wavelet", WAVELETS)
def test_filters_are_orthonormal(wavelet):
    dec_lo, dec_hi = _filter_bank(wavelet)
    # Orthonormal scaling filter: unit L2 norm and sum sqrt(2).
    assert abs(float((dec_lo**2).sum()) - 1.0) < 1e-6
    assert abs(float(dec_lo.sum()) - 2.0**0.5) < 1e-5
    # High-pass is orthogonal to the low-pass.
    assert abs(float((dec_lo * dec_hi).sum())) < 1e-6


def test_filter_bank_rejects_unknown():
    with pytest.raises(ValueError, match="unknown wavelet"):
        _filter_bank("bogus")


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("nd,shape", [(1, (3, 64)), (2, (2, 32, 32))])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_perfect_reconstruction(wavelet, nd, shape, level):
    axes = tuple(range(1, nd + 1))
    x = jr.normal(KEY, shape)
    rec = idwt(dwt(x, wavelet, level, axes), wavelet, axes)
    assert float(jnp.abs(rec - x).max()) < 1e-4


def test_perfect_reconstruction_3d():
    x = jr.normal(KEY, (2, 16, 16, 16))
    rec = idwt(dwt(x, "db2", 2, (1, 2, 3)), "db2", (1, 2, 3))
    assert float(jnp.abs(rec - x).max()) < 1e-4


def test_dwt_subband_shapes():
    x = jnp.ones((2, 32, 32))
    approx, details = dwt(x, "haar", 2, (1, 2))
    assert approx.shape == (2, 8, 8)  # halved twice
    assert len(details) == 2
    assert set(details[0]) == {("a", "d"), ("d", "a"), ("d", "d")}  # 3 detail bands


# ---- WaveletConv ------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", WAVELETS)
def test_wavelet_conv_shapes(wavelet):
    conv = geonnax.WaveletConv.init(4, 6, 2, key=KEY, wavelet=wavelet, level=2)
    assert conv(jnp.ones((4, 32, 32))).shape == (6, 32, 32)


def test_wavelet_conv_resolution_flexible():
    conv = geonnax.WaveletConv.init(2, 2, 2, key=KEY, wavelet="db4", level=2)
    assert conv(jnp.ones((2, 32, 32))).shape == (2, 32, 32)
    assert conv(jnp.ones((2, 64, 64))).shape == (2, 64, 64)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_wavelet_conv_dimension_flexible(nd):
    conv = geonnax.WaveletConv.init(2, 3, nd, key=KEY, wavelet="db2", level=1)
    x = jnp.ones((2,) + (16,) * nd)
    assert conv(x).shape == (3,) + (16,) * nd


@pytest.mark.integration
def test_wavelet_conv_grad_and_jit():
    conv = geonnax.WaveletConv.init(2, 2, 2, key=KEY, level=2)
    x = jnp.ones((2, 32, 32))
    assert _grads_finite(conv, x)
    assert eqx.filter_jit(conv)(x).shape == (2, 32, 32)


def test_wavelet_conv_rejects_bad_args():
    with pytest.raises(ValueError, match="level must be"):
        geonnax.WaveletConv.init(2, 2, 2, key=KEY, level=0)
    with pytest.raises(ValueError, match="num_spatial_dims must be"):
        geonnax.WaveletConv.init(2, 2, 0, key=KEY)


# ---- WNO --------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", WAVELETS)
def test_wno_wavelets(wavelet):
    m = WNO.init(
        3, 1, 2, key=KEY, hidden_channels=16, n_layers=3, wavelet=wavelet, level=2
    )
    assert m(jnp.ones((3, 64, 64))).shape == (1, 64, 64)


@pytest.mark.integration
def test_wno_vmap_jit_grad():
    m = WNO.init(2, 2, 2, key=KEY, hidden_channels=16, n_layers=2)
    x = jnp.ones((2, 32, 32))
    assert jax.vmap(m)(jnp.ones((4, 2, 32, 32))).shape == (4, 2, 32, 32)
    assert eqx.filter_jit(m)(x).shape == (2, 32, 32)
    assert _grads_finite(m, x)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_wno_dimension_flexible(nd):
    m = WNO.init(2, 3, nd, key=KEY, hidden_channels=8, n_layers=2, level=1)
    x = jnp.ones((2,) + (16,) * nd)
    assert m(x).shape == (3,) + (16,) * nd


def test_wno_block_shape():
    blk = WNOBlock.init(8, 2, key=KEY, wavelet="db4", level=2)
    assert blk(jnp.ones((8, 32, 32))).shape == (8, 32, 32)


def test_wno_rejects_bad_args():
    with pytest.raises(ValueError, match="n_layers must be"):
        WNO.init(1, 1, 2, key=KEY, n_layers=0)
    with pytest.raises(ValueError, match="num_spatial_dims must be"):
        WNO.init(1, 1, 0, key=KEY)


def test_wno_alias_and_exports():
    assert WaveletNeuralOperator is WNO
    assert geonnax.WNO is WNO
    for name in ["WNO", "WNOBlock", "WaveletNeuralOperator", "WaveletConv"]:
        assert hasattr(geonnax, name), name
    assert hasattr(geonnax, "wno")
