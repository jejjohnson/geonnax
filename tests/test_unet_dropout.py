"""Tests for threading dropout (Monte-Carlo dropout) through the U-Net."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from geonnax import UNet
from geonnax.layers import ConvNeXtBlock, ResnetBlock
from geonnax.unet import NestedResidualUNet


KEY = jr.PRNGKey(0)


def _grads_finite(model, *args, **kwargs):
    def loss(m):
        return jnp.mean(m(*args, **kwargs) ** 2)

    grads = eqx.filter_grad(loss)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    return bool(leaves) and all(jnp.isfinite(g).all() for g in leaves)


# ---- backward compatibility -------------------------------------------------


def test_no_dropout_callable_without_key():
    net = UNet.init(8, key=KEY, channels=3, out_channels=1, dim_mults=(1, 2))
    assert net(jnp.ones((3, 16, 16))).shape == (1, 16, 16)
    # An explicit key is harmlessly accepted (and ignored) when dropout is off.
    assert net(jnp.ones((3, 16, 16)), key=KEY).shape == (1, 16, 16)


def test_no_dropout_is_deterministic():
    net = UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2))
    x = jnp.ones((2, 16, 16))
    assert jnp.allclose(net(x, key=jr.PRNGKey(1)), net(x, key=jr.PRNGKey(2)))


# ---- dropout enabled --------------------------------------------------------


def test_dropout_requires_key():
    net = UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2), dropout=0.3)
    with pytest.raises(RuntimeError):
        net(jnp.ones((2, 16, 16)))  # active dropout without a key


def test_dropout_is_stochastic_but_key_reproducible():
    net = UNet.init(
        8, key=KEY, channels=2, out_channels=1, dim_mults=(1, 2), dropout=0.3
    )
    x = jnp.ones((2, 16, 16))
    ya, yb = net(x, key=jr.PRNGKey(1)), net(x, key=jr.PRNGKey(2))
    assert float(jnp.abs(ya - yb).max()) > 1e-3  # different keys differ
    assert jnp.allclose(net(x, key=jr.PRNGKey(1)), ya)  # same key reproduces


def test_inference_mode_disables_dropout_without_key():
    net = UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2), dropout=0.3)
    net_eval = eqx.nn.inference_mode(net)
    x = jnp.ones((2, 16, 16))
    assert jnp.allclose(net_eval(x), net_eval(x))  # deterministic, no key needed


def test_mc_dropout_ensemble_has_spread():
    net = UNet.init(
        8, key=KEY, channels=2, out_channels=1, dim_mults=(1, 2), dropout=0.3
    )
    x = jnp.ones((2, 16, 16))
    ensemble = jax.vmap(lambda k: net(x, key=k))(jr.split(KEY, 8))
    assert ensemble.shape == (8, 1, 16, 16)
    assert float(ensemble.std(axis=0).mean()) > 1e-4


def test_dropout_grad_and_jit():
    net = UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2), dropout=0.2)
    x = jnp.ones((2, 16, 16))
    assert _grads_finite(net, x, key=KEY)
    assert eqx.filter_jit(net)(x, key=KEY).shape == (2, 16, 16)


@pytest.mark.parametrize("block_type", ["resnet", "convnext"])
def test_dropout_with_block_types_and_nesting(block_type):
    net = UNet.init(
        8,
        key=KEY,
        channels=2,
        dim_mults=(1, 2),
        block_type=block_type,
        nested_unet_depths=(2, 1),
        dropout=0.1,
    )
    assert net(jnp.ones((2, 16, 16)), key=KEY).shape == (2, 16, 16)


# ---- block-level ------------------------------------------------------------


def test_nested_residual_unet_dropout():
    block = NestedResidualUNet.init(
        8, depth=2, num_spatial_dims=2, key=KEY, dropout=0.2
    )
    x = jnp.ones((8, 16, 16))
    assert block(x, key=jr.PRNGKey(1)).shape == (8, 16, 16)
    with pytest.raises(RuntimeError):
        block(x)


@pytest.mark.parametrize("block_cls", [ResnetBlock, ConvNeXtBlock])
def test_invalid_dropout_rate_rejected(block_cls):
    # A negative (or >= 1) rate must raise, not silently disable dropout.
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="dropout rate must be in"):
            block_cls.init(4, 4, 2, key=KEY, dropout=bad)
    # The U-Net surfaces the same validation through its blocks.
    with pytest.raises(ValueError, match="dropout rate must be in"):
        UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2), dropout=-0.1)


@pytest.mark.parametrize("block_cls", [ResnetBlock, ConvNeXtBlock])
def test_block_dropout_requires_key_and_is_stochastic(block_cls):
    block = block_cls.init(4, 4, 2, key=KEY, dropout=0.5)
    x = jnp.ones((4, 8, 8))
    with pytest.raises(RuntimeError):
        block(x)
    ya, yb = block(x, key=jr.PRNGKey(1)), block(x, key=jr.PRNGKey(2))
    assert float(jnp.abs(ya - yb).max()) > 1e-3
    # Default (no dropout) stays key-free and deterministic.
    plain = block_cls.init(4, 4, 2, key=KEY)
    assert jnp.allclose(plain(x), plain(x))
