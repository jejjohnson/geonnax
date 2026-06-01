"""Shape, batching, and gradient tests for the geonnax U-Net surface."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.layers import (
    Attention,
    Block,
    ConvNeXtBlock,
    Downsample,
    GlobalResponseNorm,
    LinearAttention,
    ResnetBlock,
    SqueezeExcitation,
    StandardizedConv,
    Upsample,
)
from geonnax.unet import NestedResidualUNet, Stage, UNet, XUNet


KEY = jr.PRNGKey(0)


# ---- reusable layers --------------------------------------------------------


def test_standardized_conv_shapes_and_toggle():
    plain = StandardizedConv.init(2, 3, 4, 3, key=KEY, padding=1, standardize=False)
    ws = StandardizedConv.init(2, 3, 4, 3, key=KEY, padding=1, standardize=True)
    x = jnp.ones((3, 8, 8))
    assert plain(x).shape == (4, 8, 8)
    assert ws(x).shape == (4, 8, 8)
    # Standardization changes the effective weights, hence the output.
    assert not jnp.allclose(plain(x), ws(x))


def test_downsample_upsample_roundtrip_shapes():
    down = Downsample.init(4, 8, 2, key=KEY)
    up = Upsample.init(8, 4, 2, key=KEY)
    x = jnp.ones((4, 16, 16))
    d = down(x)
    assert d.shape == (8, 8, 8)
    u = up(d)
    assert u.shape == (4, 16, 16)


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_downsample_halves_each_dim(nd):
    down = Downsample.init(2, 2, nd, key=KEY)
    x = jnp.ones((2,) + (8,) * nd)
    assert down(x).shape == (2,) + (4,) * nd


def test_global_response_norm_identity_at_init():
    grn = GlobalResponseNorm.init(4, 2)
    x = jr.normal(KEY, (4, 6, 6))
    # gamma/beta init to zero -> GRN is the identity transform.
    assert jnp.allclose(grn(x), x)


def test_squeeze_excitation_gates_channels():
    se = SqueezeExcitation.init(8, 2, key=KEY)
    x = jr.normal(KEY, (8, 5, 5))
    out = se(x)
    assert out.shape == x.shape


def test_block_and_resnet_block():
    blk = Block.init(3, 6, 2, key=KEY)
    assert blk(jnp.ones((3, 8, 8))).shape == (6, 8, 8)
    rb = ResnetBlock.init(3, 6, 2, key=KEY)
    assert rb(jnp.ones((3, 8, 8))).shape == (6, 8, 8)


def test_convnext_block():
    cb = ConvNeXtBlock.init(4, 8, 2, key=KEY)
    assert cb(jnp.ones((4, 8, 8))).shape == (8, 8, 8)


@pytest.mark.parametrize("attn_cls", [Attention, LinearAttention])
def test_attention_preserves_shape(attn_cls):
    attn = attn_cls.init(16, 2, key=KEY, heads=2, dim_head=8)
    x = jr.normal(KEY, (16, 8, 8))
    assert attn(x).shape == x.shape


# ---- nested residual u-net --------------------------------------------------


def test_nested_residual_unet_preserves_shape():
    nru = NestedResidualUNet.init(8, 2, 2, key=KEY)
    x = jnp.ones((8, 16, 16))
    assert nru(x).shape == x.shape


def test_nested_residual_unet_rejects_zero_depth():
    with pytest.raises(ValueError, match="depth must be >= 1"):
        NestedResidualUNet.init(8, 0, 2, key=KEY)


# ---- full u-net -------------------------------------------------------------


def test_unet_default_preserves_spatial_shape():
    model = UNet.init(16, key=KEY, channels=3, dim_mults=(1, 2, 4))
    x = jnp.ones((3, 32, 32))
    assert model(x).shape == (3, 32, 32)


def test_unet_separate_out_channels():
    model = UNet.init(8, key=KEY, channels=2, out_channels=5, dim_mults=(1, 2))
    assert model(jnp.ones((2, 16, 16))).shape == (5, 16, 16)


def test_unet_vmaps_over_batch():
    model = UNet.init(8, key=KEY, channels=1, dim_mults=(1, 2))
    out = jax.vmap(model)(jnp.ones((4, 1, 16, 16)))
    assert out.shape == (4, 1, 16, 16)


@pytest.mark.parametrize("block_type", ["resnet", "convnext"])
def test_unet_block_types(block_type):
    model = UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2), block_type=block_type)
    assert model(jnp.ones((2, 16, 16))).shape == (2, 16, 16)


def test_unet_nested_stages():
    model = UNet.init(
        8, key=KEY, channels=2, dim_mults=(1, 2), nested_unet_depths=(2, 1)
    )
    assert model(jnp.ones((2, 16, 16))).shape == (2, 16, 16)


def test_unet_without_consolidation():
    model = UNet.init(
        8, key=KEY, channels=2, dim_mults=(1, 2), consolidate_upsample_fmaps=False
    )
    assert model(jnp.ones((2, 16, 16))).shape == (2, 16, 16)


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_unet_dimension_flexible(nd):
    model = UNet.init(8, key=KEY, channels=2, num_spatial_dims=nd, dim_mults=(1, 2))
    x = jnp.ones((2,) + (16,) * nd)
    assert model(x).shape == x.shape


def test_unet_rejects_empty_dim_mults():
    with pytest.raises(ValueError, match="dim_mults must be non-empty"):
        UNet.init(8, key=KEY, dim_mults=())


def test_unet_rejects_mismatched_nested_depths():
    with pytest.raises(ValueError, match="nested_unet_depths must match"):
        UNet.init(8, key=KEY, dim_mults=(1, 2), nested_unet_depths=(1, 1, 1))


def test_unet_is_jittable_and_differentiable():
    model = UNet.init(8, key=KEY, channels=2, dim_mults=(1, 2))
    x = jnp.ones((2, 16, 16))
    assert eqx.filter_jit(model)(x).shape == (2, 16, 16)

    def loss(m, x):
        return jnp.mean(m(x) ** 2)

    grads = eqx.filter_grad(loss)(model, x)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert leaves
    assert all(jnp.isfinite(g).all() for g in leaves)


def test_xunet_alias_and_public_exports():
    assert XUNet is UNet
    assert geonnax.UNet is UNet
    assert isinstance(
        Stage.init(
            4,
            4,
            2,
            0,
            "resnet",
            key=KEY,
            groups=8,
            weight_standardize=False,
            squeeze_excite=True,
        ),
        Stage,
    )
    assert hasattr(geonnax, "layers")
    assert hasattr(geonnax, "unet")
