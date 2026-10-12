"""Tests for `geonnax.cno`: the CNO blocks and the operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from geonnax.cno import CNO, CNOBlock, CNOResBlock, ConvolutionalNeuralOperator
from geonnax.layers import band_limited_resize


KEY = jr.PRNGKey(0)
ND = [1, 2, pytest.param(3, marks=pytest.mark.slow)]


def _grid(nd: int, n: int) -> tuple[int, ...]:
    return (n,) * nd


def _small(nd: int, **kw) -> CNO:
    kw = {"hidden_channels": 4, "depth": 2, "n_res_blocks": 1} | kw
    return CNO.init(2, 3, nd, key=KEY, **kw)


# ---- blocks -------------------------------------------------------------------


@pytest.mark.parametrize("nd", ND)
@pytest.mark.parametrize(
    ("resample", "factor"), [("same", 1.0), ("down", 0.5), ("up", 2.0)]
)
def test_cno_block_shapes(nd, resample, factor):
    blk = CNOBlock.init(3, 5, nd, key=KEY, resample=resample)
    out = blk(jnp.ones((3, *_grid(nd, 8))))
    assert out.shape == (5, *_grid(nd, int(8 * factor)))


def test_cno_block_same_is_conv_then_antialiased_activation():
    blk = CNOBlock.init(2, 4, 1, key=KEY)
    x = jr.normal(jr.PRNGKey(1), (2, 32))
    assert jnp.allclose(blk(x), blk.activation(blk.conv.conv(x)), atol=1e-6)


def test_cno_block_down_matches_direct_resampling():
    # D block = resize(sigma(resize(Kx, 2N)), N/2), the up -> sigma -> low-pass
    # of SM A.4 with the decimation folded into the final low-pass.
    blk = CNOBlock.init(2, 4, 1, key=KEY, resample="down")
    x = jr.normal(jr.PRNGKey(1), (2, 32))
    h = blk.conv.conv(x)
    ref = band_limited_resize(jax.nn.leaky_relu(band_limited_resize(h, (64,))), (16,))
    assert jnp.allclose(blk(x), ref, atol=1e-6)


@pytest.mark.parametrize("nd", ND)
def test_cno_res_block_shape_and_residual(nd):
    blk = CNOResBlock.init(4, nd, key=KEY)
    x = jr.normal(jr.PRNGKey(1), (4, *_grid(nd, 8)))
    assert blk(x).shape == x.shape
    zeroed = eqx.tree_at(
        lambda b: b.conv2.conv.weight, blk, jnp.zeros_like(blk.conv2.conv.weight)
    )
    zeroed = eqx.tree_at(
        lambda b: b.conv2.conv.bias, zeroed, jnp.zeros_like(blk.conv2.conv.bias)
    )
    assert jnp.allclose(zeroed(x), x)  # identity when K_2 = 0


def test_cno_block_down_rejects_odd_grid():
    blk = CNOBlock.init(1, 1, 1, key=KEY, resample="down")
    with pytest.raises(ValueError, match="even"):
        blk(jnp.ones((1, 9)))


# ---- the operator -------------------------------------------------------------


@pytest.mark.parametrize("nd", ND)
def test_cno_shapes(nd):
    op = _small(nd)
    assert op(jnp.ones((2, *_grid(nd, 16)))).shape == (3, *_grid(nd, 16))


@pytest.mark.parametrize("nd", ND)
def test_cno_resolution_resamples_to_computational_grid(nd):
    op = _small(nd, resolution=8)
    for n in (8, 16):
        assert op(jnp.ones((2, *_grid(nd, n)))).shape == (3, *_grid(nd, n))
    # Odd and non-divisible grids are fine: they are resized onto the grid.
    assert op(jnp.ones((2, *_grid(nd, 6)))).shape == (3, *_grid(nd, 6))


def test_cno_issue_example_and_alias():
    assert ConvolutionalNeuralOperator is CNO
    op = CNO.init(3, 1, 2, key=KEY, hidden_channels=8)  # depth 3
    assert op(jnp.ones((3, 32, 32))).shape == (1, 32, 32)
    assert op.resolution is None
    assert op.kernel_size == 3


def test_cno_widths_and_structure():
    op = CNO.init(1, 1, 1, key=KEY, hidden_channels=4, depth=3, n_res_blocks=2)
    assert len(op.down_blocks) == len(op.up_blocks) == len(op.inv_blocks) == 3
    assert [len(r) for r in op.res_blocks] == [2, 2, 2, 2]
    # Level l has 4 * 2**l channels; decoder inputs are the bottleneck (32)
    # or the [U, skip] concatenation (2 * 4 * 2**(l + 1)).
    assert [b.conv.conv.out_channels for b in op.down_blocks] == [8, 16, 32]
    assert [b.conv.conv.in_channels for b in op.inv_blocks] == [16, 32, 32]
    assert [b.conv.conv.out_channels for b in op.up_blocks] == [4, 8, 16]
    assert op.proj_block.conv.conv.in_channels == 8


def test_cno_zero_res_blocks():
    op = _small(1, n_res_blocks=0)
    assert op.res_blocks == [[], [], []]
    assert op(jnp.ones((2, 16))).shape == (3, 16)


def test_cno_gelu_and_up_configurable():
    op = _small(1, activation=jax.nn.gelu, activation_up=3)
    assert op.lift.activation.activation is jax.nn.gelu
    assert op.lift.activation.up == 3
    assert jnp.all(jnp.isfinite(op(jnp.ones((2, 16)))))


def test_cno_default_activation_is_leaky_relu_up2():
    op = _small(1)
    assert op.lift.activation.activation is jax.nn.leaky_relu
    assert op.lift.activation.up == 2


def test_cno_weight_standardize():
    op = _small(2, weight_standardize=True)
    assert op.lift.conv.standardize
    assert op(jr.normal(KEY, (2, 16, 16))).shape == (3, 16, 16)


# ---- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_spatial_dims": 4}, "num_spatial_dims"),
        ({"num_spatial_dims": 0}, "num_spatial_dims"),
        ({"hidden_channels": 0}, "hidden_channels"),
        ({"in_channels": 0}, "in_channels"),
        ({"depth": 0}, "depth"),
        ({"n_res_blocks": -1}, "n_res_blocks"),
        ({"kernel_size": 2}, "kernel_size"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"activation_up": 0}, "activation_up"),
        ({"resolution": (16,)}, "resolution must have 2"),
        ({"resolution": 10}, "divisible"),
        ({"resolution": (16, 2)}, "divisible"),
    ],
)
def test_cno_init_validation(kwargs, match):
    args = {"in_channels": 1, "out_channels": 1, "num_spatial_dims": 2} | kwargs
    with pytest.raises(ValueError, match=match):
        CNO.init(
            args.pop("in_channels"),
            args.pop("out_channels"),
            args.pop("num_spatial_dims"),
            key=KEY,
            depth=args.pop("depth", 2),
            **args,
        )


def test_cno_block_validation():
    with pytest.raises(ValueError, match="resample"):
        CNOBlock.init(1, 1, 1, key=KEY, resample="sideways")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="out_channels"):
        CNOBlock.init(1, 0, 1, key=KEY)
    with pytest.raises(ValueError, match="channels"):
        CNOResBlock.init(0, 1, key=KEY)
    with pytest.raises(ValueError, match="kernel_size"):
        CNOResBlock.init(2, 1, key=KEY, kernel_size=4)


def test_cno_call_validation():
    op = _small(2)
    with pytest.raises(ValueError, match="spatial axes"):
        op(jnp.ones((2, 16)))  # missing a spatial axis
    with pytest.raises(ValueError, match="divisible"):
        op(jnp.ones((2, 16, 18)))  # 18 not divisible by 2**2 without resolution


# ---- transforms ---------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("nd", ND)
def test_cno_jit_vmap_grad(nd):
    op = _small(nd, resolution=8)
    xb = jr.normal(KEY, (3, 2, *_grid(nd, 16)))
    out = eqx.filter_jit(jax.vmap(op))(xb)
    assert out.shape == (3, 3, *_grid(nd, 16))
    assert jnp.allclose(out[1], op(xb[1]), atol=1e-5)

    def loss(model):
        return jnp.mean(jax.vmap(model)(xb) ** 2)

    grads = eqx.filter_grad(loss)(op)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(float(jnp.abs(g).max()) > 0 for g in leaves)
