"""Residual and ConvNeXt convolutional blocks with channel re-calibration."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers._conv import StandardizedConv
from geonnax.layers._norm import GlobalResponseNorm
from geonnax.layers._utils import act, group_count


class SqueezeExcitation(eqx.Module):
    """Squeeze-and-Excitation channel gating (Hu et al., 2018).

    Globally average-pools each channel, passes the descriptor through a
    bottleneck MLP, and multiplies the channels by the resulting sigmoid gate.

    Attributes:
        fc1: Squeeze projection ``C -> C // reduction``.
        fc2: Excitation projection ``C // reduction -> C``.
        num_spatial_dims: Number of trailing spatial axes.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import SqueezeExcitation
        >>> se = SqueezeExcitation.init(8, 2, key=jr.PRNGKey(0))
        >>> se(jnp.ones((8, 5, 5))).shape   # re-weights channels, shape kept
        (8, 5, 5)
    """

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    num_spatial_dims: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        reduction: int = 4,
    ) -> SqueezeExcitation:
        """Construct an SE block with a ``channels // reduction`` bottleneck."""
        hidden = max(1, channels // reduction)
        k1, k2 = jax.random.split(key)
        fc1 = eqx.nn.Linear(channels, hidden, key=k1)
        fc2 = eqx.nn.Linear(hidden, channels, key=k2)
        return cls(fc1=fc1, fc2=fc2, num_spatial_dims=num_spatial_dims)

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        axes = tuple(range(1, self.num_spatial_dims + 1))
        s = jnp.mean(x, axis=axes)
        s = act(self.fc1(s))
        s = jax.nn.sigmoid(self.fc2(s))
        return x * s.reshape(s.shape + (1,) * self.num_spatial_dims)


class Block(eqx.Module):
    """Conv → GroupNorm → SiLU, the residual-block primitive.

    Attributes:
        conv: Same-padding convolution (default ``kernel_size=3``).
        norm: GroupNorm over the output channels.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import Block
        >>> blk = Block.init(3, 6, 2, key=jr.PRNGKey(0))   # 3 -> 6 channels
        >>> blk(jnp.ones((3, 8, 8))).shape
        (6, 8, 8)
    """

    conv: StandardizedConv
    norm: eqx.nn.GroupNorm

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        kernel_size: int = 3,
        groups: int = 8,
        weight_standardize: bool = False,
    ) -> Block:
        """Construct a conv/norm/act block."""
        conv = StandardizedConv.init(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            key=key,
            padding=kernel_size // 2,
            standardize=weight_standardize,
        )
        norm = eqx.nn.GroupNorm(
            groups=group_count(out_channels, groups), channels=out_channels
        )
        return cls(conv=conv, norm=norm)

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        return act(self.norm(self.conv(x)))


class ResnetBlock(eqx.Module):
    r"""Two `Block`\ s with squeeze-excitation and a residual skip.

    Attributes:
        block1: First conv/norm/act block (``in -> out``).
        block2: Second conv/norm/act block (``out -> out``).
        se: Optional squeeze-excitation gate on the output.
        res_conv: ``1x1`` projection of the residual when channels change,
            otherwise ``None`` (identity).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import ResnetBlock
        >>> blk = ResnetBlock.init(3, 6, 2, key=jr.PRNGKey(0))
        >>> blk(jnp.ones((3, 8, 8))).shape
        (6, 8, 8)
    """

    block1: Block
    block2: Block
    se: SqueezeExcitation | None
    res_conv: StandardizedConv | None

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        kernel_size: int = 3,
        groups: int = 8,
        weight_standardize: bool = False,
        squeeze_excite: bool = True,
    ) -> ResnetBlock:
        """Construct a residual block."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        block1 = Block.init(
            in_channels,
            out_channels,
            num_spatial_dims,
            key=k1,
            kernel_size=kernel_size,
            groups=groups,
            weight_standardize=weight_standardize,
        )
        block2 = Block.init(
            out_channels,
            out_channels,
            num_spatial_dims,
            key=k2,
            kernel_size=kernel_size,
            groups=groups,
            weight_standardize=weight_standardize,
        )
        se = (
            SqueezeExcitation.init(out_channels, num_spatial_dims, key=k3)
            if squeeze_excite
            else None
        )
        res_conv = (
            StandardizedConv.init(
                num_spatial_dims,
                in_channels,
                out_channels,
                1,
                key=k4,
                standardize=weight_standardize,
            )
            if in_channels != out_channels
            else None
        )
        return cls(block1=block1, block2=block2, se=se, res_conv=res_conv)

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        h = self.block2(self.block1(x))
        if self.se is not None:
            h = self.se(h)
        res = x if self.res_conv is None else self.res_conv(x)
        return h + res


class ConvNeXtBlock(eqx.Module):
    """ConvNeXt-V2 block: depthwise conv → norm → MLP with GRN (Woo et al., 2023).

    Attributes:
        ds_conv: Depthwise (grouped) ``kernel_size=7`` convolution.
        norm: GroupNorm over the input channels.
        pw1: Pointwise expansion ``in -> in * mult``.
        grn: Global Response Normalization on the expanded features.
        pw2: Pointwise projection ``in * mult -> out``.
        res_conv: ``1x1`` residual projection when channels change.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import ConvNeXtBlock
        >>> blk = ConvNeXtBlock.init(4, 8, 2, key=jr.PRNGKey(0))
        >>> blk(jnp.ones((4, 8, 8))).shape
        (8, 8, 8)
    """

    ds_conv: StandardizedConv
    norm: eqx.nn.GroupNorm
    pw1: StandardizedConv
    grn: GlobalResponseNorm
    pw2: StandardizedConv
    res_conv: StandardizedConv | None

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        mult: int = 2,
        groups: int = 8,
        weight_standardize: bool = False,
    ) -> ConvNeXtBlock:
        """Construct a ConvNeXt-V2 block."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        hidden = in_channels * mult
        ds_conv = StandardizedConv.init(
            num_spatial_dims,
            in_channels,
            in_channels,
            7,
            key=k1,
            padding=3,
            groups=in_channels,
            standardize=weight_standardize,
        )
        norm = eqx.nn.GroupNorm(
            groups=group_count(in_channels, groups), channels=in_channels
        )
        pw1 = StandardizedConv.init(
            num_spatial_dims,
            in_channels,
            hidden,
            1,
            key=k2,
            standardize=weight_standardize,
        )
        grn = GlobalResponseNorm.init(hidden, num_spatial_dims)
        pw2 = StandardizedConv.init(
            num_spatial_dims,
            hidden,
            out_channels,
            1,
            key=k3,
            standardize=weight_standardize,
        )
        res_conv = (
            StandardizedConv.init(
                num_spatial_dims,
                in_channels,
                out_channels,
                1,
                key=k4,
                standardize=weight_standardize,
            )
            if in_channels != out_channels
            else None
        )
        return cls(
            ds_conv=ds_conv, norm=norm, pw1=pw1, grn=grn, pw2=pw2, res_conv=res_conv
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        h = self.norm(self.ds_conv(x))
        h = self.pw1(h)
        h = jax.nn.gelu(h)
        h = self.grn(h)
        h = self.pw2(h)
        res = x if self.res_conv is None else self.res_conv(x)
        return h + res


__all__ = ["Block", "ConvNeXtBlock", "ResnetBlock", "SqueezeExcitation"]
