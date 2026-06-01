"""Convolution primitives: weight-standardized conv and resampling.

These layers operate on a single example with a leading channel axis and
``num_spatial_dims`` trailing spatial axes — shape ``(C, *spatial)`` — and are
meant to be ``jax.vmap``-ed over a batch.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers._utils import shuffle_kwargs, shuffle_patterns


class StandardizedConv(eqx.Module):
    """Convolution with optional on-the-fly weight standardization.

    Wraps `equinox.nn.Conv`. When ``standardize`` is set, each output
    filter's weights are mean/variance-normalised at call time (Qiao et al.,
    2019, "Weight Standardization"), which — paired with GroupNorm — stabilises
    training at the small effective batch sizes common in scientific
    workloads.

    Attributes:
        conv: The underlying Equinox convolution.
        standardize: Whether to standardize weights before convolving.
        eps: Numerical floor added to the per-filter variance.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import StandardizedConv
        >>> conv = StandardizedConv.init(  # 2D, 3->8 channels, 3x3, same pad
        ...     2, 3, 8, 3, key=jr.PRNGKey(0), padding=1, standardize=True)
        >>> conv(jnp.ones((3, 16, 16))).shape
        (8, 16, 16)
    """

    conv: eqx.nn.Conv
    standardize: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        key: Array,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        use_bias: bool = True,
        standardize: bool = False,
        eps: float = 1e-5,
    ) -> StandardizedConv:
        """Construct a (optionally weight-standardized) convolution."""
        conv = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            use_bias=use_bias,
            key=key,
        )
        return cls(conv=conv, standardize=standardize, eps=eps)

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *out_spatial"]:
        if not self.standardize:
            return self.conv(x)
        w = self.conv.weight
        axes = tuple(range(1, w.ndim))
        mean = jnp.mean(w, axis=axes, keepdims=True)
        var = jnp.var(w, axis=axes, keepdims=True)
        w = (w - mean) * jax.lax.rsqrt(var + self.eps)
        conv = eqx.tree_at(lambda c: c.weight, self.conv, w)
        return conv(x)


class Downsample(eqx.Module):
    """Memory-efficient factor-2 downsample: space-to-depth, then ``1x1`` conv.

    Folding the spatial patches into channels (rather than using a strided
    convolution) keeps all information available to the channel mix and avoids
    aliasing. Each spatial extent must be even.

    Attributes:
        conv: ``1x1`` conv mapping ``in * 2**d -> out`` channels.
        num_spatial_dims: Number of trailing spatial axes.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import Downsample
        >>> down = Downsample.init(4, 8, 2, key=jr.PRNGKey(0))
        >>> down(jnp.ones((4, 16, 16))).shape   # halves spatial, sets channels
        (8, 8, 8)
    """

    conv: StandardizedConv
    num_spatial_dims: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        weight_standardize: bool = False,
    ) -> Downsample:
        """Construct a downsample block."""
        folded = in_channels * (2**num_spatial_dims)
        conv = StandardizedConv.init(
            num_spatial_dims,
            folded,
            out_channels,
            1,
            key=key,
            standardize=weight_standardize,
        )
        return cls(conv=conv, num_spatial_dims=num_spatial_dims)

    def __call__(self, x: Float[Array, "C_in *spatial"]) -> Float[Array, "C_out *half"]:
        down, _ = shuffle_patterns(self.num_spatial_dims)
        # einx's keyword-only `backend` precedes **parameters, so unpacking a
        # dict of axis sizes trips ty's argument-type check; the keys are axes.
        sizes = shuffle_kwargs(self.num_spatial_dims)
        x = einx.id(down, x, **sizes)  # ty: ignore[invalid-argument-type]
        return self.conv(x)


class Upsample(eqx.Module):
    """Factor-2 upsample via pixel-shuffle: ``1x1`` conv, then depth-to-space.

    The sub-pixel convolution (Shi et al., 2016) avoids the checkerboard
    artefacts of transposed convolutions.

    Attributes:
        conv: ``1x1`` conv mapping ``in -> out * 2**d`` channels.
        num_spatial_dims: Number of trailing spatial axes.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import Upsample
        >>> up = Upsample.init(8, 4, 2, key=jr.PRNGKey(0))
        >>> up(jnp.ones((8, 8, 8))).shape   # doubles spatial, sets channels
        (4, 16, 16)
    """

    conv: StandardizedConv
    num_spatial_dims: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        weight_standardize: bool = False,
    ) -> Upsample:
        """Construct an upsample block."""
        expanded = out_channels * (2**num_spatial_dims)
        conv = StandardizedConv.init(
            num_spatial_dims,
            in_channels,
            expanded,
            1,
            key=key,
            standardize=weight_standardize,
        )
        return cls(conv=conv, num_spatial_dims=num_spatial_dims)

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *double"]:
        _, up = shuffle_patterns(self.num_spatial_dims)
        x = self.conv(x)
        sizes = shuffle_kwargs(self.num_spatial_dims)
        return einx.id(up, x, **sizes)  # ty: ignore[invalid-argument-type]


__all__ = ["Downsample", "StandardizedConv", "Upsample"]
