"""Modern, dimension-flexible U-Net cores (pure ``equinox.Module``).

A JAX/Equinox port of the design popularised by `lucidrains/x-unet
<https://github.com/lucidrains/x-unet>`_, assembling the reusable primitives in
:mod:`geonnax.layers` into a convolutional encoder–decoder that synthesises
several lines of research:

- **U-Net** (Ronneberger et al., 2015) — the encoder/decoder skeleton with
  skip connections.
- **U²-Net** (Qin et al., 2020) — optional *nested* residual U-Nets used as a
  per-stage block (:class:`NestedResidualUNet`).
- **Weight Standardization** (Qiao et al., 2019) + **GroupNorm** (Wu & He,
  2018) — micro-batch-friendly normalisation.
- **ConvNeXt-V2** (Woo et al., 2023) — depthwise blocks with Global Response
  Normalisation.
- **Squeeze-and-Excitation** (Hu et al., 2018) — channel re-calibration.
- **Cosine / QK-normalised attention** (Henry et al., 2020) — stable
  self-attention at the bottleneck.

Like the underlying layers, :class:`UNet` operates on a *single* example of
shape ``(channels, *spatial)`` with ``num_spatial_dims`` spatial axes
(``2`` ⇒ ``(C, H, W)`` maps by default, the natural layout for gridded
geoscience data; ``1`` gives time series / profiles, ``3`` gives volumes).
``jax.vmap`` over a batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers import (
    Attention,
    ConvNeXtBlock,
    Downsample,
    ResnetBlock,
    StandardizedConv,
    Upsample,
)


BlockType = Literal["resnet", "convnext"]


class NestedResidualUNet(eqx.Module):
    """A small residual U-Net used as one stage block (Qin et al., 2020, "U²-Net").

    Keeps the channel width fixed at ``dim`` while recursively down- and
    up-sampling the spatial grid ``depth`` times, with additive, ``1/√2``-scaled
    skip connections and a final residual to the input. Each spatial axis must
    be divisible by ``2**depth``.

    Attributes:
        down_blocks: Per-level residual blocks on the encoder path.
        downsamples: Per-level factor-2 downsamplers.
        mid_block: Bottleneck residual block.
        upsamples: Per-level factor-2 upsamplers.
        up_blocks: Per-level residual blocks on the decoder path.
        skip_scale: Multiplier applied after adding each skip (``1/√2``).
    """

    down_blocks: list[ResnetBlock]
    downsamples: list[Downsample]
    mid_block: ResnetBlock
    upsamples: list[Upsample]
    up_blocks: list[ResnetBlock]
    skip_scale: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dim: int,
        depth: int,
        num_spatial_dims: int,
        *,
        key: Array,
        groups: int = 8,
        weight_standardize: bool = False,
    ) -> NestedResidualUNet:
        """Construct a depth-``depth`` nested residual U-Net at width ``dim``."""
        if depth < 1:
            raise ValueError(f"NestedResidualUNet depth must be >= 1, got {depth}")
        keys = iter(jax.random.split(key, depth * 4 + 1))
        down_blocks, downsamples, upsamples, up_blocks = [], [], [], []
        for _ in range(depth):
            down_blocks.append(
                ResnetBlock.init(
                    dim,
                    dim,
                    num_spatial_dims,
                    key=next(keys),
                    groups=groups,
                    weight_standardize=weight_standardize,
                )
            )
            downsamples.append(
                Downsample.init(
                    dim,
                    dim,
                    num_spatial_dims,
                    key=next(keys),
                    weight_standardize=weight_standardize,
                )
            )
            upsamples.append(
                Upsample.init(
                    dim,
                    dim,
                    num_spatial_dims,
                    key=next(keys),
                    weight_standardize=weight_standardize,
                )
            )
            up_blocks.append(
                ResnetBlock.init(
                    dim,
                    dim,
                    num_spatial_dims,
                    key=next(keys),
                    groups=groups,
                    weight_standardize=weight_standardize,
                )
            )
        mid_block = ResnetBlock.init(
            dim,
            dim,
            num_spatial_dims,
            key=next(keys),
            groups=groups,
            weight_standardize=weight_standardize,
        )
        return cls(
            down_blocks=down_blocks,
            downsamples=downsamples,
            mid_block=mid_block,
            upsamples=upsamples,
            up_blocks=up_blocks,
            skip_scale=2**-0.5,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        residual = x
        skips: list[Array] = []
        for block, down in zip(self.down_blocks, self.downsamples, strict=True):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.mid_block(x)
        for up, block in zip(
            reversed(self.upsamples), reversed(self.up_blocks), strict=True
        ):
            x = up(x)
            x = (x + skips.pop()) * self.skip_scale
            x = block(x)
        return x + residual


class Stage(eqx.Module):
    """One encoder/decoder stage: a leaf block, then an optional nested U-Net.

    The leaf block adapts the channel count; the optional channel-preserving
    :class:`NestedResidualUNet` then refines features at the stage resolution.

    Attributes:
        leaf: Channel-adapting block (:class:`ResnetBlock` or
            :class:`ConvNeXtBlock`).
        nested: Optional nested residual U-Net at the leaf's output width.
    """

    leaf: ResnetBlock | ConvNeXtBlock
    nested: NestedResidualUNet | None

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        nested_depth: int,
        block_type: BlockType,
        *,
        key: Array,
        groups: int,
        weight_standardize: bool,
        squeeze_excite: bool,
    ) -> Stage:
        """Construct a stage block, optionally wrapping a nested U-Net."""
        k_leaf, k_nested = jax.random.split(key)
        leaf: ResnetBlock | ConvNeXtBlock
        if block_type == "convnext":
            leaf = ConvNeXtBlock.init(
                in_channels,
                out_channels,
                num_spatial_dims,
                key=k_leaf,
                groups=groups,
                weight_standardize=weight_standardize,
            )
        elif block_type == "resnet":
            leaf = ResnetBlock.init(
                in_channels,
                out_channels,
                num_spatial_dims,
                key=k_leaf,
                groups=groups,
                weight_standardize=weight_standardize,
                squeeze_excite=squeeze_excite,
            )
        else:
            raise ValueError(
                f"block_type must be 'resnet' or 'convnext', got {block_type!r}"
            )
        nested = (
            NestedResidualUNet.init(
                out_channels,
                nested_depth,
                num_spatial_dims,
                key=k_nested,
                groups=groups,
                weight_standardize=weight_standardize,
            )
            if nested_depth >= 1
            else None
        )
        return cls(leaf=leaf, nested=nested)

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        x = self.leaf(x)
        if self.nested is not None:
            x = self.nested(x)
        return x


class UNet(eqx.Module):
    """Dimension-flexible U-Net with modern blocks, attention, and nested stages.

    Operates on a single example of shape ``(channels, *spatial)`` with
    ``num_spatial_dims`` spatial axes (``2`` ⇒ ``(C, H, W)`` maps by default);
    ``jax.vmap`` over a batch. Each spatial extent must be divisible by
    ``2**len(dim_mults)`` (and, where nested stages are used, by the extra
    ``2**nested_depth`` at that resolution).

    The encoder applies a :class:`Stage` then a factor-2 downsample at each
    level; the bottleneck is ``ResnetBlock → Attention → ResnetBlock``; the
    decoder upsamples, fuses the ``1/√2``-scaled skip, and applies a stage. With
    ``consolidate_upsample_fmaps`` the decoder outputs at every resolution are
    resized to full resolution and concatenated before the output head (x-unet
    style), giving the head multi-scale context.

    Attributes:
        init_conv: Stem convolution ``channels -> dim``.
        down_blocks: Encoder stages.
        downsamples: Encoder downsamplers.
        mid_block1, mid_attn, mid_block2: Bottleneck.
        upsamples: Decoder upsamplers.
        up_blocks: Decoder stages.
        consolidate_conv: Optional ``1x1`` fuse of multi-scale decoder maps.
        final_block: Output residual block.
        final_conv: Output head ``dim -> out_channels``.
        num_spatial_dims, channels, out_channels: Static config.
        skip_scale: ``1/√2`` skip multiplier.
    """

    init_conv: StandardizedConv
    down_blocks: list[Stage]
    downsamples: list[Downsample]
    mid_block1: ResnetBlock
    mid_attn: Attention
    mid_block2: ResnetBlock
    upsamples: list[Upsample]
    up_blocks: list[Stage]
    consolidate_conv: StandardizedConv | None
    final_block: ResnetBlock
    final_conv: StandardizedConv
    num_spatial_dims: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    skip_scale: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dim: int,
        *,
        key: Array,
        channels: int = 1,
        out_channels: int | None = None,
        num_spatial_dims: int = 2,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        nested_unet_depths: Sequence[int] | int = 0,
        block_type: BlockType = "resnet",
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        groups: int = 8,
        weight_standardize: bool = True,
        squeeze_excite: bool = True,
        consolidate_upsample_fmaps: bool = True,
    ) -> UNet:
        """Construct a U-Net.

        Args:
            dim: Base channel width; stage widths are ``dim * m`` for ``m`` in
                ``dim_mults``.
            key: PRNG key.
            channels: Number of input channels.
            out_channels: Number of output channels (defaults to ``channels``).
            num_spatial_dims: Spatial rank (1, 2, or 3).
            dim_mults: Per-stage channel multipliers; its length sets the
                number of resolution levels.
            nested_unet_depths: Per-stage :class:`NestedResidualUNet` depth
                (``0`` disables nesting). An int broadcasts to all stages.
            block_type: Leaf block, ``"resnet"`` or ``"convnext"``.
            attn_heads: Bottleneck attention heads.
            attn_dim_head: Bottleneck attention channels per head.
            groups: GroupNorm groups (clamped per layer via gcd).
            weight_standardize: Standardize conv weights at call time.
            squeeze_excite: Add SE gating to residual blocks.
            consolidate_upsample_fmaps: Fuse multi-scale decoder maps before
                the head.

        Raises:
            ValueError: If ``dim_mults`` is empty or ``nested_unet_depths``
                length mismatches ``dim_mults``.
        """
        if len(dim_mults) == 0:
            raise ValueError("dim_mults must be non-empty.")
        out_channels = channels if out_channels is None else out_channels
        n_stages = len(dim_mults)
        if isinstance(nested_unet_depths, int):
            depths = (nested_unet_depths,) * n_stages
        else:
            depths = tuple(nested_unet_depths)
            if len(depths) != n_stages:
                raise ValueError(
                    "nested_unet_depths must match len(dim_mults); "
                    f"got {len(depths)} and {n_stages}."
                )

        dims = [dim, *(dim * m for m in dim_mults)]
        in_out = list(pairwise(dims))

        # Keys consumed: init_conv (1) + down stages (2 each) + mid (3) +
        # up stages (2 each) + consolidate (1) + final_block (1) + final_conv (1).
        keys = iter(jax.random.split(key, 4 * n_stages + 7))

        init_conv = StandardizedConv.init(
            num_spatial_dims,
            channels,
            dim,
            7,
            key=next(keys),
            padding=3,
            standardize=weight_standardize,
        )

        down_blocks: list[Stage] = []
        downsamples: list[Downsample] = []
        for (d_in, d_out), depth in zip(in_out, depths, strict=True):
            down_blocks.append(
                Stage.init(
                    d_in,
                    d_in,
                    num_spatial_dims,
                    depth,
                    block_type,
                    key=next(keys),
                    groups=groups,
                    weight_standardize=weight_standardize,
                    squeeze_excite=squeeze_excite,
                )
            )
            downsamples.append(
                Downsample.init(
                    d_in,
                    d_out,
                    num_spatial_dims,
                    key=next(keys),
                    weight_standardize=weight_standardize,
                )
            )

        mid_dim = dims[-1]
        mid_block1 = ResnetBlock.init(
            mid_dim,
            mid_dim,
            num_spatial_dims,
            key=next(keys),
            groups=groups,
            weight_standardize=weight_standardize,
            squeeze_excite=squeeze_excite,
        )
        mid_attn = Attention.init(
            mid_dim,
            num_spatial_dims,
            key=next(keys),
            heads=attn_heads,
            dim_head=attn_dim_head,
            groups=groups,
        )
        mid_block2 = ResnetBlock.init(
            mid_dim,
            mid_dim,
            num_spatial_dims,
            key=next(keys),
            groups=groups,
            weight_standardize=weight_standardize,
            squeeze_excite=squeeze_excite,
        )

        upsamples: list[Upsample] = []
        up_blocks: list[Stage] = []
        up_out_channels: list[int] = []
        for (d_in, d_out), depth in zip(
            reversed(in_out), reversed(depths), strict=True
        ):
            upsamples.append(
                Upsample.init(
                    d_out,
                    d_in,
                    num_spatial_dims,
                    key=next(keys),
                    weight_standardize=weight_standardize,
                )
            )
            up_blocks.append(
                Stage.init(
                    d_in * 2,
                    d_in,
                    num_spatial_dims,
                    depth,
                    block_type,
                    key=next(keys),
                    groups=groups,
                    weight_standardize=weight_standardize,
                    squeeze_excite=squeeze_excite,
                )
            )
            up_out_channels.append(d_in)

        if consolidate_upsample_fmaps:
            consolidate_conv = StandardizedConv.init(
                num_spatial_dims,
                sum(up_out_channels),
                dim,
                1,
                key=next(keys),
                standardize=weight_standardize,
            )
        else:
            consolidate_conv = None
            next(keys)  # keep the key stream aligned

        final_block = ResnetBlock.init(
            dim,
            dim,
            num_spatial_dims,
            key=next(keys),
            groups=groups,
            weight_standardize=weight_standardize,
            squeeze_excite=squeeze_excite,
        )
        final_conv = StandardizedConv.init(
            num_spatial_dims,
            dim,
            out_channels,
            1,
            key=next(keys),
            standardize=weight_standardize,
        )

        return cls(
            init_conv=init_conv,
            down_blocks=down_blocks,
            downsamples=downsamples,
            mid_block1=mid_block1,
            mid_attn=mid_attn,
            mid_block2=mid_block2,
            upsamples=upsamples,
            up_blocks=up_blocks,
            consolidate_conv=consolidate_conv,
            final_block=final_block,
            final_conv=final_conv,
            num_spatial_dims=num_spatial_dims,
            channels=channels,
            out_channels=out_channels,
            skip_scale=2**-0.5,
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        x = self.init_conv(x)

        skips: list[Array] = []
        for block, down in zip(self.down_blocks, self.downsamples, strict=True):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x)

        up_fmaps: list[Array] = []
        for up, block in zip(self.upsamples, self.up_blocks, strict=True):
            x = up(x)
            skip = skips.pop() * self.skip_scale
            x = jnp.concatenate([x, skip], axis=0)
            x = block(x)
            up_fmaps.append(x)

        if self.consolidate_conv is not None:
            full = up_fmaps[-1].shape[1:]
            resized = [
                jax.image.resize(f, (f.shape[0], *full), method="nearest")
                for f in up_fmaps
            ]
            x = self.consolidate_conv(jnp.concatenate(resized, axis=0))

        x = self.final_block(x)
        return self.final_conv(x)


XUNet = UNet
"""Alias for :class:`UNet`, echoing the ``x-unet`` design this port follows."""


__all__ = [
    "BlockType",
    "NestedResidualUNet",
    "Stage",
    "UNet",
    "XUNet",
]
