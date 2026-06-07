"""Multi-Scale Wavelet Transformer (Wang et al., 2026) — a wavelet-attention operator.

A U-shaped neural operator that performs self-attention in the wavelet domain
(`geonnax.layers.WaveletAttention`) at each scale, with wavelet-based
down/up-sampling between scales so that high-frequency detail bands survive
resolution changes instead of being pooled away. A patch tokenizer lifts the
physical grid into token space before the U-net runs and projects back after.

Unlike the FNO/WNO/SFNO operators, MSWT is *attention-based*: reach for it when
you want global (or windowed) mixing that is explicitly aware of multi-scale
frequency content — e.g. chaotic or long-rollout dynamics where small scales
matter. Operates on a single example ``(channels, *spatial)``; ``jax.vmap`` over
a batch. Each spatial extent must be divisible by ``patch_size * 2**(depth + 1)``.

References:
    Wang et al. (2026), "Multi-Scale Wavelet Transformers for Operator Learning
    of Dynamical Systems", arXiv:2602.01486. The wavelet-attention block follows
    the Wave-ViT / torch_wavelets lineage; windowed attention follows Swin
    (Liu et al., 2021).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers import WaveletAttention
from geonnax.layers._conv import StandardizedConv
from geonnax.layers._utils import group_count
from geonnax.layers._wavelet import Wavelet, _flatten, _unflatten, dwt, idwt


def _level1_layout(num_spatial_dims: int) -> list[list[tuple[str, ...]]]:
    """Sub-band key layout of a single-level DWT, matching `_flatten`'s ordering.

    `_flatten` lists the approximation band first, then the detail bands in
    ``sorted`` key order; this reproduces that detail-key ordering so a stacked
    tensor can be split back into coefficients without a reference transform.
    """
    all_a = ("a",) * num_spatial_dims
    detail_keys = sorted(
        k for k in product("ad", repeat=num_spatial_dims) if k != all_a
    )
    return [detail_keys]


class WaveletDownsample(eqx.Module):
    """Halve resolution by a single-level DWT, carrying all sub-bands forward.

    Unlike strided pooling, the detail bands are not discarded: the ``2**d``
    sub-bands are stacked and a ``1x1`` conv mixes them back to ``dim`` channels
    at half resolution.
    """

    conv: StandardizedConv
    wavelet: Wavelet = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)

    @classmethod
    def init(
        cls, dim: int, num_spatial_dims: int, *, key: Array, wavelet: Wavelet = "haar"
    ) -> WaveletDownsample:
        """Construct a wavelet downsampler at fixed width ``dim``."""
        factor = 2**num_spatial_dims
        conv = StandardizedConv.init(num_spatial_dims, factor * dim, dim, 1, key=key)
        return cls(conv=conv, wavelet=wavelet, num_spatial_dims=num_spatial_dims)

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *half"]:
        axes = tuple(range(1, self.num_spatial_dims + 1))
        bands, _ = _flatten(dwt(x, self.wavelet, 1, axes))
        return self.conv(jnp.concatenate(bands, axis=0))


class WaveletUpsample(eqx.Module):
    """Double resolution with a single-level inverse DWT (mirror of the downsampler)."""

    conv: StandardizedConv
    wavelet: Wavelet = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)

    @classmethod
    def init(
        cls, dim: int, num_spatial_dims: int, *, key: Array, wavelet: Wavelet = "haar"
    ) -> WaveletUpsample:
        """Construct a wavelet upsampler at fixed width ``dim``."""
        factor = 2**num_spatial_dims
        conv = StandardizedConv.init(num_spatial_dims, dim, factor * dim, 1, key=key)
        return cls(conv=conv, wavelet=wavelet, num_spatial_dims=num_spatial_dims)

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *double"]:
        axes = tuple(range(1, self.num_spatial_dims + 1))
        factor = 2**self.num_spatial_dims
        bands = list(jnp.split(self.conv(x), factor, axis=0))
        coeffs = _unflatten(bands, _level1_layout(self.num_spatial_dims))
        return idwt(coeffs, self.wavelet, axes)


class MSWTBlock(eqx.Module):
    """Pre-norm wavelet-attention block: an attention residual, then an FFN residual.

    A transformer block whose token mixer is `WaveletAttention` (global or
    windowed) and whose channel mixer is a pointwise feed-forward network, each
    with its own GroupNorm pre-norm and residual.
    """

    norm1: eqx.nn.GroupNorm
    attn: WaveletAttention
    norm2: eqx.nn.GroupNorm
    ff1: StandardizedConv
    ff2: StandardizedConv
    activation: Callable[[Array], Array] = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dim: int,
        num_spatial_dims: int,
        *,
        key: Array,
        wavelet: Wavelet = "haar",
        heads: int = 4,
        dim_head: int = 32,
        groups: int = 8,
        window: int | None = None,
        ff_mult: int = 2,
        activation: Callable[[Array], Array] = jax.nn.gelu,
    ) -> MSWTBlock:
        """Construct a wavelet-attention transformer block at width ``dim``."""
        k_attn, k_ff1, k_ff2 = jax.random.split(key, 3)
        gc = group_count(dim, groups)
        attn = WaveletAttention.init(
            dim,
            num_spatial_dims,
            key=k_attn,
            wavelet=wavelet,
            heads=heads,
            dim_head=dim_head,
            groups=groups,
            window=window,
        )
        hidden = dim * ff_mult
        return cls(
            norm1=eqx.nn.GroupNorm(groups=gc, channels=dim),
            attn=attn,
            norm2=eqx.nn.GroupNorm(groups=gc, channels=dim),
            ff1=StandardizedConv.init(num_spatial_dims, dim, hidden, 1, key=k_ff1),
            ff2=StandardizedConv.init(num_spatial_dims, hidden, dim, 1, key=k_ff2),
            activation=activation,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        x = x + self.attn(self.norm1(x))
        return x + self.ff2(self.activation(self.ff1(self.norm2(x))))


class MSWT(eqx.Module):
    """Multi-Scale Wavelet Transformer operator (Wang et al., 2026).

    Patch tokenizer → U-shaped stack of `MSWTBlock`s with wavelet
    down/up-sampling and ``1/√2``-scaled skips → inverse tokenizer. Operates on
    ``(in_channels, *spatial)`` with ``num_spatial_dims`` spatial axes; each
    extent must be divisible by ``patch_size * 2**(depth + 1)``.

    Attributes:
        tokenizer: Strided conv patchify ``in_channels -> hidden``.
        encoder_blocks, downsamples: Per-scale blocks and wavelet downsamplers.
        bottleneck: Blocks at the coarsest scale.
        upsamples, decoder_blocks: Per-scale wavelet upsamplers and blocks.
        untokenizer: Transposed conv un-patchify ``hidden -> out_channels``.
        skip_scale: ``1/√2`` skip multiplier.
        num_spatial_dims, in_channels, out_channels, hidden_channels: Static.
        depth, patch_size: Static config.

    Examples:
        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.mswt import MSWT
        >>> op = MSWT.init(3, 1, num_spatial_dims=2, key=jr.PRNGKey(0),
        ...                hidden_channels=16, depth=2, n_blocks=1)
        >>> op(jnp.ones((3, 32, 32))).shape
        (1, 32, 32)
        >>> jax.vmap(op)(jnp.ones((4, 3, 32, 32))).shape
        (4, 1, 32, 32)
    """

    tokenizer: eqx.nn.Conv
    encoder_blocks: list[list[MSWTBlock]]
    downsamples: list[WaveletDownsample]
    bottleneck: list[MSWTBlock]
    upsamples: list[WaveletUpsample]
    decoder_blocks: list[list[MSWTBlock]]
    untokenizer: eqx.nn.ConvTranspose
    skip_scale: float = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        hidden_channels: int = 64,
        depth: int = 2,
        n_blocks: int = 1,
        wavelet: Wavelet = "haar",
        patch_size: int = 1,
        window: int | None = None,
        heads: int = 4,
        dim_head: int = 32,
        groups: int = 8,
        ff_mult: int = 2,
    ) -> MSWT:
        """Construct an MSWT operator.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_spatial_dims: Number of spatial axes (1, 2, or 3).
            key: PRNG key.
            hidden_channels: Token/channel width (divisible by
                ``2**num_spatial_dims`` for the wavelet-attention split).
            depth: Number of wavelet down/up-sampling scales.
            n_blocks: `MSWTBlock`s per scale (encoder, bottleneck, decoder).
            wavelet: Wavelet family for the sub-band splits.
            patch_size: Patchify stride; ``1`` is a plain channel lift. Larger
                patches help long rollouts on big grids but hurt small grids.
            window: Swin window side for `WaveletAttention`; ``None`` is global.
            heads: Number of attention heads.
            dim_head: Channels per attention head.
            groups: GroupNorm groups for the block norms.
            ff_mult: Feed-forward expansion factor inside each block.

        Raises:
            ValueError: If ``num_spatial_dims < 1``, ``depth < 0``,
                ``n_blocks < 1``, ``patch_size < 1``, or ``hidden_channels`` is
                not divisible by ``2**num_spatial_dims``.
        """
        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims must be >= 1, got {num_spatial_dims}.")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}.")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}.")
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}.")
        factor = 2**num_spatial_dims
        if hidden_channels % factor != 0:
            raise ValueError(
                f"hidden_channels must be divisible by 2**num_spatial_dims "
                f"({factor}), got {hidden_channels}."
            )

        d = num_spatial_dims
        h = hidden_channels
        keys = iter(jax.random.split(key, 4 + (2 * depth + 1) * n_blocks + 2 * depth))

        def _block() -> MSWTBlock:
            return MSWTBlock.init(
                h,
                d,
                key=next(keys),
                wavelet=wavelet,
                heads=heads,
                dim_head=dim_head,
                groups=groups,
                window=window,
                ff_mult=ff_mult,
            )

        tokenizer = eqx.nn.Conv(
            d,
            in_channels,
            h,
            kernel_size=patch_size,
            stride=patch_size,
            key=next(keys),
        )
        encoder_blocks = [[_block() for _ in range(n_blocks)] for _ in range(depth)]
        downsamples = [
            WaveletDownsample.init(h, d, key=next(keys), wavelet=wavelet)
            for _ in range(depth)
        ]
        bottleneck = [_block() for _ in range(n_blocks)]
        upsamples = [
            WaveletUpsample.init(h, d, key=next(keys), wavelet=wavelet)
            for _ in range(depth)
        ]
        decoder_blocks = [[_block() for _ in range(n_blocks)] for _ in range(depth)]
        untokenizer = eqx.nn.ConvTranspose(
            d,
            h,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            key=next(keys),
        )
        return cls(
            tokenizer=tokenizer,
            encoder_blocks=encoder_blocks,
            downsamples=downsamples,
            bottleneck=bottleneck,
            upsamples=upsamples,
            decoder_blocks=decoder_blocks,
            untokenizer=untokenizer,
            skip_scale=2**-0.5,
            num_spatial_dims=d,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=h,
            depth=depth,
            patch_size=patch_size,
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        req = self.patch_size * 2 ** (self.depth + 1)
        if any(s % req for s in x.shape[1:]):
            raise ValueError(
                f"MSWT needs each spatial extent divisible by "
                f"patch_size * 2**(depth+1) = {req}, got {x.shape[1:]}."
            )

        x = self.tokenizer(x)
        skips: list[Array] = []
        for blocks, down in zip(self.encoder_blocks, self.downsamples, strict=True):
            for block in blocks:
                x = block(x)
            skips.append(x)
            x = down(x)

        for block in self.bottleneck:
            x = block(x)

        for up, blocks in zip(self.upsamples, self.decoder_blocks, strict=True):
            x = up(x)
            x = (x + skips.pop()) * self.skip_scale
            for block in blocks:
                x = block(x)

        return self.untokenizer(x)


MultiScaleWaveletTransformer = MSWT
"""Verbose alias for `MSWT`."""


__all__ = [
    "MSWT",
    "MSWTBlock",
    "MultiScaleWaveletTransformer",
    "WaveletDownsample",
    "WaveletUpsample",
]
