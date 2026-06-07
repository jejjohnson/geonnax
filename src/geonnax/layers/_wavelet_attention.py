"""Wavelet-domain self-attention — the MSWT core (Wang et al., 2026).

`WaveletAttention` runs self-attention *inside* the wavelet domain instead of
on the physical grid. It compresses channels, applies a single-level DWT to
expose the ``2**d`` sub-bands (in 2D: LL, LH, HL, HH), mixes them with a local
convolution and global attention, then inverts the transform. Because the
high-frequency detail bands are explicit in the attended representation, they
cannot be silently smoothed away — the inductive bias the paper uses to fight
the spectral bias of grid-space attention on chaotic / long-rollout dynamics.

The transform, sub-band layout, and orthonormal filter bank are reused from
`geonnax.layers._wavelet`; attention is the existing `Attention` (global) or
`WindowedAttention` (Swin-style, for large grids). This is a composition of
primitives geonnax already owns — no new transform or custom autograd.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers._attention import Attention, WindowedAttention
from geonnax.layers._conv import StandardizedConv
from geonnax.layers._wavelet import Wavelet, _flatten, _unflatten, dwt, idwt


class WaveletAttention(eqx.Module):
    """Self-attention performed inside the wavelet domain (Wang et al., 2026).

    Compresses channels by ``2**num_spatial_dims``, applies a single-level DWT
    to expose the sub-bands (stacking them restores the full width at half
    resolution), mixes them with a ``3x3`` conv and self-attention, then inverts
    the transform and restores the channel width. Shape-preserving:
    ``(C, *spatial) -> (C, *spatial)``. Each spatial extent must be even.

    Attributes:
        proj_in: Channel compression ``dim -> dim // 2**d``.
        proj_out: Channel restoration ``dim // 2**d -> dim``.
        band_conv: Local ``3x3`` mixing across the stacked sub-bands.
        attention: Global `Attention` or Swin `WindowedAttention` over the bands.
        wavelet: Wavelet family used for the sub-band split (static).
        num_spatial_dims: Number of spatial axes (static).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import WaveletAttention
        >>> wa = WaveletAttention.init(16, 2, key=jr.PRNGKey(0))
        >>> wa(jnp.ones((16, 32, 32))).shape
        (16, 32, 32)
    """

    proj_in: StandardizedConv
    proj_out: StandardizedConv
    band_conv: StandardizedConv
    attention: Attention | WindowedAttention
    wavelet: Wavelet = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)

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
    ) -> WaveletAttention:
        """Construct a wavelet-attention layer.

        Args:
            dim: Channel width (must be divisible by ``2**num_spatial_dims``).
            num_spatial_dims: Number of spatial axes (1, 2, or 3).
            key: PRNG key.
            wavelet: Wavelet family for the sub-band split (``"haar"`` default,
                matching the paper; any `geonnax.layers.Wavelet` is accepted).
            heads: Attention heads.
            dim_head: Channels per attention head.
            groups: GroupNorm groups for the attention pre-norm.
            window: If given, use Swin-style `WindowedAttention` with this window
                side (``O(N * window**d)``); otherwise global `Attention`
                (``O(N**2)``, suitable for modest grids only).

        Raises:
            ValueError: If ``num_spatial_dims < 1`` or ``dim`` is not divisible
                by ``2**num_spatial_dims``.
        """
        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims must be >= 1, got {num_spatial_dims}.")
        factor = 2**num_spatial_dims
        if dim % factor == 0:
            compressed = dim // factor
        else:
            raise ValueError(
                f"dim must be divisible by 2**num_spatial_dims ({factor}), "
                f"got dim={dim}."
            )
        k_in, k_band, k_attn, k_out = jax.random.split(key, 4)
        proj_in = StandardizedConv.init(num_spatial_dims, dim, compressed, 1, key=k_in)
        band_conv = StandardizedConv.init(
            num_spatial_dims, dim, dim, 3, key=k_band, padding=1
        )
        attention: Attention | WindowedAttention
        if window is None:
            attention = Attention.init(
                dim,
                num_spatial_dims,
                key=k_attn,
                heads=heads,
                dim_head=dim_head,
                groups=groups,
            )
        else:
            attention = WindowedAttention.init(
                dim,
                num_spatial_dims,
                window=window,
                key=k_attn,
                heads=heads,
                dim_head=dim_head,
                groups=groups,
            )
        proj_out = StandardizedConv.init(
            num_spatial_dims, compressed, dim, 1, key=k_out
        )
        return cls(
            proj_in=proj_in,
            proj_out=proj_out,
            band_conv=band_conv,
            attention=attention,
            wavelet=wavelet,
            num_spatial_dims=num_spatial_dims,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        spatial = x.shape[1:]
        if any(s % 2 for s in spatial):
            raise ValueError(
                f"WaveletAttention needs even spatial extents, got {spatial}."
            )
        axes = tuple(range(1, self.num_spatial_dims + 1))
        factor = 2**self.num_spatial_dims

        h = self.proj_in(x)  # (dim // factor, *spatial)
        bands, layout = _flatten(dwt(h, self.wavelet, 1, axes))
        stacked = jnp.concatenate(bands, axis=0)  # (dim, *half)

        y = self.band_conv(stacked)
        y = y + self.attention(y)  # local then global mixing over the sub-bands

        coeffs = _unflatten(list(jnp.split(y, factor, axis=0)), layout)
        out = idwt(coeffs, self.wavelet, axes)  # (dim // factor, *spatial)
        return self.proj_out(out)


__all__ = ["WaveletAttention"]
