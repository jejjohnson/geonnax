"""Per-scale channel mixing in the spherical-wavelet domain.

`SphericalWaveletConv` analyses a field into needlet scales with
`functax.transform.SphericalWaveletTransform`, mixes channels with one matrix
per scale and synthesises the result. The transform itself moved to functax
(jejjohnson/geonnax#158); its old geonnax paths resolve for one release, with a
`DeprecationWarning`.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from functax.transform import SphericalWaveletTransform
from jaxtyping import Array, Float


class SphericalWaveletConv(eqx.Module):
    """Per-scale channel mixing in the spherical-wavelet domain.

    The wavelet analogue of `SphericalSpectralConv`: analyse the input into
    scale maps, mix channels with a separate matrix per scale, then synthesise.
    The shared transform is passed at call time so it can be reused across
    blocks.

    Attributes:
        weight: Per-scale mixing of shape ``(n_scales, out_channels,
            in_channels)``.
        bias: Optional bias of shape ``(out_channels, 1, 1)``.
        in_channels, out_channels, n_scales: Static config.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from functax.transform import (
        ...     SphericalHarmonicTransform, SphericalWaveletTransform)
        >>> from geonnax.layers import SphericalWaveletConv
        >>> sht = SphericalHarmonicTransform.init(16, 32, 12)
        >>> swt = SphericalWaveletTransform.init(sht)
        >>> conv = SphericalWaveletConv.init(3, 5, swt.n_scales, key=jr.PRNGKey(0))
        >>> conv(jnp.ones((3, 16, 32)), swt).shape
        (5, 16, 32)
    """

    weight: Float[Array, "S C_out C_in"]
    bias: Float[Array, "C_out 1 1"] | None
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_scales: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        n_scales: int,
        *,
        key: Array,
        bias: bool = True,
        init_scale: float | None = None,
    ) -> SphericalWaveletConv:
        """Construct a spherical wavelet convolution over ``n_scales`` scales."""
        scale = (
            math.sqrt(2.0 / (in_channels + out_channels))
            if init_scale is None
            else init_scale
        )
        weight = scale * jax.random.normal(key, (n_scales, out_channels, in_channels))
        bias_arr = jnp.zeros((out_channels, 1, 1)) if bias else None
        return cls(
            weight=weight,
            bias=bias_arr,
            in_channels=in_channels,
            out_channels=out_channels,
            n_scales=n_scales,
        )

    def __call__(
        self,
        x: Float[Array, "C_in n_lat n_lon"],
        transform: SphericalWaveletTransform,
    ) -> Float[Array, "C_out n_lat n_lon"]:
        scales = transform.forward(x)  # (S, in, lat, lon)
        # Per-scale channel mix: (S, out, in) x (S, in, ...) -> (S, out, ...).
        mixed = jnp.einsum("soi,si...->so...", self.weight, scales)
        out = transform.inverse(mixed)
        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = ["SphericalWaveletConv"]
