"""Spherical spectral convolution: channel mixing per spherical-harmonic degree.

The Spherical Fourier Neural Operator (Bonev et al., 2023) replaces the FFT of
a standard FNO with the Spherical Harmonic Transform (SHT), so the operator
lives on the sphere and respects its geometry — the right inductive bias for
global geoscience fields on a lat/lon grid.

The transform is `functax.transform.SphericalHarmonicTransform`, a dense,
differentiable matrix SHT on a Gauss–Legendre grid; it moved there from this
module (jejjohnson/geonnax#153). The convolution below is the network half.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from functax.basis.sphere import harmonic_degrees
from functax.transform import SphericalHarmonicTransform
from jaxtyping import Array, Float


class SphericalSpectralConv(eqx.Module):
    """Spherical spectral convolution: per-degree channel mixing in SH space.

    Mixes channels with a separate matrix per spherical-harmonic degree ``l``
    (shared across orders ``m``), which is the rotation-equivariant analogue of
    the FNO's spectral mixing. The (heavyweight, shared) transform is passed in
    at call time rather than stored, so an `SFNO` can reuse one
    `SphericalHarmonicTransform` across all of its blocks.

    Attributes:
        weight: Per-degree mixing of shape ``(l_max + 1, out_channels,
            in_channels)``.
        bias: Optional bias of shape ``(out_channels, 1, 1)``.
        degree_index: Per-coefficient degree, as an int array for gathering.
        in_channels, out_channels, l_max: Static config.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from functax.transform import SphericalHarmonicTransform
        >>> from geonnax.layers import SphericalSpectralConv
        >>> sht = SphericalHarmonicTransform.init(12, 24, 8)
        >>> conv = SphericalSpectralConv.init(3, 5, 8, key=jr.PRNGKey(0))
        >>> conv(jnp.ones((3, 12, 24)), sht).shape   # 3 -> 5 channels
        (5, 12, 24)
    """

    weight: Float[Array, "L C_out C_in"]
    bias: Float[Array, "C_out 1 1"] | None
    degree_index: Array
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    l_max: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        l_max: int,
        *,
        key: Array,
        bias: bool = True,
        init_scale: float | None = None,
    ) -> SphericalSpectralConv:
        """Construct a spherical spectral convolution for band limit ``l_max``."""
        scale = (
            math.sqrt(2.0 / (in_channels + out_channels))
            if init_scale is None
            else init_scale
        )
        weight = scale * jax.random.normal(key, (l_max + 1, out_channels, in_channels))
        bias_arr = jnp.zeros((out_channels, 1, 1)) if bias else None
        degree_index = jnp.asarray(harmonic_degrees(l_max), dtype=jnp.int32)
        return cls(
            weight=weight,
            bias=bias_arr,
            degree_index=degree_index,
            in_channels=in_channels,
            out_channels=out_channels,
            l_max=l_max,
        )

    def __call__(
        self,
        x: Float[Array, "C_in n_lat n_lon"],
        sht: SphericalHarmonicTransform,
    ) -> Float[Array, "C_out n_lat n_lon"]:
        coeffs = sht.forward(x)  # (in, M)
        w_per_coeff = self.weight[self.degree_index]  # (M, out, in)
        out_coeffs = jnp.einsum("moi,im->om", w_per_coeff, coeffs)  # (out, M)
        out = sht.inverse(out_coeffs)
        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = ["SphericalSpectralConv"]
