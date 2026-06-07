"""Spherical wavelet (scale-discretised / needlet) transform.

A multiresolution analysis on the sphere built on the existing
`SphericalHarmonicTransform`. Each wavelet scale is a band-pass filter in
harmonic space: a smooth window selects a range of degrees $\\ell$, and the
windowed coefficients are synthesised back to a full-resolution map. With a
dyadic (octave) tiling these are the axisymmetric *needlets* / *scale-discretised
wavelets* used for localised analysis of global geoscience and CMB fields
(Marinucci et al., 2008; Wiaux et al., 2008).

The per-scale windows form a squared partition of unity,
$\\sum_j g_j(\\ell)^2 = 1$, built from the Meyer interpolating polynomial. That
makes analysis followed by synthesis exact (up to the harmonic transform's own
accuracy) — no tuning. Like the SHT, the windows are carried as fixed
``stop_gradient`` buffers rather than trained parameters.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from geonnax._basis._spherical import harmonic_degrees
from geonnax.layers._spherical_transform import SphericalHarmonicTransform


def _meyer_nu(t: np.ndarray) -> np.ndarray:
    """Meyer interpolating polynomial: 0 at ``t<=0``, 1 at ``t>=1``, smooth between.

    Satisfies ``nu(t) + nu(1 - t) == 1``, which makes the sine/cosine tapers
    below a squared partition of unity.
    """
    t = np.clip(t, 0.0, 1.0)
    return t**4 * (35.0 - 84.0 * t + 70.0 * t**2 - 20.0 * t**3)


def _partition_filters(l_max: int, n_scales: int | None = None) -> np.ndarray:
    r"""Dyadic harmonic band-pass windows over degrees ``0..l_max``.

    Tiles ``x = log2(max(l, 1))`` into unit (octave) bands. Within band
    ``[k, k+1]`` exactly two scales overlap, a cosine taper handing off to a
    sine taper, so their squares sum to one. Scale 0 also captures the lowest
    degrees (a scaling function).

    With the default ``n_scales`` (full octave coverage) the squared windows
    form a partition of unity, ``sum_j g_j(l)**2 == 1`` for every degree, which
    is what makes the transform's round-trip exact. Passing a smaller
    ``n_scales`` drops the highest scales, leaving the finest (high-degree) end
    of the spectrum uncovered, so the partition (and exact reconstruction) no
    longer holds.

    Returns:
        ``(n_scales, l_max + 1)`` array of non-negative window weights.
    """
    degrees = np.arange(l_max + 1)
    x = np.log2(np.maximum(degrees, 1))
    j_max = int(np.ceil(np.log2(max(l_max, 2))))
    n = j_max + 1 if n_scales is None else n_scales

    filters = np.zeros((n, l_max + 1))
    for k in range(n):
        u = x - k
        g = np.zeros_like(x)
        rising = (u > -1.0) & (u <= 0.0)  # hand-off from the previous scale
        g[rising] = np.sin(0.5 * np.pi * _meyer_nu(u[rising] + 1.0))
        falling = (u > 0.0) & (u < 1.0)  # hand-off to the next scale
        g[falling] = np.cos(0.5 * np.pi * _meyer_nu(u[falling]))
        if k == 0:
            g[x <= 0.0] = 1.0  # coarsest scale has no lower neighbour
        filters[k] = g
    return filters


class SphericalWaveletTransform(eqx.Module):
    """Scale-discretised (needlet) wavelet analysis/synthesis on the sphere.

    Wraps a `SphericalHarmonicTransform` and a set of harmonic band-pass
    windows. Analysis maps a field to one full-resolution band-pass map per
    scale; synthesis sums them back. Reconstruction is exact because the squared
    windows partition unity.

    Attributes:
        sht: The underlying spherical harmonic transform.
        filters: ``(n_scales, l_max + 1)`` harmonic windows (buffer).
        degree_index: Per-coefficient degree, as an int array (buffer).
        n_scales: Number of wavelet scales (static).

    Examples:
        Analysis returns one map per scale and reconstructs exactly:

        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import SphericalHarmonicTransform
        >>> from geonnax.layers._spherical_wavelet import SphericalWaveletTransform
        >>> sht = SphericalHarmonicTransform.init(16, 32, 12)
        >>> swt = SphericalWaveletTransform.init(sht)
        >>> field = jnp.ones((2, 16, 32))
        >>> scales = swt.forward(field)        # (n_scales, channels, lat, lon)
        >>> scales.shape[1:]
        (2, 16, 32)
        >>> bool(jnp.allclose(swt.inverse(scales), field, atol=1e-4))
        True
    """

    sht: SphericalHarmonicTransform
    filters: Float[Array, "S L"]
    degree_index: Array
    n_scales: int = eqx.field(static=True)

    @classmethod
    def init(
        cls, sht: SphericalHarmonicTransform, *, n_scales: int | None = None
    ) -> SphericalWaveletTransform:
        """Build wavelet windows for the band limit of ``sht``.

        Args:
            sht: A constructed `SphericalHarmonicTransform`.
            n_scales: Number of dyadic scales; defaults to
                ``ceil(log2(l_max)) + 1`` (full octave coverage).
        """
        filters = _partition_filters(sht.l_max, n_scales)
        degree_index = jnp.asarray(harmonic_degrees(sht.l_max), dtype=jnp.int32)
        return cls(
            sht=sht,
            filters=jnp.asarray(filters),
            degree_index=degree_index,
            n_scales=filters.shape[0],
        )

    def _windows(self) -> Float[Array, "S M"]:
        """Per-coefficient window weights ``(n_scales, n_coeffs)`` (frozen)."""
        return jax.lax.stop_gradient(self.filters)[:, self.degree_index]

    def forward(
        self, field: Float[Array, "C n_lat n_lon"]
    ) -> Float[Array, "S C n_lat n_lon"]:
        """Analyse a field into one band-pass map per wavelet scale."""
        coeffs = self.sht.forward(field)  # (C, M)
        windowed = self._windows()[:, None, :] * coeffs[None]  # (S, C, M)
        return jax.vmap(self.sht.inverse)(windowed)  # (S, C, n_lat, n_lon)

    def inverse(
        self, scales: Float[Array, "S C n_lat n_lon"]
    ) -> Float[Array, "C n_lat n_lon"]:
        """Synthesise a field from its per-scale wavelet maps."""
        coeffs = jax.vmap(self.sht.forward)(scales)  # (S, C, M)
        combined = jnp.sum(self._windows()[:, None, :] * coeffs, axis=0)  # (C, M)
        return self.sht.inverse(combined)


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
        >>> from geonnax.layers import SphericalHarmonicTransform
        >>> from geonnax.layers._spherical_wavelet import (
        ...     SphericalWaveletConv, SphericalWaveletTransform)
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


__all__ = ["SphericalWaveletConv", "SphericalWaveletTransform"]
