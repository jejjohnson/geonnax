"""Spherical harmonic transform and spherical spectral convolution.

The Spherical Fourier Neural Operator (Bonev et al., 2023) replaces the FFT of
a standard FNO with the Spherical Harmonic Transform (SHT), so the operator
lives on the sphere and respects its geometry — the right inductive bias for
global geoscience fields on a lat/lon grid.

The transform here is a dense, matrix-based SHT built directly from geonnax's
`real_spherical_harmonics` together with a
Gauss–Legendre latitude quadrature. It needs no external SHT package and is
fully differentiable. The (fixed) synthesis/analysis matrices are carried as
``stop_gradient`` buffers rather than trained parameters.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from geonnax._basis._spherical import harmonic_degrees, real_spherical_harmonics


def _gauss_legendre_grid(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build unit-sphere grid points and quadrature weights.

    Latitudes are Gauss–Legendre nodes in ``cos(theta)`` (so the ``sin(theta)``
    Jacobian is absorbed into the weights); longitudes are equispaced. Points
    are ordered latitude-major to match a ``(n_lat, n_lon)`` field reshaped
    row-major.

    Returns:
        ``(xyz, weights)`` with ``xyz`` of shape ``(n_lat * n_lon, 3)`` and
        ``weights`` of shape ``(n_lat * n_lon,)``.
    """
    z, w_lat = np.polynomial.legendre.leggauss(n_lat)  # nodes in [-1, 1]
    sin_theta = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    lon = 2.0 * np.pi * np.arange(n_lon) / n_lon
    cos_p, sin_p = np.cos(lon), np.sin(lon)

    x = sin_theta[:, None] * cos_p[None, :]
    y = sin_theta[:, None] * sin_p[None, :]
    zz = np.broadcast_to(z[:, None], (n_lat, n_lon))
    xyz = np.stack([x, y, zz], axis=-1).reshape(-1, 3)

    w_grid = np.broadcast_to(w_lat[:, None] * (2.0 * np.pi / n_lon), (n_lat, n_lon))
    return xyz, w_grid.reshape(-1)


class SphericalHarmonicTransform(eqx.Module):
    """Dense, differentiable spherical harmonic analysis/synthesis on a grid.

    Built for a fixed ``(n_lat, n_lon)`` grid and band limit ``l_max``; the
    coefficient layout matches `real_spherical_harmonics`
    (``(l_max + 1)**2`` real coefficients, outer loop over degree ``l``).

    Attributes:
        synth: Synthesis matrix ``(P, M)`` mapping coefficients → grid values.
        analy: Analysis matrix ``(M, P)`` mapping grid values → coefficients.
        degrees: Per-coefficient degree ``l`` (static).
        n_lat, n_lon, l_max: Grid/band configuration (static).

    Examples:
        Analysis and synthesis are inverse on band-limited fields:

        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import SphericalHarmonicTransform
        >>> sht = SphericalHarmonicTransform.init(n_lat=12, n_lon=24, l_max=8)
        >>> field = jnp.ones((2, 12, 24))
        >>> coeffs = sht.forward(field)          # (channels, (l_max+1)**2)
        >>> coeffs.shape
        (2, 81)
        >>> sht.inverse(coeffs).shape            # back to the grid
        (2, 12, 24)
        >>> # round-trip is (near) exact for band-limited signals:
        >>> c = jr.normal(jr.PRNGKey(0), (2, 81))
        >>> bool(jnp.allclose(sht.forward(sht.inverse(c)), c, atol=1e-4))
        True
    """

    synth: Float[Array, "P M"]
    analy: Float[Array, "M P"]
    degrees: tuple[int, ...] = eqx.field(static=True)
    n_lat: int = eqx.field(static=True)
    n_lon: int = eqx.field(static=True)
    l_max: int = eqx.field(static=True)

    @classmethod
    def init(cls, n_lat: int, n_lon: int, l_max: int) -> SphericalHarmonicTransform:
        """Construct the transform matrices for a Gauss–Legendre lat/lon grid.

        The grid must resolve the band limit, otherwise the quadrature is
        under-resolved and the analysis/synthesis pair is not invertible on the
        requested ``l_max`` (it would silently alias high modes): a
        Gauss–Legendre rule needs ``n_lat > l_max`` (exact to degree
        ``2 * n_lat - 1``) and equispaced longitudes need ``n_lon > 2 * l_max``
        to represent every order ``m``.

        Args:
            n_lat: Number of Gauss–Legendre latitudes (must be ``> l_max``).
            n_lon: Number of equispaced longitudes (must be ``> 2 * l_max``).
            l_max: Maximum spherical-harmonic degree.

        Raises:
            ValueError: If ``l_max < 0`` or the grid under-resolves ``l_max``.
        """
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}.")
        if n_lat <= l_max:
            raise ValueError(
                "n_lat must exceed l_max for an exact Gauss–Legendre quadrature; "
                f"got n_lat={n_lat}, l_max={l_max}."
            )
        if n_lon <= 2 * l_max:
            raise ValueError(
                "n_lon must exceed 2 * l_max to resolve every order m; "
                f"got n_lon={n_lon}, l_max={l_max}."
            )
        xyz, weights = _gauss_legendre_grid(n_lat, n_lon)
        synth = np.asarray(real_spherical_harmonics(jnp.asarray(xyz), l_max))
        analy = (synth * weights[:, None]).T
        return cls(
            synth=jnp.asarray(synth),
            analy=jnp.asarray(analy),
            degrees=harmonic_degrees(l_max),
            n_lat=n_lat,
            n_lon=n_lon,
            l_max=l_max,
        )

    def forward(self, field: Float[Array, "C n_lat n_lon"]) -> Float[Array, "C M"]:
        """Analyse a grid field into spherical-harmonic coefficients."""
        flat = field.reshape(field.shape[0], -1)
        return flat @ jax.lax.stop_gradient(self.analy).T

    def inverse(self, coeffs: Float[Array, "C M"]) -> Float[Array, "C n_lat n_lon"]:
        """Synthesise a grid field from spherical-harmonic coefficients."""
        flat = coeffs @ jax.lax.stop_gradient(self.synth).T
        return flat.reshape(coeffs.shape[0], self.n_lat, self.n_lon)


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
        >>> from geonnax.layers import (
        ...     SphericalHarmonicTransform, SphericalSpectralConv)
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


__all__ = ["SphericalHarmonicTransform", "SphericalSpectralConv"]
