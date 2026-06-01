"""Spherical Fourier Neural Operator (Bonev et al., 2023).

The SFNO is the FNO with its planar FFT replaced by a Spherical Harmonic
Transform, so the learned operator acts on functions on the sphere — the
appropriate geometry for global geoscience fields sampled on a lat/lon grid. It
shares the FNO's structure: lift to a hidden width, apply spherical Fourier
blocks (a :class:`~geonnax.layers.SphericalSpectralConv` summed with a pointwise
skip), then project to the output channels.

A single :class:`~geonnax.layers.SphericalHarmonicTransform` is built for the
input grid and shared across all blocks. Inputs are single examples of shape
``(channels, n_lat, n_lon)`` matching that grid; ``jax.vmap`` over a batch.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float

from geonnax.layers._spherical_transform import (
    SphericalHarmonicTransform,
    SphericalSpectralConv,
)


def _pointwise(in_channels: int, out_channels: int, *, key: Array) -> eqx.nn.Conv:
    """A ``1x1`` convolution over the ``(n_lat, n_lon)`` grid."""
    return eqx.nn.Conv(
        num_spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        key=key,
    )


class SphericalFNOBlock(eqx.Module):
    """One spherical Fourier layer: spherical spectral conv + pointwise skip.

    The shared transform is threaded in at call time (see :class:`SFNO`).

    Attributes:
        spectral: Spherical spectral mixing (:class:`SphericalSpectralConv`).
        pointwise: Local ``1x1`` channel-mixing skip.
        activation: Pointwise nonlinearity (static).
        use_activation: Whether to apply the activation (static).
    """

    spectral: SphericalSpectralConv
    pointwise: eqx.nn.Conv
    activation: Callable[[Array], Array] = eqx.field(static=True)
    use_activation: bool = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        channels: int,
        l_max: int,
        *,
        key: Array,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        use_activation: bool = True,
    ) -> SphericalFNOBlock:
        """Construct a spherical Fourier block at fixed width ``channels``."""
        k_sp, k_pw = jax.random.split(key)
        spectral = SphericalSpectralConv.init(channels, channels, l_max, key=k_sp)
        pointwise = _pointwise(channels, channels, key=k_pw)
        return cls(
            spectral=spectral,
            pointwise=pointwise,
            activation=activation,
            use_activation=use_activation,
        )

    def __call__(
        self,
        x: Float[Array, "C n_lat n_lon"],
        sht: SphericalHarmonicTransform,
    ) -> Float[Array, "C n_lat n_lon"]:
        h = self.spectral(x, sht) + self.pointwise(x)
        return self.activation(h) if self.use_activation else h


class SFNO(eqx.Module):
    """Spherical Fourier Neural Operator: lift → spherical blocks → project.

    Operates on ``(in_channels, n_lat, n_lon)`` fields on the transform's
    Gauss–Legendre grid; ``jax.vmap`` over a batch.

    Attributes:
        sht: The shared spherical harmonic transform.
        lifting: Pointwise map ``in_channels -> hidden_channels``.
        blocks: The spherical Fourier layers.
        proj1, proj2: Two-layer pointwise projection ``hidden -> proj -> out``.
        activation: Pointwise nonlinearity (static).
        in_channels, out_channels, hidden_channels, l_max: Static config.
    """

    sht: SphericalHarmonicTransform
    lifting: eqx.nn.Conv
    blocks: list[SphericalFNOBlock]
    proj1: eqx.nn.Conv
    proj2: eqx.nn.Conv
    activation: Callable[[Array], Array] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    l_max: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        *,
        key: Array,
        n_lat: int,
        n_lon: int,
        l_max: int,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
    ) -> SFNO:
        """Construct an SFNO.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            key: PRNG key.
            n_lat: Number of Gauss–Legendre latitudes of the input grid.
            n_lon: Number of equispaced longitudes of the input grid.
            l_max: Spherical-harmonic band limit (use ``l_max < n_lat``).
            hidden_channels: Channel width of the spherical blocks.
            n_layers: Number of spherical Fourier blocks.
            projection_channels: Hidden width of the projection MLP (defaults to
                ``hidden_channels``).
            activation: Pointwise nonlinearity.

        Raises:
            ValueError: If ``n_layers < 1``.
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        projection_channels = projection_channels or hidden_channels

        sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
        k_lift, k_p1, k_p2, k_blocks = jax.random.split(key, 4)
        lifting = _pointwise(in_channels, hidden_channels, key=k_lift)
        bkeys = jax.random.split(k_blocks, n_layers)
        blocks = [
            SphericalFNOBlock.init(
                hidden_channels,
                l_max,
                key=bkeys[i],
                activation=activation,
                use_activation=i < n_layers - 1,
            )
            for i in range(n_layers)
        ]
        proj1 = _pointwise(hidden_channels, projection_channels, key=k_p1)
        proj2 = _pointwise(projection_channels, out_channels, key=k_p2)
        return cls(
            sht=sht,
            lifting=lifting,
            blocks=blocks,
            proj1=proj1,
            proj2=proj2,
            activation=activation,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            l_max=l_max,
        )

    def __call__(
        self, x: Float[Array, "C_in n_lat n_lon"]
    ) -> Float[Array, "C_out n_lat n_lon"]:
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x, self.sht)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)
        return x


__all__ = ["SFNO", "SphericalFNOBlock"]
