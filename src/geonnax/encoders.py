"""Deterministic coordinate encoders for lon/lat and spherical inputs.

Pure ``equinox.Module`` wrappers around the helpers in
:mod:`geonnax.geo` and :mod:`geonnax._basis`. No PRNG, no learnable
parameters — they compose into ``eqx.nn.Sequential``.

All encoders take a single example as input; use :func:`jax.vmap` to
batch over leading dimensions.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from geonnax._basis import real_spherical_harmonics
from geonnax.geo import (
    _validate_input_unit,
    _validate_range,
    cyclic_encode,
    deg2rad,
    lonlat_scale,
    lonlat_to_cartesian3d,
)


class Deg2Rad(eqx.Module):
    """Element-wise degrees-to-radians conversion.

    Stateless wrapper around :func:`geonnax.geo.deg2rad` — no learnable
    parameters.

    Example:
        >>> import jax.numpy as jnp
        >>> Deg2Rad()(jnp.array([0.0, 90.0, 180.0]))
        Array([0.       , 1.5707964, 3.1415927], dtype=float32)
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        return deg2rad(x)


class LonLatScale(eqx.Module):
    """Affine-rescale a single lon/lat pair into ``[-1, 1]``.

    Values inside the given ranges map into ``[-1, 1]``; out-of-range
    values are *not* clipped. The default ranges assume ``lonlat`` is
    in degrees.

    Attributes:
        lon_range: ``(min, max)`` longitude domain (must satisfy
            ``min < max``).
        lat_range: ``(min, max)`` latitude domain (must satisfy
            ``min < max``).
    """

    lon_range: tuple[float, float] = eqx.field(static=True, default=(-180.0, 180.0))
    lat_range: tuple[float, float] = eqx.field(static=True, default=(-90.0, 90.0))

    def __post_init__(self) -> None:
        _validate_range(self.lon_range, name="lon_range")
        _validate_range(self.lat_range, name="lat_range")

    def __call__(self, lonlat: Num[Array, " 2"]) -> Float[Array, " 2"]:
        out = lonlat_scale(
            lonlat[None, :],
            lon_range=self.lon_range,
            lat_range=self.lat_range,
        )
        return out[0]


class Cartesian3DEncoder(eqx.Module):
    """Lift a single lon/lat coordinate onto the unit sphere :math:`S^2`.

    Stateless wrapper around :func:`geonnax.geo.lonlat_to_cartesian3d`.

    Attributes:
        input_unit: Whether the input is in ``"degrees"`` or
            ``"radians"``.
    """

    input_unit: Literal["degrees", "radians"] = eqx.field(
        static=True, default="radians"
    )

    def __post_init__(self) -> None:
        _validate_input_unit(self.input_unit)

    def __call__(self, lonlat: Float[Array, " 2"]) -> Float[Array, " 3"]:
        out = lonlat_to_cartesian3d(lonlat[None, :], input_unit=self.input_unit)
        return out[0]


class CyclicEncoder(eqx.Module):
    """Encode a single periodic input as concatenated cos/sin features.

    Stateless wrapper around :func:`geonnax.geo.cyclic_encode`.
    """

    def __call__(
        self,
        angles: Float[Array, ""] | Float[Array, " D"],
    ) -> Float[Array, " F"]:
        promoted = jnp.atleast_1d(angles)
        out = cyclic_encode(promoted[None, :])
        return out[0]


class SphericalHarmonicEncoder(eqx.Module):
    """Real spherical-harmonic features on the unit 2-sphere for a single point.

    Stateless wrapper that evaluates
    :func:`geonnax._basis.real_spherical_harmonics` on either an already-
    cartesian input (``input_mode='cartesian'``) or a lon/lat pair
    (``input_mode='lonlat'``, assumed in radians).

    Attributes:
        l_max: Maximum harmonic degree (must be ``>= 0``). The output
            has ``(l_max + 1) ** 2`` features.
        input_mode: ``"cartesian"`` for a ``(3,)`` unit-sphere input
            or ``"lonlat"`` for a ``(2,)`` lon/lat pair in radians.
    """

    l_max: int = eqx.field(static=True)
    input_mode: Literal["cartesian", "lonlat"] = eqx.field(
        static=True, default="cartesian"
    )

    def __post_init__(self) -> None:
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0; got {self.l_max}.")
        if self.input_mode not in {"cartesian", "lonlat"}:
            raise ValueError(
                f"input_mode must be 'cartesian' or 'lonlat'; got {self.input_mode!r}."
            )

    @property
    def num_features(self) -> int:
        return (self.l_max + 1) ** 2

    def __call__(
        self,
        x: Float[Array, " 3"] | Float[Array, " 2"],
    ) -> Float[Array, " M"]:
        if self.input_mode == "cartesian":
            if x.ndim != 1 or x.shape[-1] != 3:
                raise ValueError(
                    f"x must be (3,) when input_mode='cartesian'; got shape {x.shape}."
                )
            unit_xyz = x[None, :]
        else:
            if x.ndim != 1 or x.shape[-1] != 2:
                raise ValueError(
                    f"x must be (2,) when input_mode='lonlat'; got shape {x.shape}."
                )
            unit_xyz = lonlat_to_cartesian3d(x[None, :], input_unit="radians")
        return real_spherical_harmonics(unit_xyz, l_max=self.l_max)[0]


__all__ = [
    "Cartesian3DEncoder",
    "CyclicEncoder",
    "Deg2Rad",
    "LonLatScale",
    "SphericalHarmonicEncoder",
]
