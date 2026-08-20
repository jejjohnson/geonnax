"""Deterministic coordinate encoders for lon/lat and spherical inputs.

Pure ``equinox.Module`` wrappers around the helpers in
`geonnax.geo` and `geonnax._basis`. No PRNG, no learnable
parameters — they compose into ``eqx.nn.Sequential``.

All encoders take a single example as input; use `jax.vmap` to
batch over leading dimensions.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Num

from geonnax._basis import real_spherical_harmonics
from geonnax.geo import (
    _promote_to_floating,
    _validate_input_unit,
    _validate_range,
    cyclic_encode,
    deg2rad,
    lonlat_scale,
    lonlat_to_cartesian3d,
)


def _as_scalar(value: Num[Array, ""] | float, name: str) -> Float[Array, ""]:
    """Coerce a scalar input to a floating 0-d array, rejecting arrays."""
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(
            f"{name} must be a scalar; got shape {array.shape}. "
            "Use jax.vmap to encode a batch of observations."
        )
    return _promote_to_floating(array)


class Deg2Rad(eqx.Module):
    """Element-wise degrees-to-radians conversion.

    Stateless wrapper around `geonnax.geo.deg2rad` — no learnable
    parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.encoders import Deg2Rad
        >>> Deg2Rad()(jnp.array([0.0, 90.0, 180.0])).shape
        (3,)
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Convert ``x`` from degrees to radians, preserving its shape.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.encoders import Deg2Rad
            >>> Deg2Rad()(jnp.zeros((4,))).shape
            (4,)
        """
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

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.encoders import LonLatScale
        >>> LonLatScale()(jnp.array([180.0, 90.0])).shape
        (2,)
    """

    lon_range: tuple[float, float] = eqx.field(static=True, default=(-180.0, 180.0))
    lat_range: tuple[float, float] = eqx.field(static=True, default=(-90.0, 90.0))

    def __post_init__(self) -> None:
        _validate_range(self.lon_range, name="lon_range")
        _validate_range(self.lat_range, name="lat_range")

    def __call__(self, lonlat: Num[Array, " 2"]) -> Float[Array, " 2"]:
        """Rescale a single ``(2,)`` lon/lat pair into ``[-1, 1]``.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.encoders import LonLatScale
            >>> # Domain max maps to +1 in each column.
            >>> out = LonLatScale()(jnp.array([180.0, 90.0]))
            >>> bool(jnp.allclose(out, 1.0))
            True
        """
        # (2,) -> (1, 2) so the batched helper applies, then drop the batch dim.
        out = lonlat_scale(
            lonlat[None, :],
            lon_range=self.lon_range,
            lat_range=self.lat_range,
        )
        return out[0]  # (1, 2) -> (2,)


class Cartesian3DEncoder(eqx.Module):
    """Lift a single lon/lat coordinate onto the unit sphere $S^2$.

    Stateless wrapper around `geonnax.geo.lonlat_to_cartesian3d`.

    Attributes:
        input_unit: Whether the input is in ``"degrees"`` or
            ``"radians"``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.encoders import Cartesian3DEncoder
        >>> Cartesian3DEncoder()(jnp.array([0.0, 0.0])).shape
        (3,)
    """

    input_unit: Literal["degrees", "radians"] = eqx.field(
        static=True, default="radians"
    )

    def __post_init__(self) -> None:
        _validate_input_unit(self.input_unit)

    def __call__(self, lonlat: Float[Array, " 2"]) -> Float[Array, " 3"]:
        """Map a single ``(2,)`` lon/lat pair to a ``(3,)`` unit vector.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.encoders import Cartesian3DEncoder
            >>> # (lon, lat) = (0, 0) → +x on the unit sphere.
            >>> out = Cartesian3DEncoder()(jnp.array([0.0, 0.0]))
            >>> bool(jnp.allclose(out, jnp.array([1.0, 0.0, 0.0]), atol=1e-6))
            True
        """
        # (2,) -> (1, 2) for the batched helper, then drop the batch dim.
        out = lonlat_to_cartesian3d(lonlat[None, :], input_unit=self.input_unit)
        return out[0]  # (1, 3) -> (3,)


class CyclicEncoder(eqx.Module):
    """Encode a single periodic input as concatenated cos/sin features.

    Stateless wrapper around `geonnax.geo.cyclic_encode`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax
        >>> from geonnax.encoders import CyclicEncoder
        >>> jax.vmap(CyclicEncoder())(jnp.array([0.0, jnp.pi])).shape
        (2, 2)
    """

    def __call__(
        self,
        angles: Float[Array, ""] | Float[Array, " D"],
    ) -> Float[Array, " F"]:
        """Encode a scalar or ``(D,)`` angle vector into ``[cos, sin]`` features.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.encoders import CyclicEncoder
            >>> # Scalar angle → 2 features (cos, sin).
            >>> CyclicEncoder()(jnp.array(0.0)).shape
            (2,)
            >>> # (D,) angle vector → 2·D features.
            >>> CyclicEncoder()(jnp.zeros((3,))).shape
            (6,)
        """
        promoted = jnp.atleast_1d(angles)  # () or (D,) -> (D,)
        out = cyclic_encode(promoted[None, :])  # (D,) -> (1, D) -> (1, 2·D)
        return out[0]  # (1, 2·D) -> (2·D,)


class SphericalHarmonicEncoder(eqx.Module):
    """Real spherical-harmonic features on the unit 2-sphere for a single point.

    Stateless wrapper that evaluates
    `geonnax._basis.real_spherical_harmonics` on either an already-
    cartesian input (``input_mode='cartesian'``) or a lon/lat pair
    (``input_mode='lonlat'``, assumed in radians).

    Attributes:
        l_max: Maximum harmonic degree (must be ``>= 0``). The output
            has ``(l_max + 1) ** 2`` features.
        input_mode: ``"cartesian"`` for a ``(3,)`` unit-sphere input
            or ``"lonlat"`` for a ``(2,)`` lon/lat pair in radians.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.encoders import SphericalHarmonicEncoder
        >>> # l_max=3 → (l_max + 1)^2 = 16 features for a (3,) cartesian input.
        >>> SphericalHarmonicEncoder(l_max=3)(jnp.array([1.0, 0.0, 0.0])).shape
        (16,)
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
        """Number of harmonic features, ``(l_max + 1) ** 2``.

        Examples:
            >>> from geonnax.encoders import SphericalHarmonicEncoder
            >>> SphericalHarmonicEncoder(l_max=2).num_features
            9
        """
        return (self.l_max + 1) ** 2  # degrees 0..l_max give Σ(2l+1) = (l_max+1)^2

    def __call__(
        self,
        x: Float[Array, " 3"] | Float[Array, " 2"],
    ) -> Float[Array, " M"]:
        """Evaluate real spherical harmonics at a single point.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.encoders import SphericalHarmonicEncoder
            >>> # lonlat mode: (2,) radian pair → (l_max + 1)^2 features.
            >>> enc = SphericalHarmonicEncoder(l_max=2, input_mode="lonlat")
            >>> enc(jnp.array([0.0, 0.0])).shape
            (9,)
        """
        if self.input_mode == "cartesian":
            if x.ndim != 1 or x.shape[-1] != 3:
                raise ValueError(
                    f"x must be (3,) when input_mode='cartesian'; got shape {x.shape}."
                )
            unit_xyz = x[None, :]  # (3,) -> (1, 3)
        else:
            if x.ndim != 1 or x.shape[-1] != 2:
                raise ValueError(
                    f"x must be (2,) when input_mode='lonlat'; got shape {x.shape}."
                )
            # (2,) lon/lat radians -> (1, 3) on the unit sphere.
            unit_xyz = lonlat_to_cartesian3d(x[None, :], input_unit="radians")
        # (1, 3) -> (1, (l_max+1)^2) -> drop batch dim -> ((l_max+1)^2,)
        return real_spherical_harmonics(unit_xyz, l_max=self.l_max)[0]


class GeoContextEncoder(eqx.Module):
    r"""Encode geoscience metadata into a single context vector.

    Combines longitude/latitude, day-of-year, and arbitrary physical
    covariates (wind, retrieval uncertainty, viewing geometry, …) into
    one feature vector suitable for conditioning a downstream model —
    for example the ``condition`` argument of a conditional
    normalizing flow or an anomaly scorer.

    Features are concatenated in the fixed order
    ``[lon/lat block, time block, extra block]``:

    | Block | Option | Features |
    | --- | --- | --- |
    | lon/lat | ``"spherical"`` | ``[cosϕ cosλ, cosϕ sinλ, sinϕ]`` (3) |
    | lon/lat | ``"sincos"`` | ``[cosλ, sinλ, ϕ/90]`` (3) |
    | lon/lat | ``"raw"`` | ``[lon, lat]`` rescaled to ``[-1, 1]`` (2) |
    | time | ``"cyclic"`` | ``[cos(2π d/P), sin(2π d/P)]`` (2) |
    | time | ``"raw"`` | ``[d/P]`` (1) |
    | extra | — | covariate values in **sorted key order** |

    where ``λ`` is longitude, ``ϕ`` is latitude, ``d`` is day of year
    and ``P`` is `period`. The cyclic blocks put cosine before sine,
    matching `geonnax.geo.cyclic_encode`.

    Inputs are in **degrees** (``lat ∈ [-90, 90]``,
    ``lon ∈ [-180, 180]``) with ``day_of_year ∈ [0, period)``. Values
    outside those ranges are *not* clipped — validation is shape-only,
    so the encoder stays ``jit``-safe. The ``"raw"`` longitude feature
    is discontinuous at the dateline; use ``"sincos"`` or
    ``"spherical"`` to avoid the jump.

    Attributes:
        use_latlon: Whether to encode longitude/latitude.
        latlon_encoding: One of ``"spherical"``, ``"sincos"``,
            ``"raw"``.
        use_time: Whether to encode day of year.
        time_encoding: One of ``"cyclic"``, ``"raw"``.
        period: Days in a full seasonal cycle (must be positive).
        include_extra: Whether to accept an ``extra`` covariate dict.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from geonnax.encoders import GeoContextEncoder
        >>> encoder = GeoContextEncoder()
        >>> # Single example: 3 spherical + 2 cyclic-time features.
        >>> encoder(lat=45.0, lon=10.0, day_of_year=200.0).shape
        (5,)
        >>> # Batch with jax.vmap; `extra` is a pytree of covariates.
        >>> batched = jax.vmap(
        ...     lambda lat, lon, doy, extra: encoder(
        ...         lat=lat, lon=lon, day_of_year=doy, extra=extra
        ...     )
        ... )
        >>> out = batched(
        ...     jnp.linspace(-60.0, 60.0, 4),
        ...     jnp.linspace(-120.0, 120.0, 4),
        ...     jnp.array([1.0, 90.0, 180.0, 270.0]),
        ...     {"wind": jnp.zeros((4, 2)), "sigma": jnp.ones((4,))},
        ... )
        >>> out.shape
        (4, 8)
        >>> encoder.output_dim(extra_dim=3)
        8
    """

    use_latlon: bool = eqx.field(static=True, default=True)
    latlon_encoding: Literal["spherical", "sincos", "raw"] = eqx.field(
        static=True, default="spherical"
    )
    use_time: bool = eqx.field(static=True, default=True)
    time_encoding: Literal["cyclic", "raw"] = eqx.field(static=True, default="cyclic")
    period: float = eqx.field(static=True, default=365.25)
    include_extra: bool = eqx.field(static=True, default=True)

    def __post_init__(self) -> None:
        if self.latlon_encoding not in {"spherical", "sincos", "raw"}:
            raise ValueError(
                "latlon_encoding must be one of 'spherical', 'sincos', 'raw'; "
                f"got {self.latlon_encoding!r}."
            )
        if self.time_encoding not in {"cyclic", "raw"}:
            hint = (
                " ('fourier' is planned but not yet implemented; compose "
                "geonnax.basis.seasonal_features for multi-harmonic features.)"
                if self.time_encoding == "fourier"
                else ""
            )
            raise ValueError(
                "time_encoding must be one of 'cyclic', 'raw'; "
                f"got {self.time_encoding!r}.{hint}"
            )
        if self.period <= 0.0:
            raise ValueError(f"period must be positive; got {self.period}.")
        if not (self.use_latlon or self.use_time or self.include_extra):
            raise ValueError(
                "at least one of use_latlon, use_time, include_extra must be "
                "enabled; got all disabled."
            )

    def output_dim(self, extra_dim: int = 0) -> int:
        """Return the number of context features produced by this encoder.

        Args:
            extra_dim: Total number of features contributed by the
                ``extra`` covariates. Must be ``0`` when
                `include_extra` is ``False``.

        Returns:
            The trailing dimension ``F`` of the encoded context vector.

        Examples:
            >>> from geonnax.encoders import GeoContextEncoder
            >>> GeoContextEncoder().output_dim()
            5
            >>> GeoContextEncoder(latlon_encoding="raw").output_dim(extra_dim=2)
            6
            >>> GeoContextEncoder(use_time=False).output_dim()
            3
        """
        if extra_dim < 0:
            raise ValueError(f"extra_dim must be non-negative; got {extra_dim}.")
        if extra_dim and not self.include_extra:
            raise ValueError(
                f"extra_dim must be 0 when include_extra=False; got {extra_dim}."
            )

        n_latlon = 0
        if self.use_latlon:
            n_latlon = 2 if self.latlon_encoding == "raw" else 3
        n_time = 0
        if self.use_time:
            n_time = 1 if self.time_encoding == "raw" else 2
        return n_latlon + n_time + extra_dim

    def _encode_latlon(self, lat: Array, lon: Array) -> Float[Array, " F"]:
        lonlat = jnp.stack([lon, lat])[None, :]  # () x2 -> (2,) -> (1, 2)
        if self.latlon_encoding == "raw":
            return lonlat_scale(lonlat)[0]  # (1, 2) -> (2,)
        if self.latlon_encoding == "spherical":
            # (1, 2) degrees -> (1, 3) on the unit sphere -> (3,)
            return lonlat_to_cartesian3d(lonlat, input_unit="degrees")[0]
        # "sincos": [cos λ, sin λ] from the shared helper, plus normalized ϕ.
        lon_cyc = cyclic_encode(deg2rad(lonlat[:, :1]))[0]  # (1, 1) -> (2,)
        return jnp.concatenate([lon_cyc, (lat / 90.0)[None]])  # (2,) + (1,) -> (3,)

    def _encode_time(self, day_of_year: Array) -> Float[Array, " F"]:
        scaled = day_of_year / self.period  # () -> () fraction of a cycle
        if self.time_encoding == "raw":
            return scaled[None]  # () -> (1,)
        angle = 2.0 * jnp.pi * scaled  # fraction -> radians
        return cyclic_encode(angle[None, None])[0]  # (1, 1) -> (1, 2) -> (2,)

    def __call__(
        self,
        *,
        lat: Num[Array, ""] | float | None = None,
        lon: Num[Array, ""] | float | None = None,
        day_of_year: Num[Array, ""] | float | None = None,
        extra: dict[str, Num[Array, "..."]] | None = None,
    ) -> Float[Array, " F"]:
        """Encode one observation's metadata into a context vector.

        Args:
            lat: Scalar latitude in degrees. Required when
                `use_latlon` is ``True``, forbidden otherwise.
            lon: Scalar longitude in degrees. Required when
                `use_latlon` is ``True``, forbidden otherwise.
            day_of_year: Scalar day of year. Required when `use_time`
                is ``True``, forbidden otherwise.
            extra: Optional covariate mapping. Each value is a scalar
                (one feature) or a ``(k,)`` vector (``k`` features);
                keys are concatenated in sorted order. Forbidden when
                `include_extra` is ``False``.

        Returns:
            Context vector of shape ``(F,)`` where ``F`` is
            `output_dim` evaluated with the total ``extra`` width.

        Raises:
            ValueError: If a required input is missing, a disabled
                input is supplied, or an input has the wrong shape.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.encoders import GeoContextEncoder
            >>> # Methane-style context: geometry, season, and covariates.
            >>> encoder = GeoContextEncoder(latlon_encoding="spherical")
            >>> context = encoder(
            ...     lat=31.7,
            ...     lon=-102.1,
            ...     day_of_year=204.0,
            ...     extra={"wind": jnp.array([3.2, -1.1]), "sigma": 12.0},
            ... )
            >>> context.shape
            (8,)
            >>> # Sea-surface context without any covariates.
            >>> sst = GeoContextEncoder(include_extra=False)
            >>> sst(lat=-20.0, lon=175.0, day_of_year=15.0).shape
            (5,)
        """
        blocks: list[Array] = []

        if self.use_latlon:
            if lat is None or lon is None:
                raise ValueError(
                    "lat and lon must both be provided when use_latlon=True."
                )
            lat_arr = _as_scalar(lat, name="lat")
            lon_arr = _as_scalar(lon, name="lon")
            blocks.append(self._encode_latlon(lat_arr, lon_arr))
        elif lat is not None or lon is not None:
            raise ValueError("lat and lon must be omitted when use_latlon=False.")

        if self.use_time:
            if day_of_year is None:
                raise ValueError("day_of_year must be provided when use_time=True.")
            blocks.append(self._encode_time(_as_scalar(day_of_year, "day_of_year")))
        elif day_of_year is not None:
            raise ValueError("day_of_year must be omitted when use_time=False.")

        if not self.include_extra and extra is not None:
            raise ValueError("extra must be omitted when include_extra=False.")
        if self.include_extra and extra is not None:
            for name in sorted(extra):  # sorted keys keep the layout deterministic
                value = jnp.asarray(extra[name])
                if value.ndim > 1:
                    raise ValueError(
                        f"extra[{name!r}] must be a scalar or a (k,) vector; "
                        f"got shape {value.shape}."
                    )
                blocks.append(jnp.atleast_1d(_promote_to_floating(value)))

        if not blocks:
            raise ValueError(
                "no context features were produced; provide extra covariates "
                "when use_latlon and use_time are both False."
            )
        return jnp.concatenate(blocks)  # blocks of (k_i,) -> (F,)


__all__ = [
    "Cartesian3DEncoder",
    "CyclicEncoder",
    "Deg2Rad",
    "GeoContextEncoder",
    "LonLatScale",
    "SphericalHarmonicEncoder",
]
