"""Pure-JAX geographic and spherical feature helpers.

These helpers are deterministic, pandas-free building blocks for
longitude/latitude preprocessing and spherical-harmonic feature maps.
The corresponding stateful encoders expose
the same transforms as `equinox.Module` instances so they
compose inside `equinox.nn.Sequential`.

The Earth frames (``lonlat_to_ecef``, ``ecef_to_lonlat``, ``enu_basis``,
``tangent_to_cartesian``, ``cartesian_to_tangent``, ``local_tangent_plane``,
``rotation_to_pole``) moved to `functax.geometry` (jejjohnson/geonnax#153),
and ``lonlat_to_cartesian3d`` moved there as ``lonlat_to_unit``
(jejjohnson/geonnax#158). Their names still resolve here for one release, with
geonnax's defaults and a `DeprecationWarning`. functax takes ``unit=`` in place
of ``input_unit=`` (with no default for the frames), and ``semi_major`` /
``flattening`` in place of the ellipsoid name.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
from functax.basis.sphere import real_spherical_harmonics
from functax.geometry import lonlat_to_unit
from jaxtyping import Array, Float, Num

from geonnax import _compat
from geonnax._deprecation import Moved, deprecated_dir, deprecated_getattr
from geonnax._earth import EARTH_MEAN_RADIUS_M, WGS84_A, WGS84_F


def _validate_lonlat_shape(lonlat: Float[Array, ...], *, name: str = "lonlat") -> None:
    if lonlat.ndim != 2 or lonlat.shape[-1] != 2:
        raise ValueError(f"{name} must be (N, 2); got shape {lonlat.shape}.")


def _validate_range(bounds: tuple[float, float], *, name: str) -> None:
    lower, upper = bounds
    if upper <= lower:
        raise ValueError(f"{name} must satisfy min < max; got {bounds}.")


def _validate_input_unit(input_unit: Literal["degrees", "radians"]) -> None:
    if input_unit not in {"degrees", "radians"}:
        raise ValueError(
            f"input_unit must be 'degrees' or 'radians'; got {input_unit!r}."
        )


def _promote_to_floating(x: Num[Array, ...]) -> Float[Array, ...]:
    """Promote integer arrays to ``float32`` so affine ops don't truncate."""
    if jnp.issubdtype(x.dtype, jnp.integer):
        return x.astype(jnp.float32)
    return x


def deg2rad(x: Float[Array, ...]) -> Float[Array, ...]:
    r"""Convert degrees to radians element-wise.

    Computes ``x · π / 180`` and preserves the input shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import deg2rad
        >>> deg2rad(jnp.array([0.0, 90.0, 180.0])).shape
        (3,)
        >>> deg2rad(jnp.array([[45.0, -90.0], [270.0, 360.0]])).shape
        (2, 2)
    """
    return x * (jnp.pi / 180.0)  # deg → rad: scale by π/180, shape preserved


def lonlat_scale(
    lonlat: Num[Array, "N 2"],
    *,
    lon_range: tuple[float, float] = (-180.0, 180.0),
    lat_range: tuple[float, float] = (-90.0, 90.0),
) -> Float[Array, "N 2"]:
    """Affine-rescale lon/lat columns.

    Values inside the given ranges map into ``[-1, 1]``; out-of-range
    values are not clipped and map outside ``[-1, 1]`` linearly. The
    default ranges assume ``lonlat`` is in degrees; pass matching
    ``lon_range`` / ``lat_range`` in whatever unit you use.

    Integer inputs are promoted to ``float32`` before the affine step
    so ``(lonlat - lower) / (upper - lower)`` is not computed in
    integer arithmetic (which would silently round the output to
    ``-1 / 0 / 1``).

    Args:
        lonlat: Longitude/latitude matrix of shape ``(N, 2)``.
        lon_range: ``(min, max)`` longitude domain (must satisfy
            ``min < max``).
        lat_range: ``(min, max)`` latitude domain (must satisfy
            ``min < max``).

    Returns:
        Rescaled lon/lat array of shape ``(N, 2)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import lonlat_scale
        >>> lonlat = jnp.array([[-180.0, -90.0], [0.0, 0.0], [180.0, 90.0]])
        >>> lonlat_scale(lonlat).shape
        (3, 2)
        >>> # Midpoint of each range maps exactly to 0.
        >>> bool((lonlat_scale(jnp.array([[0.0, 0.0]]))[0] == 0.0).all())
        True
        >>> # Custom domain (e.g. a regional grid in degrees).
        >>> lonlat_scale(
        ...     jnp.array([[0.0, 50.0]]),
        ...     lon_range=(-10.0, 10.0),
        ...     lat_range=(40.0, 60.0),
        ... ).shape
        (1, 2)
    """
    _validate_lonlat_shape(lonlat)
    _validate_range(lon_range, name="lon_range")
    _validate_range(lat_range, name="lat_range")

    lonlat = _promote_to_floating(lonlat)
    lower = jnp.asarray([lon_range[0], lat_range[0]], dtype=lonlat.dtype)  # (2,)
    upper = jnp.asarray([lon_range[1], lat_range[1]], dtype=lonlat.dtype)  # (2,)
    # Affine map: 2·(x − lo)/(hi − lo) − 1, so [lo, hi] → [-1, 1]. (N,2) -> (N,2)
    return 2.0 * (lonlat - lower) / (upper - lower) - 1.0


def cyclic_encode(
    angles: Float[Array, " N"] | Float[Array, "N D"],
) -> Float[Array, "N F"]:
    """Encode periodic inputs as concatenated cos/sin features.

    Args:
        angles: Angle vector ``(N,)`` or matrix ``(N, D)`` in radians.

    Returns:
        ``(N, 2)`` for vector input or ``(N, 2 * D)`` for matrix input,
        laid out as ``[cos_0, ..., cos_{D-1}, sin_0, ..., sin_{D-1}]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import cyclic_encode
        >>> # Vector input (N,) → (N, 2) of [cos, sin].
        >>> cyclic_encode(jnp.array([0.0, jnp.pi])).shape
        (2, 2)
        >>> # Multi-dimensional input: each column is encoded independently.
        >>> cyclic_encode(jnp.zeros((3, 2))).shape
        (3, 4)
    """
    if angles.ndim == 1:
        promoted = einx.id("n -> n 1", angles)  # (N,) -> (N, 1)
    elif angles.ndim == 2:
        promoted = angles  # (N, D)
    else:
        raise ValueError(f"angles must be (N,) or (N, D); got shape {angles.shape}.")
    # [cos θ, sin θ] embeds each angle on the unit circle. (N, D) -> (N, 2·D)
    return jnp.concatenate([jnp.cos(promoted), jnp.sin(promoted)], axis=-1)


def spherical_harmonic_encode(
    lonlat: Float[Array, "N 2"],
    l_max: int,
    *,
    input_unit: Literal["degrees", "radians"] = "radians",
) -> Float[Array, "N M"]:
    """Lift lon/lat to $S^2$ and evaluate real spherical harmonics.

    Args:
        lonlat: Longitude/latitude matrix of shape ``(N, 2)``.
        l_max: Maximum harmonic degree.
        input_unit: Whether ``lonlat`` is in ``"degrees"`` or
            ``"radians"``.

    Returns:
        Real spherical-harmonic features of shape ``(N, (l_max + 1)^2)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import spherical_harmonic_encode
        >>> lonlat = jnp.array([[0.0, 0.0], [1.5707964, 0.0]])
        >>> # (N, 2) -> (N, (l_max + 1)^2); here (l_max + 1)^2 = 16.
        >>> spherical_harmonic_encode(lonlat, l_max=3).shape
        (2, 16)
        >>> # l_max=0 keeps only the constant Y_0^0 mode.
        >>> spherical_harmonic_encode(lonlat, l_max=0).shape
        (2, 1)
    """
    _validate_lonlat_shape(lonlat)
    _validate_input_unit(input_unit)
    # (N, 2) -> (N, 3) unit sphere, then evaluate Y_l^m up to l_max.
    unit_xyz = lonlat_to_unit(_promote_to_floating(lonlat), unit=input_unit)
    return real_spherical_harmonics(unit_xyz, l_max=l_max)  # (N, (l_max+1)^2)


#: names that moved to functax, served with a DeprecationWarning until removal
_MOVED: dict[str, Moved] = {
    "cartesian_to_tangent": Moved(
        "functax.geometry.cartesian_to_tangent",
        _compat.cartesian_to_tangent,
        "Note that it takes unit= in place of input_unit=, with no default.",
    ),
    "ecef_to_lonlat": Moved(
        "functax.geometry.ecef_to_lonlat",
        _compat.ecef_to_lonlat,
        "Note that it takes unit= and semi_major= / flattening= instead.",
    ),
    "enu_basis": Moved(
        "functax.geometry.enu_basis",
        _compat.enu_basis,
        "Note that it takes unit= in place of input_unit=, with no default.",
    ),
    "lonlat_to_cartesian3d": Moved(
        "functax.geometry.lonlat_to_unit",
        _compat.lonlat_to_cartesian3d,
        "Note that it takes unit= in place of input_unit=.",
    ),
    "local_tangent_plane": Moved(
        "functax.geometry.local_tangent_plane",
        _compat.local_tangent_plane,
        "Note that it takes unit= in place of input_unit=, and radius= has no default.",
    ),
    "lonlat_to_ecef": Moved(
        "functax.geometry.lonlat_to_ecef",
        _compat.lonlat_to_ecef,
        "Note that it takes unit= and semi_major= / flattening= instead.",
    ),
    "rotation_to_pole": Moved(
        "functax.geometry.rotation_to_pole",
        _compat.rotation_to_pole,
        "Note that it takes unit= in place of input_unit=, with no default.",
    ),
    "tangent_to_cartesian": Moved(
        "functax.geometry.tangent_to_cartesian",
        _compat.tangent_to_cartesian,
        "Note that it takes unit= in place of input_unit=, with no default.",
    ),
}

__all__ = [
    "EARTH_MEAN_RADIUS_M",
    "WGS84_A",
    "WGS84_F",
    "cyclic_encode",
    "deg2rad",
    "lonlat_scale",
    "spherical_harmonic_encode",
]

__getattr__ = deprecated_getattr(__name__, _MOVED)
__dir__ = deprecated_dir(__all__, _MOVED)
