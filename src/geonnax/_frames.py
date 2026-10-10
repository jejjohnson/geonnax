r"""Coordinate frames on the Earth: ellipsoid, local tangent frames and rotations.

Pure-JAX helpers re-exported from `geonnax.geo`:

- `lonlat_to_ecef` / `ecef_to_lonlat` — geodetic ↔ Earth-centred,
  Earth-fixed coordinates on the WGS84 ellipsoid (or a sphere).
- `enu_basis` — the local east/north/up unit vectors.
- `tangent_to_cartesian` / `cartesian_to_tangent` — east/north vector
  components ↔ 3-D tangent vectors.
- `local_tangent_plane` — azimuthal-equidistant or gnomonic projection onto
  the tangent plane at an origin.
- `rotation_to_pole` — the rotation that takes a given centre to the north
  pole.

Geographic inputs are ``(N, 2)`` arrays of ``(lon, lat)``, longitude first.
This module imports nothing from the rest of geonnax, so the basis package can
use it without an import cycle.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float


WGS84_A = 6378137.0
"""WGS84 semi-major axis, in metres."""
WGS84_F = 1.0 / 298.257223563
"""WGS84 flattening."""
EARTH_MEAN_RADIUS_M = 6371008.8
"""IUGG mean Earth radius, in metres."""

Unit = Literal["degrees", "radians"]


def _to_radians(lonlat: Float[Array, "... 2"], unit: Unit) -> Float[Array, "... 2"]:
    lonlat = jnp.asarray(lonlat)
    if not jnp.issubdtype(lonlat.dtype, jnp.floating):
        lonlat = lonlat.astype(jnp.float32)
    if unit == "degrees":
        return jnp.deg2rad(lonlat)
    if unit == "radians":
        return lonlat
    raise ValueError(f"unit must be 'degrees' or 'radians'; got {unit!r}.")


def _check_lonlat(lonlat: Float[Array, "..."], name: str = "lonlat") -> None:
    if lonlat.ndim != 2 or lonlat.shape[-1] != 2:
        raise ValueError(f"{name} must be (N, 2); got shape {lonlat.shape}.")


def _enu(lon: Float[Array, " N"], lat: Float[Array, " N"]) -> Float[Array, "N 3 3"]:
    sl, cl = jnp.sin(lon), jnp.cos(lon)
    sp, cp = jnp.sin(lat), jnp.cos(lat)
    east = jnp.stack([-sl, cl, jnp.zeros_like(lon)], axis=-1)
    north = jnp.stack([-sp * cl, -sp * sl, cp], axis=-1)
    up = jnp.stack([cp * cl, cp * sl, sp], axis=-1)
    return jnp.stack([east, north, up], axis=-2)


def lonlat_to_ecef(
    lonlat: Float[Array, "N 2"],
    altitude: float | Float[Array, " N"] = 0.0,
    *,
    ellipsoid: Literal["wgs84", "sphere"] = "wgs84",
    radius: float = EARTH_MEAN_RADIUS_M,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 3"]:
    r"""Geodetic ``(lon, lat)`` plus altitude to Earth-centred, Earth-fixed metres.

    On the WGS84 ellipsoid ($a = 6378137$ m, $f = 1/298.257223563$,
    $e^2 = f(2 - f)$, $N(\phi) = a/\sqrt{1 - e^2\sin^2\phi}$):

    $$
    X = (N + h)\cos\phi\cos\lambda,\quad
    Y = (N + h)\cos\phi\sin\lambda,\quad
    Z = \big(N(1 - e^2) + h\big)\sin\phi.
    $$

    With ``ellipsoid="sphere"`` this is $(R + h)$ times the unit vector, with
    $R$ = ``radius``.

    Args:
        lonlat: ``(N, 2)`` longitude/latitude.
        altitude: Height above the ellipsoid (or sphere) in metres, scalar or
            ``(N,)``.
        ellipsoid: ``"wgs84"`` or ``"sphere"``.
        radius: Sphere radius in metres, used with ``ellipsoid="sphere"``.
        input_unit: Unit of ``lonlat``.

    Returns:
        ``(N, 3)`` ECEF coordinates in metres.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import lonlat_to_ecef
        >>> lonlat_to_ecef(jnp.array([[0.0, 0.0]])).tolist()  # (a, 0, 0)
        [[6378137.0, 0.0, 0.0]]
    """
    _check_lonlat(jnp.asarray(lonlat))
    rad = _to_radians(lonlat, input_unit)
    lon, lat = rad[:, 0], rad[:, 1]
    h = jnp.asarray(altitude, dtype=rad.dtype)
    if ellipsoid == "sphere":
        n_rad = jnp.full_like(lat, radius)
        e2 = 0.0
    elif ellipsoid == "wgs84":
        e2 = WGS84_F * (2.0 - WGS84_F)
        n_rad = WGS84_A / jnp.sqrt(1.0 - e2 * jnp.sin(lat) ** 2)
    else:
        raise ValueError(f"ellipsoid must be 'wgs84' or 'sphere'; got {ellipsoid!r}.")
    cos_lat = jnp.cos(lat)
    return jnp.stack(
        [
            (n_rad + h) * cos_lat * jnp.cos(lon),
            (n_rad + h) * cos_lat * jnp.sin(lon),
            (n_rad * (1.0 - e2) + h) * jnp.sin(lat),
        ],
        axis=-1,
    )


def ecef_to_lonlat(
    xyz: Float[Array, "N 3"],
    *,
    ellipsoid: Literal["wgs84", "sphere"] = "wgs84",
    radius: float = EARTH_MEAN_RADIUS_M,
    output_unit: Unit = "degrees",
) -> tuple[Float[Array, "N 2"], Float[Array, " N"]]:
    r"""Earth-centred, Earth-fixed metres to geodetic ``(lon, lat)`` and altitude.

    The inverse of `lonlat_to_ecef`. On WGS84 it uses Bowring's (1976)
    closed-form latitude followed by two fixed refinement steps of
    $\phi \leftarrow \operatorname{atan2}\!\big(Z,\ p\,(1 - e^2 N/(N + h))\big)$,
    accurate to well under a millimetre for altitudes within ±10 km. Altitude
    is $h = p\cos\phi + Z\sin\phi - a\sqrt{1 - e^2\sin^2\phi}$, which stays
    stable at the poles.

    Args:
        xyz: ``(N, 3)`` ECEF coordinates in metres.
        ellipsoid: ``"wgs84"`` or ``"sphere"``.
        radius: Sphere radius in metres, used with ``ellipsoid="sphere"``.
        output_unit: Unit of the returned ``lonlat``.

    Returns:
        ``(lonlat, altitude)`` of shapes ``(N, 2)`` and ``(N,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import ecef_to_lonlat, lonlat_to_ecef
        >>> lonlat, h = ecef_to_lonlat(lonlat_to_ecef(jnp.array([[10.0, 45.0]]), 100.0))
        >>> [round(float(v), 3) for v in lonlat[0]]
        [10.0, 45.0]
        >>> abs(float(h[0]) - 100.0) < 1.0  # float32 resolves ECEF to ~1 m; use x64
        True
    """
    xyz = jnp.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (N, 3); got shape {xyz.shape}.")
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = jnp.arctan2(y, x)
    p = jnp.hypot(x, y)
    if ellipsoid == "sphere":
        lat = jnp.arctan2(z, p)
        h = jnp.sqrt(p**2 + z**2) - radius
    elif ellipsoid == "wgs84":
        a = WGS84_A
        e2 = WGS84_F * (2.0 - WGS84_F)
        b = a * (1.0 - WGS84_F)
        ep2 = e2 / (1.0 - e2)
        theta = jnp.arctan2(z * a, p * b)
        lat = jnp.arctan2(
            z + ep2 * b * jnp.sin(theta) ** 3, p - e2 * a * jnp.cos(theta) ** 3
        )
        for _ in range(2):  # fixed refinement steps (jit-safe)
            n_rad = a / jnp.sqrt(1.0 - e2 * jnp.sin(lat) ** 2)
            h = (
                p * jnp.cos(lat)
                + z * jnp.sin(lat)
                - a * jnp.sqrt(1.0 - e2 * jnp.sin(lat) ** 2)
            )
            lat = jnp.arctan2(z, p * (1.0 - e2 * n_rad / (n_rad + h)))
        h = (
            p * jnp.cos(lat)
            + z * jnp.sin(lat)
            - a * jnp.sqrt(1.0 - e2 * jnp.sin(lat) ** 2)
        )
    else:
        raise ValueError(f"ellipsoid must be 'wgs84' or 'sphere'; got {ellipsoid!r}.")
    lonlat = jnp.stack([lon, lat], axis=-1)
    if output_unit == "degrees":
        lonlat = jnp.rad2deg(lonlat)
    elif output_unit != "radians":
        raise ValueError(
            f"output_unit must be 'degrees' or 'radians'; got {output_unit!r}."
        )
    return lonlat, h


def enu_basis(
    lonlat: Float[Array, "N 2"], *, input_unit: Unit = "degrees"
) -> Float[Array, "N 3 3"]:
    r"""Local east/north/up unit vectors at each point.

    At longitude $\lambda$ and latitude $\phi$:

    $$
    \hat e = (-\sin\lambda, \cos\lambda, 0),\quad
    \hat n = (-\sin\phi\cos\lambda, -\sin\phi\sin\lambda, \cos\phi),\quad
    \hat u = (\cos\phi\cos\lambda, \cos\phi\sin\lambda, \sin\phi).
    $$

    The frame is orthonormal and right-handed ($\hat e\times\hat n = \hat u$).
    At exactly $\pm 90°$ east and north are not unique; the formula still
    applies, so the given longitude picks the frame.

    Args:
        lonlat: ``(N, 2)`` longitude/latitude.
        input_unit: Unit of ``lonlat``.

    Returns:
        ``(N, 3, 3)`` with rows east, north, up.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import enu_basis
        >>> (enu_basis(jnp.array([[0.0, 0.0]]))[0].round(6) + 0.0).tolist()
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    """
    _check_lonlat(jnp.asarray(lonlat))
    rad = _to_radians(lonlat, input_unit)
    return _enu(rad[:, 0], rad[:, 1])


def tangent_to_cartesian(
    lonlat: Float[Array, "N 2"],
    uv: Float[Array, "N 2"],
    *,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 3"]:
    r"""East/north components ``(u, v)`` to 3-D tangent vectors $u\hat e + v\hat n$.

    Args:
        lonlat: ``(N, 2)`` longitude/latitude of the vectors' base points.
        uv: ``(N, 2)`` eastward and northward components.
        input_unit: Unit of ``lonlat``.

    Returns:
        ``(N, 3)`` Cartesian tangent vectors.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import tangent_to_cartesian
        >>> east = jnp.array([[1.0, 0.0]])  # u = 1 m/s eastward at (0°, 0°)
        >>> (tangent_to_cartesian(jnp.array([[0.0, 0.0]]), east) + 0.0).tolist()
        [[0.0, 1.0, 0.0]]
    """
    frame = enu_basis(lonlat, input_unit=input_unit)[:, :2, :]  # (N, 2, 3)
    return einx.dot("n c, n c d -> n d", jnp.asarray(uv), frame)


def cartesian_to_tangent(
    lonlat: Float[Array, "N 2"],
    vec: Float[Array, "N 3"],
    *,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 2"]:
    """3-D vectors to their east/north components ``(v · ê, v · n̂)``.

    The radial part is discarded, so this is the inverse of
    `tangent_to_cartesian` on tangent vectors.

    Args:
        lonlat: ``(N, 2)`` longitude/latitude of the vectors' base points.
        vec: ``(N, 3)`` Cartesian vectors.
        input_unit: Unit of ``lonlat``.

    Returns:
        ``(N, 2)`` eastward and northward components.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import cartesian_to_tangent
        >>> up_z = jnp.array([[0.0, 0.0, 2.0]])  # +z is due north at (0°, 0°)
        >>> (cartesian_to_tangent(jnp.array([[0.0, 0.0]]), up_z) + 0.0).tolist()
        [[0.0, 2.0]]
    """
    frame = enu_basis(lonlat, input_unit=input_unit)[:, :2, :]
    return einx.dot("n c d, n d -> n c", frame, jnp.asarray(vec))


def local_tangent_plane(
    lonlat: Float[Array, "N 2"],
    origin: Float[Array, " 2"],
    *,
    projection: Literal["azimuthal_equidistant", "gnomonic"] = "azimuthal_equidistant",
    radius: float = 6371.0088,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 2"]:
    r"""Project points onto the tangent plane at ``origin`` (east, north), on a sphere.

    With $\hat e_0, \hat n_0, \hat u_0$ the frame at the origin and $p$ the
    point's unit vector, let $x = p\cdot\hat e_0$, $y = p\cdot\hat n_0$,
    $z = p\cdot\hat u_0$.

    - ``"azimuthal_equidistant"``: distance and bearing from the origin are
      preserved, $(x, y)\,R\,\theta/\sin\theta$ with
      $\theta = \operatorname{atan2}(\sqrt{x^2+y^2}, z)$ the great-circle angle.
    - ``"gnomonic"``: central projection, $R\,(x, y)/z$; great circles map to
      straight lines. Only the hemisphere around the origin ($z > 0$) maps to
      finite points.

    Args:
        lonlat: ``(N, 2)`` longitude/latitude.
        origin: ``(2,)`` longitude/latitude of the tangent point.
        projection: ``"azimuthal_equidistant"`` or ``"gnomonic"``.
        radius: Sphere radius; the output is in its units (km by default).
        input_unit: Unit of ``lonlat`` and ``origin``.

    Returns:
        ``(N, 2)`` east/north plane coordinates.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import local_tangent_plane
        >>> xy = local_tangent_plane(jnp.array([[0.0, 1.0]]), jnp.array([0.0, 0.0]))
        >>> [round(float(v), 1) for v in xy[0]]  # 1° due north ≈ 111.2 km
        [0.0, 111.2]
    """
    _check_lonlat(jnp.asarray(lonlat))
    rad = _to_radians(lonlat, input_unit)
    o = _to_radians(jnp.asarray(origin)[None, :], input_unit)
    frame = _enu(o[:, 0], o[:, 1])[0]  # (3, 3) rows e, n, u
    p = _enu(rad[:, 0], rad[:, 1])[:, 2, :]  # unit vectors (N, 3)
    local = einx.dot("n d, c d -> n c", p, frame)  # (N, 3): x, y, z
    xy, z = local[:, :2], local[:, 2]
    if projection == "gnomonic":
        return radius * xy / z[:, None]
    if projection != "azimuthal_equidistant":
        raise ValueError(
            "projection must be 'azimuthal_equidistant' or 'gnomonic'; "
            f"got {projection!r}."
        )
    s2 = jnp.sum(xy**2, axis=-1)
    s = jnp.sqrt(jnp.where(s2 > 0, s2, 1.0))
    theta = jnp.arctan2(s, z)
    # θ / sin θ → 1 at the origin; the where keeps its gradient finite there.
    scale = jnp.where(s2 > 0, theta / s, 1.0)
    return radius * xy * scale[:, None]


def rotation_to_pole(
    lonlat_centre: Float[Array, " 2"], *, input_unit: Unit = "degrees"
) -> Float[Array, "3 3"]:
    r"""The rotation $R$ with $R\,\hat u(\text{centre}) = (0, 0, 1)$.

    Its rows are the east, north and up vectors at the centre, so
    $R\,p = (p\cdot\hat e, p\cdot\hat n, p\cdot\hat u)$: the centre goes to
    the north pole and its local north to the $+y$ axis. This is the frame the
    Slepian cap bases are evaluated in.

    Args:
        lonlat_centre: ``(2,)`` longitude/latitude of the centre.
        input_unit: Unit of ``lonlat_centre``.

    Returns:
        ``(3, 3)`` rotation matrix (orthogonal, determinant 1).

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.geo import rotation_to_pole
        >>> R = rotation_to_pole(jnp.array([0.0, 0.0]))
        >>> (R @ jnp.array([1.0, 0.0, 0.0]) + 0.0).round(6).tolist()  # centre → pole
        [0.0, 0.0, 1.0]
    """
    centre = _to_radians(jnp.asarray(lonlat_centre), input_unit)
    if centre.shape != (2,):
        raise ValueError(f"lonlat_centre must have shape (2,), got {centre.shape}.")
    return _enu(centre[None, 0], centre[None, 1])[0]
