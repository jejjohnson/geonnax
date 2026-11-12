"""Old-signature shims for the moved functions whose signature functax changed.

functax took most of geonnax's maths unchanged, so the deprecated geonnax paths
return the functax object itself. These functions are the exceptions: functax
requires an explicit angle ``unit=`` (geonnax defaulted to degrees for lon/lat
inputs and radians for grid outputs), takes ``semi_major`` and ``flattening``
instead of an ellipsoid name, requires the tangent-plane ``radius``, and returns
a `functax.quad.Rule` from ``gauss_legendre``. Each shim keeps geonnax's
signature and defaults and calls functax, so the old paths keep their results.
They are deleted with the aliases (`geonnax._deprecation.REMOVED_IN`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import functax.diffops as _fd
import functax.geometry as _fg
import functax.quad as _fq
import numpy as np
from functax.basis import slepian as _fs
from jaxtyping import Array, Bool, Float

from geonnax._earth import EARTH_MEAN_RADIUS_M, WGS84_A, WGS84_F


Unit = Literal["degrees", "radians"]
Output = Literal["xyz", "lonlat"]
Ellipsoid = Literal["wgs84", "sphere"]


def _ellipsoid(ellipsoid: Ellipsoid, radius: float) -> tuple[float, float]:
    """``(semi_major, flattening)`` for geonnax's ellipsoid names."""
    if ellipsoid == "wgs84":
        return WGS84_A, WGS84_F
    if ellipsoid == "sphere":
        return radius, 0.0
    msg = f"ellipsoid must be 'wgs84' or 'sphere'; got {ellipsoid!r}."
    raise ValueError(msg)


def _grid_unit(output: Output, output_unit: Unit) -> str | None:
    """functax's ``unit``: only lon/lat output carries one."""
    return output_unit if output == "lonlat" else None


def fibonacci_sphere(
    n: int, *, output: Output = "xyz", output_unit: Unit = "radians"
) -> tuple[Float[Array, "n D"], Float[Array, " n"]]:
    """`functax.geometry.fibonacci_sphere` with geonnax's ``output_unit``."""
    return _fg.fibonacci_sphere(n, output=output, unit=_grid_unit(output, output_unit))


def gauss_legendre_grid(
    n_lat: int,
    n_lon: int | None = None,
    *,
    output: Output = "xyz",
    output_unit: Unit = "radians",
) -> tuple[Float[Array, "P D"], Float[Array, " P"]]:
    """`functax.geometry.gauss_legendre_grid` with geonnax's ``output_unit``."""
    return _fg.gauss_legendre_grid(
        n_lat, n_lon, output=output, unit=_grid_unit(output, output_unit)
    )


def latlon_grid(
    n_lat: int, n_lon: int, *, output: Output = "xyz", output_unit: Unit = "radians"
) -> tuple[Float[Array, "P D"], Float[Array, " P"]]:
    """`functax.geometry.latlon_grid` with geonnax's ``output_unit``."""
    return _fg.latlon_grid(
        n_lat, n_lon, output=output, unit=_grid_unit(output, output_unit)
    )


def healpix_grid(
    nside: int,
    *,
    nest: bool = True,
    output: Output = "xyz",
    output_unit: Unit = "radians",
) -> tuple[Float[Array, "N D"], Float[Array, " N"]]:
    """`functax.geometry.healpix_grid` with geonnax's ``output_unit``."""
    return _fg.healpix_grid(
        nside, nest=nest, output=output, unit=_grid_unit(output, output_unit)
    )


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`functax.quad.gauss_legendre` as geonnax's ``(nodes, weights)`` tuple."""
    nodes, weights = _fq.gauss_legendre(n)
    return np.asarray(nodes), np.asarray(weights)


def spherical_polygon_mask(
    points_xyz: Float[Array, "N 3"],
    polygon_lonlat: Float[Array, "P 2"],
    *,
    input_unit: Unit = "degrees",
) -> Bool[Array, " N"]:
    """`functax.basis.slepian.spherical_polygon_mask` with ``input_unit``."""
    return _fs.spherical_polygon_mask(points_xyz, polygon_lonlat, unit=input_unit)


def lonlat_to_ecef(
    lonlat: Float[Array, "N 2"],
    altitude: float | Float[Array, " N"] = 0.0,
    *,
    ellipsoid: Ellipsoid = "wgs84",
    radius: float = EARTH_MEAN_RADIUS_M,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 3"]:
    """`functax.geometry.lonlat_to_ecef` with geonnax's ellipsoid names."""
    semi_major, flattening = _ellipsoid(ellipsoid, radius)
    return _fg.lonlat_to_ecef(
        lonlat, altitude, semi_major=semi_major, flattening=flattening, unit=input_unit
    )


def ecef_to_lonlat(
    xyz: Float[Array, "N 3"],
    *,
    ellipsoid: Ellipsoid = "wgs84",
    radius: float = EARTH_MEAN_RADIUS_M,
    output_unit: Unit = "degrees",
) -> tuple[Float[Array, "N 2"], Float[Array, " N"]]:
    """`functax.geometry.ecef_to_lonlat` with geonnax's ellipsoid names."""
    semi_major, flattening = _ellipsoid(ellipsoid, radius)
    return _fg.ecef_to_lonlat(
        xyz, semi_major=semi_major, flattening=flattening, unit=output_unit
    )


def enu_basis(
    lonlat: Float[Array, "N 2"], *, input_unit: Unit = "degrees"
) -> Float[Array, "N 3 3"]:
    """`functax.geometry.enu_basis` with geonnax's ``input_unit``."""
    return _fg.enu_basis(lonlat, unit=input_unit)


def lonlat_to_cartesian3d(
    lonlat: Float[Array, "N 2"], *, input_unit: Unit = "radians"
) -> Float[Array, "N 3"]:
    """`functax.geometry.lonlat_to_unit` with ``input_unit``, promoting integers.

    functax takes any leading batch dims and raises on integer input; the shim
    keeps geonnax's ``(N, 2)`` and ``input_unit`` checks and promotes integers
    to float32 first.
    """
    # imported here: geonnax.geo imports this module
    from geonnax.geo import (
        _promote_to_floating,
        _validate_input_unit,
        _validate_lonlat_shape,
    )

    _validate_lonlat_shape(lonlat)
    _validate_input_unit(input_unit)
    return _fg.lonlat_to_unit(_promote_to_floating(lonlat), unit=input_unit)


def tangent_to_cartesian(
    lonlat: Float[Array, "N 2"],
    uv: Float[Array, "N 2"],
    *,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 3"]:
    """`functax.geometry.tangent_to_cartesian` with geonnax's ``input_unit``."""
    return _fg.tangent_to_cartesian(lonlat, uv, unit=input_unit)


def cartesian_to_tangent(
    lonlat: Float[Array, "N 2"],
    vec: Float[Array, "N 3"],
    *,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 2"]:
    """`functax.geometry.cartesian_to_tangent` with geonnax's ``input_unit``."""
    return _fg.cartesian_to_tangent(lonlat, vec, unit=input_unit)


def local_tangent_plane(
    lonlat: Float[Array, "N 2"],
    origin: Float[Array, " 2"],
    *,
    projection: Literal["azimuthal_equidistant", "gnomonic"] = "azimuthal_equidistant",
    radius: float = 6371.0088,
    input_unit: Unit = "degrees",
) -> Float[Array, "N 2"]:
    """`functax.geometry.local_tangent_plane` with geonnax's km Earth radius."""
    return _fg.local_tangent_plane(
        lonlat, origin, radius=radius, projection=projection, unit=input_unit
    )


def rotation_to_pole(
    lonlat_centre: Float[Array, " 2"], *, input_unit: Unit = "degrees"
) -> Float[Array, "3 3"]:
    """`functax.geometry.rotation_to_pole` with geonnax's ``input_unit``."""
    return _fg.rotation_to_pole(lonlat_centre, unit=input_unit)


def surface_grad_enu(
    f: Callable[[Float[Array, " 2"]], Any], *, input_unit: Unit = "degrees"
) -> Callable[[Float[Array, " 2"]], Float[Array, " 2"]]:
    """`functax.diffops.surface_grad_enu` with geonnax's ``input_unit``."""
    return _fd.surface_grad_enu(f, unit=input_unit)
