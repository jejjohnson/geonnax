r"""Point sets and quadrature grids on the unit sphere.

Each function returns ``(points, weights)``. The weights are quadrature
weights for $\int_{S^2} f\,d\Omega$, summing to $4\pi$:

- `fibonacci_sphere` — the near-uniform Fibonacci lattice, equal weights.
- `gauss_legendre_grid` — Gauss–Legendre in $\cos\theta$, uniform in
  longitude; exact for products of spherical harmonics up to a stated degree.
- `latlon_grid` — the regular cell-centred lat/lon grid with exact cell-area
  weights.

Points are unit Cartesian ``(N, 3)`` (``output="xyz"``) or ``(lon, lat)``
(``output="lonlat"``, longitude first, in ``output_unit``). Grids are ordered
latitude-major.
"""

from __future__ import annotations

import math
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from geonnax._basis._legendre import gauss_legendre


Output = Literal["lonlat", "xyz"]
Unit = Literal["degrees", "radians"]


def _format(
    lon: np.ndarray, lat: np.ndarray, output: Output, output_unit: Unit
) -> Float[Array, "N D"]:
    if output == "xyz":
        cos_lat = np.cos(lat)
        xyz = np.stack(
            [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1
        )
        return jnp.asarray(xyz)
    if output == "lonlat":
        lonlat = np.stack([lon, lat], axis=-1)
        if output_unit == "degrees":
            lonlat = np.degrees(lonlat)
        elif output_unit != "radians":
            raise ValueError(
                f"output_unit must be 'degrees' or 'radians'; got {output_unit!r}."
            )
        return jnp.asarray(lonlat)
    raise ValueError(f"output must be 'lonlat' or 'xyz'; got {output!r}.")


def fibonacci_sphere(
    n: int, *, output: Output = "xyz", output_unit: Unit = "radians"
) -> tuple[Float[Array, "n D"], Float[Array, " n"]]:
    r"""The Fibonacci lattice: ``n`` near-uniform points on the unit sphere.

    $z_i = 1 - 2(i + \tfrac12)/n$, $\phi_i = \arcsin z_i$ and
    $\lambda_i = 2\pi i/\varphi \bmod 2\pi$, with $\varphi$ the golden ratio,
    then wrapped to $[-\pi, \pi)$. Each point carries the weight $4\pi/n$.

    Args:
        n: Number of points, ``>= 1``.
        output: ``"xyz"`` for unit vectors or ``"lonlat"`` for ``(lon, lat)``.
        output_unit: Angle unit for ``output="lonlat"``.

    Returns:
        ``(points, weights)`` of shapes ``(n, 3 | 2)`` and ``(n,)``.

    Examples:
        >>> from geonnax.basis import fibonacci_sphere
        >>> points, weights = fibonacci_sphere(100)
        >>> points.shape, round(float(weights.sum()), 4)  # weights sum to 4π
        ((100, 3), 12.5664)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    i = np.arange(n, dtype=float)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    lat = np.arcsin(1.0 - 2.0 * (i + 0.5) / n)
    lon = np.mod(2.0 * math.pi * i / golden + math.pi, 2.0 * math.pi) - math.pi
    weights = jnp.full((n,), 4.0 * math.pi / n)
    return _format(lon, lat, output, output_unit), weights


def gauss_legendre_grid(
    n_lat: int,
    n_lon: int | None = None,
    *,
    output: Output = "xyz",
    output_unit: Unit = "radians",
) -> tuple[Float[Array, "P D"], Float[Array, " P"]]:
    r"""Gauss–Legendre grid on the sphere with its quadrature weights.

    The sines of latitude are the ``n_lat`` Gauss–Legendre nodes on $[-1, 1]$
    and the longitudes are uniform, $\lambda_j = -\pi + 2\pi j/n_{lon}$. The
    weight of node $(k, j)$ is $w_k \cdot 2\pi/n_{lon}$.

    The rule integrates $Y_{lm} Y_{l'm'}$ exactly when $l + l' \le
    2n_{lat} - 1$ and $n_{lon} > l + l'$; more generally, any band-limited
    function of degree $\le \min(2n_{lat} - 1,\ n_{lon} - 1)$.

    Args:
        n_lat: Number of latitudes, ``>= 1``.
        n_lon: Number of longitudes; defaults to ``2 * n_lat``.
        output: ``"xyz"`` for unit vectors or ``"lonlat"`` for ``(lon, lat)``.
        output_unit: Angle unit for ``output="lonlat"``.

    Returns:
        ``(points, weights)`` of shapes ``(n_lat * n_lon, 3 | 2)`` and
        ``(n_lat * n_lon,)``, latitude-major.

    Examples:
        >>> from geonnax.basis import gauss_legendre_grid
        >>> points, weights = gauss_legendre_grid(8)  # exact to degree 15
        >>> points.shape, round(float(weights.sum()), 4)
        ((128, 3), 12.5664)
    """
    if n_lat < 1:
        raise ValueError(f"n_lat must be >= 1, got {n_lat}.")
    n_lon = 2 * n_lat if n_lon is None else n_lon
    if n_lon < 1:
        raise ValueError(f"n_lon must be >= 1, got {n_lon}.")
    nodes, w = gauss_legendre(n_lat)
    lat_1d = np.arcsin(nodes)
    lon_1d = -math.pi + 2.0 * math.pi * np.arange(n_lon) / n_lon
    lat, lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    weights = np.repeat(w * 2.0 * math.pi / n_lon, n_lon)
    return _format(lon.ravel(), lat.ravel(), output, output_unit), jnp.asarray(weights)


def latlon_grid(
    n_lat: int,
    n_lon: int,
    *,
    output: Output = "xyz",
    output_unit: Unit = "radians",
) -> tuple[Float[Array, "P D"], Float[Array, " P"]]:
    r"""Regular cell-centred lat/lon grid with exact cell-area weights.

    Cells split latitude into ``n_lat`` and longitude into ``n_lon`` equal
    intervals. Each point is a cell centre, and its weight is the cell's area,
    $\Delta\lambda\,(\sin\phi_{\text{top}} - \sin\phi_{\text{bottom}})$.

    Args:
        n_lat: Number of latitude bands, ``>= 1``.
        n_lon: Number of longitude bands, ``>= 1``.
        output: ``"xyz"`` for unit vectors or ``"lonlat"`` for ``(lon, lat)``.
        output_unit: Angle unit for ``output="lonlat"``.

    Returns:
        ``(points, weights)`` of shapes ``(n_lat * n_lon, 3 | 2)`` and
        ``(n_lat * n_lon,)``, latitude-major.

    Examples:
        >>> from geonnax.basis import latlon_grid
        >>> pts, w = latlon_grid(18, 36, output="lonlat", output_unit="degrees")
        >>> pts[0].tolist(), round(float(w.sum()), 4)  # first cell centre
        ([-175.0, -85.0], 12.5664)
    """
    if n_lat < 1 or n_lon < 1:
        raise ValueError(f"n_lat and n_lon must be >= 1; got {n_lat}, {n_lon}.")
    lat_edges = np.linspace(-math.pi / 2, math.pi / 2, n_lat + 1)
    d_lon = 2.0 * math.pi / n_lon
    lat_1d = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_1d = -math.pi + d_lon * (np.arange(n_lon) + 0.5)
    band_area = d_lon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    lat, lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    weights = np.repeat(band_area, n_lon)
    return _format(lon.ravel(), lat.ravel(), output, output_unit), jnp.asarray(weights)
