r"""Point sets and quadrature grids on the unit sphere.

Each function returns ``(points, weights)``. The weights are quadrature
weights for $\int_{S^2} f\,d\Omega$, summing to $4\pi$:

- `fibonacci_sphere` — the near-uniform Fibonacci lattice, equal weights.
- `gauss_legendre_grid` — Gauss–Legendre in $\cos\theta$, uniform in
  longitude; exact for products of spherical harmonics up to a stated degree.
- `latlon_grid` — the regular cell-centred lat/lon grid with exact cell-area
  weights.
- `healpix_grid` — HEALPix pixel centres (ring or nested order), equal-area
  weights.

Points are unit Cartesian ``(N, 3)`` (``output="xyz"``) or ``(lon, lat)``
(``output="lonlat"``, longitude first, in ``output_unit``). The lat/lon grids
are ordered latitude-major; HEALPix follows its ring or nested pixel index.
"""

from __future__ import annotations

import math
import operator
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


# Base-face layout of HEALPix (Górski et al. 2005): for face f, the ring of its
# southernmost-corner pixel is _JRLL[f]·nside - 1 and its longitude offset is
# _JPLL[f]·π/4.
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def _interleave_bits(ix: np.ndarray, iy: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.zeros_like(ix)
    for b in range(n_bits):
        out |= ((ix >> b) & 1) << (2 * b)
        out |= ((iy >> b) & 1) << (2 * b + 1)
    return out


def healpix_grid(
    nside: int,
    *,
    nest: bool = True,
    output: Output = "xyz",
    output_unit: Unit = "radians",
) -> tuple[Float[Array, "N D"], Float[Array, " N"]]:
    r"""HEALPix pixel centres with their equal-area weights.

    HEALPix (Górski et al., 2005) splits the sphere into 12 base faces, each
    into $N_{side}^2$ pixels of equal area, so $N = 12 N_{side}^2$ and every
    weight is $4\pi/N$. Pixel centres lie on $4N_{side} - 1$ iso-latitude
    rings: in the polar caps ($|z| > 2/3$, $z = \cos\theta$) ring $i$ has
    $z = \pm(1 - i^2/3N_{side}^2)$ and $4i$ pixels; in the equatorial belt
    $z = 4/3 - 2i/3N_{side}$ with $4N_{side}$ pixels per ring, alternate
    rings shifted by half a pixel. The centres are computed in closed form
    from each pixel's face and face-local coordinates $(x, y)$.

    ``nest=False`` orders pixels by ring, north to south and east from
    $\lambda = 0$ within a ring. ``nest=True`` uses the nested order, face by
    face with the bits of $(x, y)$ interleaved, so pixel $p$ at $N_{side}$
    contains pixels $4p, \dots, 4p + 3$ at $2N_{side}$; it needs $N_{side}$
    to be a power of two. Both orders match ``healpy.pix2ang``. Longitudes
    are wrapped to $[-\pi, \pi)$, like the other grids here.

    Args:
        nside: Resolution parameter $N_{side}$, ``>= 1``.
        nest: Nested (``True``) or ring (``False``) pixel order.
        output: ``"xyz"`` for unit vectors or ``"lonlat"`` for ``(lon, lat)``.
        output_unit: Angle unit for ``output="lonlat"``.

    Returns:
        ``(points, weights)`` of shapes ``(12 * nside**2, 3 | 2)`` and
        ``(12 * nside**2,)``.

    Raises:
        ValueError: If ``nside < 1``, or ``nest=True`` and ``nside`` is not
            a power of two.

    References:
        Górski, K. M., Hivon, E., Banday, A. J., Wandelt, B. D., Hansen,
        F. K., Reinecke, M., Bartelmann, M. (2005). HEALPix: a framework for
        high-resolution discretization and fast analysis of data distributed
        on the sphere. *ApJ* 622, 759.

    Examples:
        >>> from geonnax.basis import healpix_grid
        >>> pts, w = healpix_grid(1, output="lonlat", output_unit="degrees")
        >>> pts.shape, round(float(w.sum()), 4)
        ((12, 2), 12.5664)
        >>> [round(v, 4) for v in pts[0].tolist()]  # first face, north
        [45.0, 41.8103]
    """
    if isinstance(nside, bool):
        raise ValueError(f"nside must be an integer >= 1, got {nside!r}.")
    try:
        nside = operator.index(nside)  # accepts NumPy integers, rejects floats
    except TypeError as err:
        raise ValueError(f"nside must be an integer >= 1, got {nside!r}.") from err
    if nside < 1:
        raise ValueError(f"nside must be >= 1, got {nside}.")
    if nest and nside & (nside - 1):
        raise ValueError(
            f"nested order needs nside to be a power of two; got {nside}. "
            "Use nest=False for the ring order."
        )
    n = nside
    npix = 12 * n * n
    ncap = 2 * n * (n - 1)
    face, ix, iy = (
        a.ravel()
        for a in np.meshgrid(np.arange(12), np.arange(n), np.arange(n), indexing="ij")
    )
    jr = _JRLL[face] * n - ix - iy - 1  # ring number, 1 .. 4n - 1
    north, south = jr < n, jr > 3 * n
    nr = np.where(north, jr, np.where(south, 4 * n - jr, n))  # pixels per ring / 4
    z = np.where(
        north,
        1.0 - nr**2 / (3.0 * n * n),
        np.where(south, nr**2 / (3.0 * n * n) - 1.0, (2 * n - jr) * 2.0 / (3.0 * n)),
    )
    kshift = np.where(north | south, 0, (jr - n) & 1)
    jp = (_JPLL[face] * nr + ix - iy + 1 + kshift) // 2  # index in ring, 1 .. 4nr
    jp = np.where(jp > 4 * n, jp - 4 * n, np.where(jp < 1, jp + 4 * n, jp))
    lon = (jp - 0.5 * (kshift + 1)) * (math.pi / (2.0 * nr))  # in [0, 2π)
    if nest:
        index = face * n * n + _interleave_bits(ix, iy, max(n.bit_length() - 1, 0))
    else:
        before = np.where(
            north,
            2 * nr * (nr - 1),
            np.where(south, npix - 2 * nr * (nr + 1), ncap + (jr - n) * 4 * n),
        )
        index = before + jp - 1
    order = np.argsort(index)
    lat = np.arcsin(z[order])
    lon = np.mod(lon[order] + math.pi, 2.0 * math.pi) - math.pi
    weights = jnp.full((npix,), 4.0 * math.pi / npix)
    return _format(lon, lat, output, output_unit), weights
