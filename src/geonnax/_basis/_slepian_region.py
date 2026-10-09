r"""Slepian bases for general regions of the sphere.

`slepian_cap_basis` handles axisymmetric caps. This module handles any region
$R$, given either as quadrature points and weights over $R$
(`slepian_region_basis`) or as the polar-gap complement of two polar caps
(`slepian_polar_gap_basis`), plus a spherical point-in-polygon test
(`spherical_polygon_mask`) for building region quadratures from polygons.

The concentration matrix is $D_{ij} = \int_R Y_i Y_j\,d\Omega$ over the real
spherical harmonics up to $l_{\max}$; its eigenvectors are the Slepian
functions and its eigenvalues $\lambda_k \in [0, 1]$ their concentration in
$R$. The Shannon number $\operatorname{tr} D = (l_{\max}+1)^2 A/4\pi$ is the
number of well-concentrated functions.
"""

from __future__ import annotations

import math
from typing import Literal

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from geonnax._basis._legendre import (
    associated_legendre,
    associated_legendre_indices,
    gauss_legendre,
)
from geonnax._basis._slepian import _check_l_max, _select_modes, _sh_index
from geonnax._basis._spherical import real_spherical_harmonics


class SlepianBasis(eqx.Module):
    r"""Slepian functions of a general region, in the real spherical-harmonic basis.

    Each retained function is $g_k = \sum_i C_{ik}\,Y_i$, evaluated in the
    global frame (no rotation, unlike `SlepianCapBasis`).

    Attributes:
        l_max: Maximum spherical-harmonic degree.
        coeffs: ``((l_max + 1)^2, K)`` mixing coefficients, columns sorted by
            decreasing concentration.
        eigenvalues: ``(K,)`` concentrations $\lambda_k \in [0, 1]$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import slepian_polar_gap_basis
        >>> basis = slepian_polar_gap_basis(4, 0.3, n_modes=5)
        >>> basis.evaluate(jnp.array([[1.0, 0.0, 0.0]])).shape  # (N, K)
        (1, 5)
    """

    l_max: int = eqx.field(static=True)
    coeffs: Float[Array, "M K"]
    eigenvalues: Float[Array, " K"]

    @property
    def num_modes(self) -> int:
        """Number of retained Slepian functions."""
        return int(self.eigenvalues.shape[0])

    def evaluate(self, unit_xyz: Float[Array, "N 3"]) -> Float[Array, "N K"]:
        """Evaluate the retained Slepian functions at unit Cartesian points."""
        harmonics = real_spherical_harmonics(unit_xyz, self.l_max)
        return einx.dot("n m, m k -> n k", harmonics, self.coeffs)


def _finish(
    l_max: int,
    matrix: Float[Array, "M M"],
    n_modes: int | None,
    eig_threshold: float | None,
) -> SlepianBasis:
    if n_modes is not None and n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
    vals, vecs = jnp.linalg.eigh(0.5 * (matrix + matrix.T))
    order = jnp.argsort(vals)[::-1]
    vals, vecs = jnp.clip(vals[order], 0.0, 1.0), vecs[:, order]
    if n_modes is None and eig_threshold is None:
        n_modes = max(1, round(float(jnp.trace(matrix))))  # Shannon number
    vals, vecs = _select_modes(vals, vecs, n_modes, eig_threshold)
    return SlepianBasis(l_max=l_max, coeffs=vecs, eigenvalues=vals)


def slepian_region_basis(
    l_max: int,
    region_points_xyz: Float[Array, "Q 3"],
    region_weights: Float[Array, " Q"],
    *,
    n_modes: int | None = None,
    eig_threshold: float | None = None,
) -> SlepianBasis:
    r"""Slepian functions of a region given by a quadrature over it.

    $D_{ij} \approx \sum_q w_q\,Y_i(x_q)\,Y_j(x_q)$ over the region's
    quadrature points. The basis is only as accurate as that quadrature: a rule
    exact to degree $2 l_{\max}$ on the region gives the exact Slepians, while
    masking a global grid (e.g. `gauss_legendre_grid` filtered by
    `spherical_polygon_mask`) has an error set by the grid spacing at the
    region's boundary.

    Args:
        l_max: Maximum spherical-harmonic degree.
        region_points_xyz: ``(Q, 3)`` unit vectors inside the region.
        region_weights: ``(Q,)`` quadrature weights (summing to the area).
        n_modes: Number of functions to keep. Defaults to the Shannon number
            $\operatorname{tr} D$ when ``eig_threshold`` is also unset.
        eig_threshold: Keep functions with concentration above this value.

    Returns:
        A `SlepianBasis`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import gauss_legendre_grid, slepian_region_basis
        >>> points, weights = gauss_legendre_grid(12)
        >>> north = points[:, 2] > 0.5  # a cap of half-angle 60°, by masking
        >>> basis = slepian_region_basis(5, points[north], weights[north])
        >>> basis.num_modes  # Shannon number 36 · (1 − cos 60°)/2 = 9
        9
    """
    _check_l_max(l_max)
    points = jnp.asarray(region_points_xyz)
    weights = jnp.asarray(region_weights)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"region_points_xyz must be (Q, 3); got {points.shape}.")
    if weights.shape != points.shape[:1]:
        raise ValueError(
            f"region_weights must be ({points.shape[0]},); got {weights.shape}."
        )
    Y = real_spherical_harmonics(points, l_max)
    weighted = einx.multiply("q i, q -> q i", Y, weights)
    matrix = einx.dot("q i, q j -> i j", weighted, Y)
    return _finish(l_max, matrix, n_modes, eig_threshold)


def slepian_polar_gap_basis(
    l_max: int,
    gap_colatitude: float,
    *,
    n_modes: int | None = None,
    eig_threshold: float | None = None,
    num_quadrature: int | None = None,
) -> SlepianBasis:
    r"""Slepian functions of the band between two polar caps (a polar gap).

    The region is $\{|z| < \cos\Theta\}$ with $\Theta$ = ``gap_colatitude``:
    the sphere minus two caps of half-angle $\Theta$ around the poles, as in
    satellite data that miss the poles. The region is zonal, so $D$ is
    block-diagonal in $m$ with blocks
    $D^{m}_{ll'} = \int_{-\cos\Theta}^{\cos\Theta}\bar P_l^m\bar P_{l'}^m\,dz$
    (orthonormal associated Legendre functions), computed exactly by
    Gauss–Legendre quadrature.

    Args:
        l_max: Maximum spherical-harmonic degree.
        gap_colatitude: Half-angle $\Theta$ of each polar cap, radians in
            $(0, \pi/2)$.
        n_modes: Number of functions to keep. Defaults to the Shannon number.
        eig_threshold: Keep functions with concentration above this value.
        num_quadrature: Gauss–Legendre order; defaults to ``l_max + 2``, exact
            for the degree-$2 l_{\max}$ integrands.

    Returns:
        A `SlepianBasis`.

    Examples:
        >>> from geonnax.basis import slepian_polar_gap_basis
        >>> basis = slepian_polar_gap_basis(6, 0.2)
        >>> basis.num_modes  # Shannon number 49 · cos(0.2) ≈ 48
        48
    """
    _check_l_max(l_max)
    if not 0.0 < gap_colatitude < 0.5 * math.pi:
        raise ValueError(
            f"gap_colatitude must be in (0, pi/2) radians, got {gap_colatitude}."
        )
    n_quad = num_quadrature or (l_max + 2)
    nodes, weights = gauss_legendre(n_quad)
    half = math.cos(gap_colatitude)
    z = jnp.asarray(half * nodes)
    w = jnp.asarray(half * weights)
    p_bar = associated_legendre(z, l_max)  # (Q, columns)
    pairs = associated_legendre_indices(l_max)
    n_harmonics = (l_max + 1) ** 2
    matrix = jnp.zeros((n_harmonics, n_harmonics), dtype=p_bar.dtype)
    for m_abs in range(l_max + 1):
        cols = [i for i, (_, mm) in enumerate(pairs) if mm == m_abs]
        block_p = p_bar[:, jnp.asarray(cols)]
        weighted = einx.multiply("q i, q -> q i", block_p, w)
        block = einx.dot("q i, q j -> i j", weighted, block_p)
        for m in {m_abs, -m_abs}:
            idx = jnp.asarray([_sh_index(ell, m) for ell in range(m_abs, l_max + 1)])
            matrix = matrix.at[jnp.ix_(idx, idx)].set(block)
    return _finish(l_max, matrix, n_modes, eig_threshold)


def spherical_polygon_mask(
    points_xyz: Float[Array, "N 3"],
    polygon_lonlat: Float[Array, "P 2"],
    *,
    input_unit: Literal["degrees", "radians"] = "degrees",
) -> Bool[Array, " N"]:
    r"""Which points lie inside a spherical polygon.

    Edges are great-circle arcs between consecutive vertices (the ring closes
    itself). A point is inside when the polygon winds once counter-clockwise
    around it: the sum over edges $(a, b)$ of the signed angle
    $\operatorname{atan2}\big((a\times b)\cdot p,\ a\cdot b - (a\cdot p)(b\cdot p)\big)$
    subtended at $p$ is $+2\pi$.

    Requirements: vertices are ordered counter-clockwise seen from outside the
    sphere (interior on the left, as in GeoJSON exterior rings), and the
    polygon's interior fits within an open hemisphere, so that a point and its
    antipode are never both inside.

    Args:
        points_xyz: ``(N, 3)`` unit vectors to classify.
        polygon_lonlat: ``(P, 2)`` vertex longitude/latitude, ``P >= 3``.
        input_unit: Unit of ``polygon_lonlat``.

    Returns:
        ``(N,)`` boolean mask.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import spherical_polygon_mask
        >>> corners = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]
        >>> square = jnp.array(corners)  # counter-clockwise, around (0°, 0°)
        >>> points = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        >>> spherical_polygon_mask(points, square).tolist()
        [True, False]
    """
    poly = jnp.asarray(polygon_lonlat)
    if poly.ndim != 2 or poly.shape[-1] != 2 or poly.shape[0] < 3:
        raise ValueError(f"polygon_lonlat must be (P >= 3, 2); got {poly.shape}.")
    if not jnp.issubdtype(poly.dtype, jnp.floating):
        poly = poly.astype(jnp.float32)
    if input_unit == "degrees":
        poly = jnp.deg2rad(poly)
    elif input_unit != "radians":
        raise ValueError(
            f"input_unit must be 'degrees' or 'radians'; got {input_unit!r}."
        )
    cos_lat = jnp.cos(poly[:, 1])
    a = jnp.stack(
        [
            cos_lat * jnp.cos(poly[:, 0]),
            cos_lat * jnp.sin(poly[:, 0]),
            jnp.sin(poly[:, 1]),
        ],
        axis=-1,
    )
    b = jnp.roll(a, -1, axis=0)
    p = jnp.asarray(points_xyz)
    numerator = einx.dot("n d, e d -> n e", p, jnp.cross(a, b))
    denominator = einx.subtract(
        "e, n e -> n e",
        einx.dot("e d, e d -> e", a, b),
        einx.dot("n d, e d -> n e", p, a) * einx.dot("n d, e d -> n e", p, b),
    )
    winding = jnp.sum(jnp.arctan2(numerator, denominator), axis=-1)
    return winding > math.pi
