r"""Product bases on a spherical shell $r_0 \le \lVert x\rVert \le r_1$.

For fields with depth or altitude, each basis function is a real spherical
harmonic in direction times a polynomial in radius,
$\phi_{lmn}(x) = Y_{lm}(x/\lVert x\rVert)\,R_n(\lVert x\rVert)$.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from geonnax._basis._legendre import legendre_polynomials
from geonnax._basis._spherical import harmonic_degrees, real_spherical_harmonics


def _chebyshev(s: Float[Array, " N"], n_radial: int) -> Float[Array, "N K"]:
    columns = [jnp.ones_like(s)]
    if n_radial > 1:
        columns.append(s)
    for _ in range(2, n_radial):
        columns.append(2.0 * s * columns[-1] - columns[-2])
    return jnp.stack(columns, axis=-1)


def shell_basis(
    xyz: Float[Array, "N 3"],
    l_max: int,
    n_radial: int,
    *,
    r_inner: float,
    r_outer: float,
    radial: Literal["legendre", "chebyshev"] = "legendre",
) -> tuple[Float[Array, "N M"], Int[Array, "M 2"]]:
    r"""Spherical harmonics times radial polynomials on a spherical shell.

    With $s = 2\frac{\lVert x\rVert - r_0}{r_1 - r_0} - 1 \in [-1, 1]$:

    - ``radial="legendre"``: $R_n = \sqrt{\tfrac{2}{r_1 - r_0}}\,
      \sqrt{\tfrac{2n+1}{2}}\,P_n(s)$, so the basis is orthonormal in
      $L^2$ with the measure $d\Omega\,dr$. This is **not** the volume
      measure $r^2\,dr\,d\Omega$; for a thin shell the two differ by roughly
      the factor $r^2$.
    - ``radial="chebyshev"``: the Chebyshev polynomials $T_n(s)$,
      unnormalised. They are well-conditioned but not orthogonal under
      $dr$.

    Exact Laplacian eigenfunctions of the shell (spherical Bessel functions in
    radius) are out of scope.

    Args:
        xyz: ``(N, 3)`` points with radius in ``[r_inner, r_outer]``.
        l_max: Maximum spherical-harmonic degree.
        n_radial: Number of radial polynomials, ``>= 1``.
        r_inner: Inner radius $r_0 \ge 0$.
        r_outer: Outer radius $r_1 > r_0$.
        radial: ``"legendre"`` or ``"chebyshev"``.

    Returns:
        ``(values, indices)``: ``(N, M)`` with
        $M = (l_{\max}+1)^2\,n_{\text{radial}}$, harmonic-major (the radial
        index varies fastest), and ``(M, 2)`` giving each column's ``(l, n)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import shell_basis
        >>> xyz = jnp.array([[0.0, 0.0, 1.5], [2.0, 0.0, 0.0]])
        >>> values, indices = shell_basis(xyz, 2, 3, r_inner=1.0, r_outer=2.0)
        >>> values.shape, indices[:4].tolist()  # (l, n) per column
        ((2, 27), [[0, 0], [0, 1], [0, 2], [1, 0]])
    """
    if n_radial < 1:
        raise ValueError(f"n_radial must be >= 1, got {n_radial}.")
    if not 0.0 <= r_inner < r_outer:
        raise ValueError(f"need 0 <= r_inner < r_outer; got {r_inner}, {r_outer}.")
    if radial not in ("legendre", "chebyshev"):
        raise ValueError(f"radial must be 'legendre' or 'chebyshev'; got {radial!r}.")
    xyz = jnp.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (N, 3); got shape {xyz.shape}.")
    radius = jnp.linalg.norm(xyz, axis=-1)
    unit = xyz / jnp.where(radius > 0, radius, 1.0)[:, None]
    harmonics = real_spherical_harmonics(unit, l_max)  # (N, H)
    s = 2.0 * (radius - r_inner) / (r_outer - r_inner) - 1.0
    if radial == "legendre":
        n = np.arange(n_radial)
        scale = np.sqrt(2.0 / (r_outer - r_inner)) * np.sqrt((2.0 * n + 1.0) / 2.0)
        radial_values = legendre_polynomials(s, n_radial - 1) * jnp.asarray(
            scale, dtype=s.dtype
        )
    else:
        radial_values = _chebyshev(s, n_radial)
    values = einx.multiply("n h, n k -> n (h k)", harmonics, radial_values)
    degrees = np.repeat(np.asarray(harmonic_degrees(l_max)), n_radial)
    radial_index = np.tile(np.arange(n_radial), (l_max + 1) ** 2)
    return values, jnp.asarray(np.stack([degrees, radial_index], axis=-1))
