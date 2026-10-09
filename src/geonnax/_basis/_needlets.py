r"""Spherical needlets: a localised, multiscale tight frame on the sphere.

Needlets (Narcowich, Petrushev & Ward 2006) are band-limited bumps
$\psi_{jk}$ centred at the nodes $\xi_{jk}$ of a cubature rule at each scale
$j$. They are localised in space and frequency and form a tight frame:
$\sum_{jk}\langle f, \psi_{jk}\rangle\,\psi_{jk} = f - \bar f$ for
band-limited $f$ of degree $\le B^{j_{\max}}$.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from geonnax._basis._legendre import gauss_legendre, legendre_polynomials
from geonnax._basis._sphere_grids import gauss_legendre_grid


_BUMP_NODES, _BUMP_WEIGHTS = gauss_legendre(64)


def _bump(t: Float[Array, "..."]) -> Float[Array, "..."]:
    inside = jnp.abs(t) < 1.0
    safe = jnp.where(inside, t, 0.0)
    return jnp.where(inside, jnp.exp(-1.0 / (1.0 - safe**2)), 0.0)


def _smooth_step(u: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""$\psi(u) = \int_{-1}^{u} f / \int_{-1}^{1} f$ for the $C^\infty$ bump $f$."""
    u = jnp.clip(u, -1.0, 1.0)
    nodes = jnp.asarray(_BUMP_NODES, dtype=u.dtype)
    weights = jnp.asarray(_BUMP_WEIGHTS, dtype=u.dtype)
    half = 0.5 * (u + 1.0)
    # Gauss–Legendre nodes mapped from [-1, 1] onto [-1, u], for every u.
    t = jnp.expand_dims(half - 1.0, -1) + jnp.expand_dims(half, -1) * nodes
    partial = half * jnp.sum(weights * _bump(t), axis=-1)
    total = jnp.sum(weights * _bump(nodes))
    return partial / total


def needlet_window(
    B: float = 2.0,
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    r"""The needlet window $b$: smooth, supported in $[1/B, B]$, a partition of unity.

    With the $C^\infty$ step $\psi$ (the normalised integral of
    $e^{-1/(1-t^2)}$) and
    $\varphi(t) = 1$ for $t \le 1/B$,
    $\psi\!\big(1 - \tfrac{2B}{B-1}(t - \tfrac1B)\big)$ for
    $1/B \le t \le 1$, and $0$ for $t \ge 1$, the window is
    $b(\xi) = \sqrt{\varphi(\xi/B) - \varphi(\xi)}$. The sum
    $\sum_{j=0}^{J} b^2(\xi/B^j) = \varphi(\xi/B^{J+1}) - \varphi(\xi)$
    telescopes to $1$ for $1 \le \xi \le B^J$.

    Args:
        B: Scale factor, ``> 1``.

    Returns:
        A function evaluating $b$ elementwise.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import needlet_window
        >>> b = needlet_window(2.0)
        >>> xi = jnp.array([0.4, 1.0, 2.5])  # b vanishes outside [1/2, 2]
        >>> [round(float(v), 4) for v in b(xi)]
        [0.0, 1.0, 0.0]
    """
    if B <= 1.0:
        raise ValueError(f"B must be > 1, got {B}.")

    def phi(t: Float[Array, "..."]) -> Float[Array, "..."]:
        ramp = _smooth_step(1.0 - 2.0 * B / (B - 1.0) * (t - 1.0 / B))
        return jnp.where(t <= 1.0 / B, 1.0, jnp.where(t >= 1.0, 0.0, ramp))

    def window(xi: Float[Array, "..."]) -> Float[Array, "..."]:
        xi = jnp.asarray(xi)
        if not jnp.issubdtype(xi.dtype, jnp.floating):
            xi = xi.astype(jnp.float32)
        return jnp.sqrt(jnp.maximum(phi(xi / B) - phi(xi), 0.0))

    return window


def needlet_basis(
    unit_xyz: Float[Array, "N 3"],
    *,
    B: float = 2.0,
    j_max: int,
) -> tuple[Float[Array, "N M"], Int[Array, " M"], Float[Array, "M 3"]]:
    r"""Spherical needlets for scales $j = 0, \ldots, j_{\max}$, plus the constant.

    At scale $j$, with $L_j = \lfloor B^{j+1}\rfloor$ and the Gauss–Legendre
    grid $(\xi_{jk}, \lambda_{jk})$ exact to degree $2L_j$ (`gauss_legendre_grid`
    with $L_j + 1$ latitudes and $2L_j + 2$ longitudes),

    $$
    \psi_{jk}(x) = \sqrt{\lambda_{jk}}\sum_{l=0}^{L_j}
    b\!\left(\tfrac{l}{B^j}\right)\frac{2l+1}{4\pi}P_l(\xi_{jk}\cdot x).
    $$

    The first column is the constant $Y_{00} = 1/\sqrt{4\pi}$ (level ``-1``,
    centre ``(0, 0, 0)``), which the needlets omit since $b(0) = 0$. With it,
    $\sum_m \langle f, \psi_m\rangle\psi_m = f$ for every $f$ of degree
    $\le B^{j_{\max}}$.

    Args:
        unit_xyz: ``(N, 3)`` evaluation points on the unit sphere.
        B: Scale factor, ``> 1``.
        j_max: Finest scale, ``>= 0``.

    Returns:
        ``(values, levels, centres)``: ``(N, M)`` needlet values, the scale of
        each column, and each column's centre on the sphere.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import needlet_basis
        >>> pole = jnp.array([[0.0, 0.0, 1.0]])
        >>> values, levels, centres = needlet_basis(pole, j_max=1)
        >>> values.shape, sorted(set(levels.tolist()))  # 1 + 3·6 + 5·10 columns
        ((1, 69), [-1, 0, 1])
    """
    if j_max < 0:
        raise ValueError(f"j_max must be >= 0, got {j_max}.")
    unit_xyz = jnp.asarray(unit_xyz)
    if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
        raise ValueError(f"unit_xyz must be (N, 3); got shape {unit_xyz.shape}.")
    window = needlet_window(B)
    dtype = unit_xyz.dtype
    n = unit_xyz.shape[0]
    columns = [jnp.full((n, 1), 1.0 / math.sqrt(4.0 * math.pi), dtype=dtype)]
    levels = [-1]
    centres = [jnp.zeros((1, 3), dtype=dtype)]
    for j in range(j_max + 1):
        degree = math.floor(B ** (j + 1))
        nodes, weights = gauss_legendre_grid(degree + 1, 2 * degree + 2)
        nodes, weights = nodes.astype(dtype), weights.astype(dtype)
        ells = np.arange(degree + 1)
        coeffs = window(jnp.asarray(ells / B**j, dtype=dtype)) * jnp.asarray(
            (2.0 * ells + 1.0) / (4.0 * math.pi), dtype=dtype
        )
        cosines = jnp.clip(einx.dot("n d, k d -> n k", unit_xyz, nodes), -1.0, 1.0)
        kernel = einx.dot(
            "n k l, l -> n k", legendre_polynomials(cosines, degree), coeffs
        )
        columns.append(einx.multiply("n k, k -> n k", kernel, jnp.sqrt(weights)))
        levels += [j] * int(nodes.shape[0])
        centres.append(nodes)
    return (
        jnp.concatenate(columns, axis=1),
        jnp.asarray(levels),
        jnp.concatenate(centres, axis=0),
    )
