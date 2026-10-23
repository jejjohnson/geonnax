r"""Legendre polynomials, associated Legendre functions and Gauss–Legendre nodes.

The shared one-dimensional building blocks of the spherical bases:

- `legendre_polynomials` — $P_0, \ldots, P_{l_{\max}}$ by the three-term
  recurrence $(l+1)P_{l+1} = (2l+1)tP_l - lP_{l-1}$.
- `associated_legendre` — $P_l^m$ for $0 \le m \le l \le l_{\max}$, by the
  fully normalised recurrences (stable to high degree), in three
  normalisations.
- `associated_legendre_indices` — the ``(l, m)`` of each column of
  `associated_legendre`.
- `gauss_legendre` — Gauss–Legendre nodes and weights on $[-1, 1]$.
"""

from __future__ import annotations

import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def legendre_polynomials(t: Float[Array, "..."], l_max: int) -> Float[Array, "... L"]:
    r"""Legendre polynomials $P_0(t), \ldots, P_{l_{\max}}(t)$.

    Evaluated by the three-term recurrence

    $$
    P_0 = 1,\qquad P_1 = t,\qquad (l+1)P_{l+1} = (2l+1)\,t\,P_l - l\,P_{l-1},
    $$

    run as a `jax.lax.scan` over $l$, so the trace size does not grow with
    ``l_max``.

    Args:
        t: Points in $[-1, 1]$, any shape.
        l_max: Maximum degree, inclusive.

    Returns:
        Array of shape ``t.shape + (l_max + 1,)``; the last axis is the degree.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import legendre_polynomials
        >>> P = legendre_polynomials(jnp.array([0.5]), 2)
        >>> [round(float(v), 4) for v in P[0]]  # 1, t, (3t² − 1)/2
        [1.0, 0.5, -0.125]
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    t = jnp.asarray(t)
    p0 = jnp.ones_like(t)
    if l_max == 0:
        return p0[..., None]

    def step(carry, ell):
        p_prev, p_curr = carry
        p_next = ((2.0 * ell + 1.0) * t * p_curr - ell * p_prev) / (ell + 1.0)
        return (p_curr, p_next), p_next

    ells = jnp.arange(1, l_max, dtype=t.dtype)
    _, rest = jax.lax.scan(step, (p0, t), ells)
    stacked = jnp.concatenate([p0[None], t[None], rest], axis=0)
    return jnp.moveaxis(stacked, 0, -1)


def associated_legendre_indices(l_max: int) -> tuple[tuple[int, int], ...]:
    """The ``(l, m)`` pair of each column of `associated_legendre`.

    Columns run with ``l`` outermost and ``m = 0, ..., l`` inner, so there are
    ``(l_max + 1)(l_max + 2) / 2`` of them.

    Examples:
        >>> from geonnax.basis import associated_legendre_indices
        >>> associated_legendre_indices(2)
        ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    return tuple((ell, m) for ell in range(l_max + 1) for m in range(ell + 1))


def associated_legendre(
    t: Float[Array, "..."],
    l_max: int,
    *,
    normalization: Literal["orthonormal", "schmidt", "none"] = "orthonormal",
) -> Float[Array, "... M"]:
    r"""Associated Legendre functions $P_l^m(t)$ for $0 \le m \le l \le l_{\max}$.

    Computed with the fully normalised recurrences, carried as a mantissa and a
    binary exponent per order (`_legendre_table_carry`), so they stay finite
    and accurate to high degree in float32 as well as float64 — Gram error on
    the $n$-point Gauss rule about $3 \times 10^{-5}$ in float32 at $l_{\max} = 511$:

    $$
    \begin{aligned}
    \bar P_0^0 &= 1/\sqrt2, \qquad
    \bar P_m^m = -\sqrt{\tfrac{2m+1}{2m}}\,\sqrt{1-t^2}\;\bar P_{m-1}^{m-1}, \qquad
    \bar P_{m+1}^m = \sqrt{2m+3}\;t\,\bar P_m^m, \\
    \bar P_l^m &= a_{lm}\big(t\,\bar P_{l-1}^m - b_{lm}\,\bar P_{l-2}^m\big),\quad
    a_{lm} = \sqrt{\tfrac{4l^2-1}{l^2-m^2}},\quad
    b_{lm} = \sqrt{\tfrac{(l-1)^2-m^2}{4(l-1)^2-1}}.
    \end{aligned}
    $$

    Normalisations:

    - ``"orthonormal"``: $\int_{-1}^{1}\bar P_l^m\,\bar P_{l'}^m\,dt =
      \delta_{ll'}$, with the Condon–Shortley phase $(-1)^m$, the phase of
      `real_spherical_harmonics`. The real spherical harmonic is
      $Y_l^0 = \bar P_l^0/\sqrt{2\pi}$ and
      $Y_l^{\pm m} = \bar P_l^m\,\{\cos, \sin\}(m\phi)/\sqrt{\pi}$.
    - ``"schmidt"``: Schmidt semi-normalised,
      $\sqrt{(2-\delta_{m0})\,(l-m)!/(l+m)!}\;P_l^m$, **without** the
      Condon–Shortley phase (the geomagnetism convention).
    - ``"none"``: the classical $P_l^m$ with the Condon–Shortley phase (as in
      ``scipy.special.lpmv``); it overflows in float64 near $l + m \approx 340$.

    Args:
        t: Points in $[-1, 1]$, any shape.
        l_max: Maximum degree, inclusive.
        normalization: One of ``"orthonormal"``, ``"schmidt"``, ``"none"``.

    Returns:
        Array of shape ``t.shape + ((l_max + 1)(l_max + 2) / 2,)``; columns in
        the order of `associated_legendre_indices`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import associated_legendre
        >>> P = associated_legendre(jnp.array([0.0]), 1, normalization="none")
        >>> [round(float(v), 4) for v in P[0]]  # P_0^0, P_1^0, P_1^1 at t = 0
        [1.0, 0.0, -1.0]
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    if normalization not in ("orthonormal", "schmidt", "none"):
        raise ValueError(
            "normalization must be 'orthonormal', 'schmidt' or 'none'; "
            f"got {normalization!r}."
        )
    t = jnp.asarray(t)
    dtype = t.dtype if jnp.issubdtype(t.dtype, jnp.floating) else jnp.float32
    t = t.astype(dtype)
    s = jnp.sqrt(jnp.maximum(1.0 - t**2, 0.0))

    # Sectoral values P̄_m^m = mantissa · 2^exponent for every m. The exponent
    # carry keeps (1 - t²)^{m/2} from underflowing near the poles, where P̄_l^m
    # grows back to O(1) for l ≫ m (#122).
    sec_mant, sec_exp = _sectoral_carry(s, l_max)  # (..., n), (..., n)
    mant, expo = _legendre_table_carry(t, sec_mant, sec_exp, l_max)  # (n_l, ..., n_m)
    table = _ldexp(mant, expo)

    pairs = associated_legendre_indices(l_max)
    ell_idx = np.array([ell for ell, _ in pairs])
    m_idx = np.array([mm_ for _, mm_ in pairs])
    out = jnp.moveaxis(table, 0, -2)[..., ell_idx, m_idx]

    if normalization == "orthonormal":
        return out
    # Convert P̄ (unit L² norm on [-1, 1]) to the classical P_l^m.
    log_ratio = np.array(
        [math.lgamma(ell + mm_ + 1) - math.lgamma(ell - mm_ + 1) for ell, mm_ in pairs]
    )
    to_classical = np.sqrt(2.0 / (2.0 * ell_idx + 1.0) * np.exp(log_ratio))
    if normalization == "none":
        return out * jnp.asarray(to_classical, dtype=dtype)
    # Schmidt: √((2 − δ_m0)(l−m)!/(l+m)!) P_l^m, without the Condon–Shortley phase.
    schmidt = np.sqrt(2.0 * (2.0 - (m_idx == 0)) / (2.0 * ell_idx + 1.0))
    schmidt = schmidt * (-1.0) ** m_idx
    return out * jnp.asarray(schmidt, dtype=dtype)


def _sectoral_scale(l_max: int) -> np.ndarray:
    r"""Sectoral constants $c_m = \bar P_m^m / (1 - t^2)^{m/2}$, $m \le l_{\max}$.

    $c_0 = 1/\sqrt2$ and $c_m = -\sqrt{(2m+1)/(2m)}\;c_{m-1}$, the
    orthonormal sectoral recurrence with the factor $\sqrt{1-t^2}$ taken out.
    """
    m = np.arange(l_max + 1)
    factor = np.ones(l_max + 1)
    factor[1:] = -np.sqrt((2.0 * m[1:] + 1.0) / (2.0 * m[1:]))
    return np.cumprod(factor) / math.sqrt(2.0)


def _ldexp(x: Float[Array, "..."], e: Int[Array, "..."]) -> Float[Array, "..."]:
    r"""$x \cdot 2^e$ with the exact derivative $2^e$ everywhere.

    `jnp.ldexp` passes ``x == 0`` through a ``where``, so its derivative there
    is 1 instead of $2^e$; the Legendre values are exactly zero at many points
    (odd $l - m$ at $t = 0$, $(x + iy)^m$ on the axes) where their derivative is
    not. The power of two is split in two halves, and ``e`` is clipped to the
    range where both halves are finite, so ``x == 0`` gives 0 (never
    ``0 · inf``) and a non-zero ``x`` overflows or underflows only when the
    result itself does.
    """
    max_half = jnp.finfo(x.dtype).maxexp - 1  # largest finite power of two
    e = jnp.clip(e, -2 * max_half, 2 * max_half)
    half = e // 2
    two = jnp.asarray(2.0, dtype=x.dtype)
    return x * two ** half.astype(x.dtype) * two ** (e - half).astype(x.dtype)


def _sectoral_carry(
    s: Float[Array, "..."], l_max: int
) -> tuple[Float[Array, "... L"], Int[Array, "... L"]]:
    r"""Sectoral values $\bar P_m^m = c_m s^m$ as mantissa and binary exponent.

    Runs $\bar P_m^m = -\sqrt{(2m+1)/(2m)}\;s\,\bar P_{m-1}^{m-1}$ as a
    `jax.lax.scan` over $m$ and renormalises with `jnp.frexp` at every step, so
    the value is ``_ldexp(mantissa, exponent)`` with the mantissa in
    $[\tfrac12, 1)$: $s^m$ never underflows, however small $s$ and large $m$.
    Where $s = 0$ the mantissa is $0$ for $m \ge 1$, which is exact.

    Args:
        s: $\sqrt{1 - t^2}$, any shape.
        l_max: Maximum order, inclusive.

    Returns:
        ``(mantissa, exponent)``, each of shape ``s.shape + (l_max + 1,)``.
    """
    dtype = s.dtype
    m = np.arange(1, l_max + 1)
    factors = jnp.asarray(-np.sqrt((2.0 * m + 1.0) / (2.0 * m)), dtype=dtype)
    mant0 = jnp.full_like(s, 1.0 / math.sqrt(2.0))
    exp0 = jnp.zeros(s.shape, dtype=jnp.int32)

    def step(carry, factor):
        mant, expo = carry
        value = factor * s * mant
        _, shift = jnp.frexp(value)
        shift = jnp.where(value != 0, shift, 0)
        mant, expo = _ldexp(value, -shift), expo + shift
        return (mant, expo), (mant, expo)

    _, (mants, expos) = jax.lax.scan(step, (mant0, exp0), factors)
    mant = jnp.concatenate([mant0[None], mants], axis=0)  # (L, ...)
    expo = jnp.concatenate([exp0[None], expos], axis=0)
    return jnp.moveaxis(mant, 0, -1), jnp.moveaxis(expo, 0, -1)


def _legendre_table(
    t: Float[Array, "..."], sectoral: Float[Array, "... L"], l_max: int
) -> Float[Array, "L ... L"]:
    r"""Run the orthonormal recurrence in $l$ for every order $m$ at once.

    Given the sectoral row ``sectoral[..., m]`` $= \bar P_m^m$, fills
    $\bar P_{m+1}^m = \sqrt{2m+3}\,t\,\bar P_m^m$ and
    $\bar P_l^m = a_{lm}(t\,\bar P_{l-1}^m - b_{lm}\,\bar P_{l-2}^m)$ in one
    `jax.lax.scan` over $l$, so the trace size does not grow with ``l_max``.
    The map is linear in ``sectoral``: passing the constants of
    `_sectoral_scale` (without $(1-t^2)^{m/2}$) gives the pole-free
    polynomials $\bar P_l^m(t) / (1-t^2)^{m/2}$ that `real_spherical_harmonics`
    uses.

    This is `_legendre_table_carry` with a zero seed exponent, returned as
    values.

    Returns:
        Array of shape ``(l_max + 1,) + t.shape + (l_max + 1,)``, indexed
        ``[l, ..., m]``; entries with ``m > l`` are zero.
    """
    mant, expo = _legendre_table_carry(
        t, sectoral, jnp.zeros(sectoral.shape, dtype=jnp.int32), l_max
    )
    return _ldexp(mant, expo)


def _legendre_table_carry(
    t: Float[Array, "..."],
    sectoral: Float[Array, "... L"],
    sectoral_exp: Int[Array, "... L"],
    l_max: int,
) -> tuple[Float[Array, "L ... L"], Int[Array, "L ... L"]]:
    r"""The recurrence of `_legendre_table` with a binary-exponent carry.

    Each order $m$ is carried as a mantissa and an integer exponent,
    $\bar P_l^m = \mathfrak m_l^m\,2^{e^m}$. The recurrence is linear and
    homogeneous in each column, so a column's two carried rows can share one
    exponent; after every step both are rescaled by the power of two
    (`jnp.frexp`) that brings the larger into $[\tfrac12, 1)$. No intermediate
    leaves the floating-point range, so float32 stays accurate where
    $\bar P_m^m \propto (1-t^2)^{m/2}$ is far below its smallest normal number
    but $\bar P_l^m$, $l \gg m$, is $O(1)$ (#122). The exponent shift is an
    integer, so gradients flow through the mantissas only, which is exact.

    Args:
        t: Points in $[-1, 1]$, any shape.
        sectoral: Mantissas of the seeds $\bar P_m^m$, shape ``t.shape + (L,)``.
        sectoral_exp: Their binary exponents (int32), same shape.
        l_max: Maximum degree, inclusive; ``L = l_max + 1``.

    Returns:
        ``(mantissa, exponent)``, each of shape ``(L,) + t.shape + (L,)``,
        indexed ``[l, ..., m]``; the value is ``_ldexp(mantissa, exponent)``
        and entries with ``m > l`` are zero.

    References:
        Fukushima, T. (2012). Numerical computation of spherical harmonics of
        arbitrary degree and order by extending exponent of floating point
        numbers. *Journal of Geodesy* 86, 271–285 (the "X-number" idea, here
        with one shared exponent per column).
    """
    dtype = sectoral.dtype
    n = l_max + 1
    m = np.arange(n)

    # Recurrence coefficients a_lm, b_lm (host constants), for rows l >= 1.
    # They are only used where m <= l - 2; elsewhere the masks below select the
    # sectoral (m = l) and first off-diagonal (m = l - 1) values instead.
    ells = np.arange(n, dtype=float)[:, None]  # (n, 1)
    mm = m[None, :].astype(float)  # (1, n)
    general_mask = mm <= ells - 2
    a_all = np.where(
        general_mask,
        np.sqrt(
            np.where(general_mask, 4.0 * ells**2 - 1.0, 0.0)
            / np.where(general_mask, ells**2 - mm**2, 1.0)
        ),
        0.0,
    )
    b_all = np.where(
        general_mask,
        np.sqrt(
            np.where(general_mask, (ells - 1.0) ** 2 - mm**2, 0.0)
            / np.where(general_mask, 4.0 * (ells - 1.0) ** 2 - 1.0, 1.0)
        ),
        0.0,
    )
    diag = (m[None, :] == np.arange(n)[:, None]).astype(float)  # (n, n): m == l
    sub = (m[None, :] == np.arange(n)[:, None] - 1).astype(float)  # m == l - 1
    sub_scale = jnp.asarray(np.sqrt(2.0 * m + 3.0), dtype=dtype)  # √(2m+3)

    def step(carry, inputs):
        # Carried rows P̄_{l-2}^m, P̄_{l-1}^m in units of 2^expo (one per column).
        p_lm2, p_lm1, expo = carry
        a, b, d, sb = inputs
        general = a * (t[..., None] * p_lm1 - b * p_lm2)
        # P̄_{m+1}^m = √(2m+3) t P̄_m^m: the previous row is the sectoral one.
        first_off = sub_scale * t[..., None] * p_lm1
        # A column's sectoral row arrives while its carry is still zero, so its
        # exponent is still the seed's.
        p_l = d * sectoral + sb * first_off + (1.0 - d - sb) * general
        biggest = jnp.maximum(jnp.abs(p_l), jnp.abs(p_lm1))
        _, shift = jnp.frexp(biggest)
        shift = jnp.where(biggest > 0, shift, 0)
        new_carry = (_ldexp(p_lm1, -shift), _ldexp(p_l, -shift), expo + shift)
        return new_carry, (p_l, expo)

    zeros = jnp.zeros_like(sectoral)
    p_00 = jnp.asarray(diag[0], dtype=dtype) * sectoral
    if n > 1:
        xs = tuple(
            jnp.asarray(arr[1:], dtype=dtype) for arr in (a_all, b_all, diag, sub)
        )
        _, (rows, row_exps) = jax.lax.scan(step, (zeros, p_00, sectoral_exp), xs)
        return (
            jnp.concatenate([p_00[None], rows], axis=0),
            jnp.concatenate([sectoral_exp[None], row_exps], axis=0),
        )
    return p_00[None], sectoral_exp[None]


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Gauss–Legendre nodes and weights on $[-1, 1]$.

    Host-side constants (``numpy.polynomial.legendre.leggauss``): an
    ``n``-point rule integrates polynomials of degree $\le 2n - 1$ exactly.

    Args:
        n: Number of nodes, ``>= 1``.

    Returns:
        ``(nodes, weights)``, two NumPy arrays of shape ``(n,)``.

    Examples:
        >>> from geonnax.basis import gauss_legendre
        >>> nodes, weights = gauss_legendre(3)
        >>> round(float(weights.sum()), 6)  # ∫_{-1}^{1} 1 dt = 2
        2.0
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    return np.polynomial.legendre.leggauss(n)
