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
from jaxtyping import Array, Float


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

    Computed with the fully normalised recurrences, which stay finite to high
    degree:

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
    n = l_max + 1
    m = np.arange(n)

    # Sectoral values P̄_m^m for every m, by a cumulative product over m.
    sect_factor = np.ones(n)
    sect_factor[1:] = -np.sqrt((2.0 * m[1:] + 1.0) / (2.0 * m[1:]))
    sect_scale = np.cumprod(sect_factor) / math.sqrt(2.0)  # (n,)
    s_pow = s[..., None] ** jnp.asarray(m, dtype=dtype)  # (..., n)
    sectoral = jnp.asarray(sect_scale, dtype=dtype) * s_pow  # P̄_m^m

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
    sub_scale = np.sqrt(2.0 * m + 3.0)  # P̄_{m+1}^m = √(2m+3) t P̄_m^m

    def step(carry, inputs):
        p_lm2, p_lm1 = carry
        a, b, d, sb = inputs
        general = a * (t[..., None] * p_lm1 - b * p_lm2)
        first_off = jnp.asarray(sub_scale, dtype=dtype) * t[..., None] * sectoral
        p_l = d * sectoral + sb * first_off + (1.0 - d - sb) * general
        return (p_lm1, p_l), p_l

    zeros = jnp.zeros_like(sectoral)
    p_00 = jnp.asarray(diag[0], dtype=dtype) * sectoral
    if n > 1:
        xs = tuple(
            jnp.asarray(arr[1:], dtype=dtype) for arr in (a_all, b_all, diag, sub)
        )
        _, rows = jax.lax.scan(step, (zeros, p_00), xs)
        table = jnp.concatenate([p_00[None], rows], axis=0)  # (n_l, ..., n_m)
    else:
        table = p_00[None]

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
