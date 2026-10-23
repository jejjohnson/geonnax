r"""Real spherical harmonics on the unit 2-sphere.

For ``l_max``, the basis is $(l_{\max} + 1)^2$ real SHs indexed by
$(l, m)$ with $l = 0, 1, \ldots, l_{\max}$ and
$m = -l, \ldots, l$. Inputs are unit Cartesian directions; the
azimuthal singularity at the poles is avoided by computing the
pole-free polynomial $\bar Q_l^m(z) = \bar P_l^m(z) / (1 - z^2)^{m/2}$
and pairing it with $(x + iy)^m$ directly — no
``arctan2`` or division by $\sin\theta$.

The $\bar Q_l^m$ come from the fully normalised recurrences shared with
`associated_legendre`, run as one `jax.lax.scan` over $l$, so the traced
graph does not grow with ``l_max`` and the values stay finite to high degree.

Tested via the addition theorem
$\sum_m Y_{lm}(x) Y_{lm}(x') = \frac{2l+1}{4\pi} P_l(x \cdot x')$.

References:
    Holmes & Featherstone (2002), *A unified approach to the Clenshaw
    summation and the recursive computation of very high degree and order
    normalised associated Legendre functions*, Journal of Geodesy 76(5),
    279–299.
"""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from geonnax._basis._legendre import _ldexp, _legendre_table_carry, _sectoral_scale


def harmonic_degrees(l_max: int) -> tuple[int, ...]:
    """Return the per-harmonic degree ``l`` for the flattened ``(l_max + 1)^2`` block.

    Layout matches `real_spherical_harmonics`: outer loop over ``l``,
    inner loop over ``m = -l, ..., l``.

    Examples:
        >>> from geonnax._basis import harmonic_degrees
        >>> # l=0 -> 1 col, l=1 -> 3 cols, l=2 -> 5 cols; total (l_max+1)^2 = 9
        >>> harmonic_degrees(2)
        (0, 1, 1, 1, 2, 2, 2, 2, 2)
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    return tuple(l for l in range(l_max + 1) for _ in range(2 * l + 1))


def _xy_powers(
    x: Float[Array, " N"], y: Float[Array, " N"], m_max: int
) -> tuple[Float[Array, "M N"], Float[Array, "M N"], Int[Array, "M N"]]:
    r"""$(x + iy)^m$ for ``m = 0, 1, ..., m_max`` as mantissas and a binary exponent.

    One `jax.lax.scan` over ``m``; rows are indexed by ``m``. The magnitude is
    $(x^2 + y^2)^{m/2} = \sin^m\theta$, which underflows float32 near the
    poles, so the real and imaginary parts share an int32 exponent and are
    renormalised by a power of two after every step: the value is
    ``(re + i im) · 2^exp`` (#122).
    """

    def step(carry, _):
        re, im, expo = carry
        # (x + iy) * (re + i im) = (x re - y im) + i (x im + y re)
        new_re, new_im = x * re - y * im, x * im + y * re
        biggest = jnp.maximum(jnp.abs(new_re), jnp.abs(new_im))
        _, shift = jnp.frexp(biggest)
        shift = jnp.where(biggest > 0, shift, 0)
        new_carry = (_ldexp(new_re, -shift), _ldexp(new_im, -shift), expo + shift)
        return new_carry, carry

    init = (jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros(x.shape, dtype=jnp.int32))
    _, (re, im, expo) = jax.lax.scan(step, init, None, length=m_max + 1)
    return re, im, expo


def real_spherical_harmonics(
    unit_xyz: Float[Array, "N 3"],
    l_max: int,
) -> Float[Array, "N M"]:
    r"""Evaluate real spherical harmonics up to degree ``l_max``.

    Convention (matches the standard "tesseral" real basis):

    $$
    \begin{aligned}
    Y_l^0    &= N_l^0\, P_l(\cos\theta)                          \\
    Y_l^m    &= \sqrt{2}\, N_l^m\, P_l^m(\cos\theta) \cos(m\phi)  & m > 0 \\
    Y_l^{-m} &= \sqrt{2}\, N_l^m\, P_l^m(\cos\theta) \sin(m\phi)  & m > 0
    \end{aligned}
    $$


    with $N_l^m = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}$ and the
    Condon–Shortley phase in $P_l^m$. The basis is orthonormal on the sphere,
    $\int_{S^2} Y_l^m Y_{l'}^{m'}\,d\Omega = \delta_{ll'}\delta_{mm'}$.

    The pole singularity is avoided by evaluating
    $P_l^m(z) \cos(m\phi) = Q_l^m(z) \cdot \mathrm{Re}[(x + iy)^m]$ with
    $Q_l^m = P_l^m / (1-z^2)^{m/2}$ (and similarly for the sine branch), so
    each $Y_l^m$ is a polynomial in $(x, y, z)$ and is differentiable
    everywhere, poles included. $Q_l^m$ is computed in the fully normalised
    form $\bar Q_l^m$ ($\int_{-1}^1 (\bar Q_l^m)^2 (1-z^2)^m\,dz = 1$) by the
    standard forward-column recurrences of Holmes & Featherstone (2002) with
    the factor $(1-z^2)^{m/2}$ taken out (their "modified forward column"):

    $$
    \begin{aligned}
    \bar Q_0^0 &= \tfrac{1}{\sqrt 2}, \qquad
    \bar Q_m^m = -\sqrt{\tfrac{2m+1}{2m}}\;\bar Q_{m-1}^{m-1}, \qquad
    \bar Q_{m+1}^m = \sqrt{2m+3}\;z\,\bar Q_m^m, \\
    \bar Q_l^m &= a_{lm}\big(z\,\bar Q_{l-1}^m - b_{lm}\,\bar Q_{l-2}^m\big),\quad
    a_{lm} = \sqrt{\tfrac{4l^2-1}{l^2-m^2}},\quad
    b_{lm} = \sqrt{\tfrac{(l-1)^2-m^2}{4(l-1)^2-1}},
    \end{aligned}
    $$

    and then $Y_l^0 = \bar Q_l^0 / \sqrt{2\pi}$,
    $Y_l^{m} = \bar Q_l^m\,\mathrm{Re}[(x+iy)^m] / \sqrt{\pi}$ and
    $Y_l^{-m} = \bar Q_l^m\,\mathrm{Im}[(x+iy)^m] / \sqrt{\pi}$. The
    recurrence runs over all orders at once inside one `jax.lax.scan` over
    $l$, so the traced graph has constant size in ``l_max`` (compile time
    stays at a second or so at ``l_max = 100``), and no factorial ratio is
    ever formed, so it does not overflow where the unnormalised
    $Q_l^m \sim (2m-1)!!$ would (near $l_{\max} \approx 30$ in float32,
    $85$ in float64). Near the poles $\bar Q_l^m$ itself grows like
    $\sin^{-m}\theta$ and $(x+iy)^m$ shrinks like $\sin^m\theta$, so each
    factor alone leaves float32's range from $l_{\max} \approx 180$ while
    their product is $O(1)$. Both are therefore carried as a mantissa and an
    int32 binary exponent, renormalised with `jnp.frexp` at every step, and
    combined by one power-of-two scaling at the end (#122): float32 stays finite
    (checked to $l_{\max} = 511$, poles included) and within about $4 \times 10^{-3}$
    of float64 at $l_{\max} = 255$.

    Pseudocode:

        c[m]       = Q̄_m^m constants                      # cumulative product, host
        Q[l], eQ   = scan over l of the 3-term recurrence  # (l, N, m), with exponents
        P[m], eP   = (x + iy)^m                            # scan over m, with exponents
        for each column (l, m), l outer, m = -l..l inner:
            Y[:, (l, m)] = ldexp(Q[l, :, |m|] * {1/√(2π), Re P[m]/√π, Im P[|m|]/√π},
                                 eQ[l, :, |m|] + eP[|m|])

    Args:
        unit_xyz: Inputs of shape ``(N, 3)``; assumed to lie on the unit
            sphere (no normalization performed).
        l_max: Maximum degree, inclusive. Total feature count ``M = (l_max + 1)^2``.

    Returns:
        ``(N, (l_max + 1)^2)`` array with the SH index flattened as
        outer loop over ``l``, inner loop over ``m = -l, ..., l``.

    References:
        Holmes & Featherstone (2002), *A unified approach to the Clenshaw
        summation and the recursive computation of very high degree and
        order normalised associated Legendre functions*, Journal of Geodesy
        76(5), 279–299.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import real_spherical_harmonics
        >>> # unit (N, 3) Cartesian inputs -> (N, (l_max + 1)^2) features
        >>> xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> real_spherical_harmonics(xyz, l_max=3).shape  # (2, 4**2)
        (2, 16)
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
        raise ValueError(f"unit_xyz must be (N, 3); got shape {unit_xyz.shape}.")

    dtype = (
        unit_xyz.dtype if jnp.issubdtype(unit_xyz.dtype, jnp.floating) else jnp.float32
    )
    unit_xyz = unit_xyz.astype(dtype)
    x = unit_xyz[:, 0]
    y = unit_xyz[:, 1]
    z = unit_xyz[:, 2]

    # Pole-free Q̄_l^m: the orthonormal recurrence seeded with the sectoral
    # constants instead of P̄_m^m = c_m (1 - z²)^{m/2}.
    seeds = jnp.asarray(_sectoral_scale(l_max), dtype=dtype)
    sectoral = einx.multiply("n, m -> n m", jnp.ones_like(z), seeds)
    # Both factors leave float32's range near the poles (Q̄ grows like s^{-m},
    # (x + iy)^m shrinks like s^m) while their product is O(1), so each is
    # carried as mantissa · 2^exponent and they are combined once (#122).
    q_mant, q_exp = _legendre_table_carry(
        z, sectoral, jnp.zeros(sectoral.shape, dtype=jnp.int32), l_max
    )  # (l, N, m) each
    re, im, xy_exp = _xy_powers(x, y, l_max)  # (m, N) each

    # Host-side column layout: l outer, m = -l, ..., l inner.
    ell = np.asarray(harmonic_degrees(l_max))
    order = np.concatenate([np.arange(-l, l + 1) for l in range(l_max + 1)])
    am = np.abs(order)
    scale = np.where(
        order == 0, 1.0 / math.sqrt(2.0 * math.pi), 1.0 / math.sqrt(math.pi)
    )

    q_cols = q_mant[ell, :, am]  # (F, N) mantissas
    trig = jnp.where(
        einx.id("f -> f 1", jnp.asarray(order > 0)),
        re[am],
        jnp.where(einx.id("f -> f 1", jnp.asarray(order < 0)), im[am], 1.0),
    )  # (F, N) mantissas; the m = 0 branch is exactly 1 with exponent 0
    expo = q_exp[ell, :, am] + xy_exp[am]  # (F, N)
    mant = einx.multiply(
        "f n, f n, f -> f n", q_cols, trig, jnp.asarray(scale, dtype=dtype)
    )
    return einx.id("f n -> n f", _ldexp(mant, expo))
