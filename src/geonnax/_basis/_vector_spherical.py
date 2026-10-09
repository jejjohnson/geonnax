r"""Vector spherical harmonics: a basis for tangent vector fields on the sphere.

For each real spherical harmonic $Y_{lm}$ with $l \ge 1$:

- the gradient (poloidal, curl-free) harmonic $\Psi_{lm} = \nabla_S Y_{lm}$;
- the toroidal (divergence-free) harmonic $\Phi_{lm} = \hat r \times \nabla_S Y_{lm}$.

Together they span the square-integrable tangent fields on $S^2$ and give a
discrete Helmholtz decomposition: $\nabla_S\cdot\Psi_{lm} = -l(l+1)Y_{lm}$ and
$\nabla_S\cdot\Phi_{lm} = 0$.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from geonnax._basis._spherical import harmonic_degrees, real_spherical_harmonics
from geonnax._frames import _enu


def vector_spherical_harmonics(
    unit_xyz: Float[Array, "N 3"],
    l_max: int,
    *,
    output: Literal["cartesian", "enu"] = "cartesian",
    normalized: bool = True,
) -> tuple[Float[Array, "N M C"], Float[Array, "N M C"], Int[Array, " M"]]:
    r"""Gradient and toroidal vector spherical harmonics for $1 \le l \le l_{\max}$.

    The surface gradient is the tangential part of the ambient gradient of any
    smooth extension of $Y_{lm}$:
    $\nabla_S Y(u) = (I - uu^\top)\,\nabla\tilde Y(u)$. `real_spherical_harmonics`
    is a polynomial in $(x, y, z)$, so its Jacobian (by `jax.jacfwd`) gives
    $\Psi_{lm}$ smoothly everywhere, poles included. Then
    $\Phi_{lm} = u \times \Psi_{lm}$.

    Both families have $\int_{S^2}\lVert\Psi_{lm}\rVert^2 d\Omega =
    \int_{S^2}\lVert\Phi_{lm}\rVert^2 d\Omega = l(l+1)$; with
    ``normalized=True`` each is divided by $\sqrt{l(l+1)}$, giving an
    orthonormal set of $2\,((l_{\max}+1)^2 - 1)$ tangent fields.

    Args:
        unit_xyz: ``(N, 3)`` points on the unit sphere.
        l_max: Maximum degree, ``>= 1``.
        output: ``"cartesian"`` for 3-D vectors, or ``"enu"`` for their
            east/north components (see `geonnax.geo.enu_basis`; at exactly the
            poles the east/north frame is chosen by ``atan2(y, x)``).
        normalized: Divide by $\sqrt{l(l+1)}$.

    Returns:
        ``(psi, phi, degrees)``: ``psi`` and ``phi`` of shape ``(N, M, 3)``
        (or ``(N, M, 2)`` for ``"enu"``) with $M = (l_{\max}+1)^2 - 1$,
        columns in the order of `real_spherical_harmonics` without $l = 0$, and
        the degree ``l`` of each column.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import vector_spherical_harmonics
        >>> xyz = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        >>> psi, phi, degrees = vector_spherical_harmonics(xyz, 2)
        >>> psi.shape, phi.shape, degrees.tolist()
        ((2, 8, 3), (2, 8, 3), [1, 1, 1, 2, 2, 2, 2, 2])
    """
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}.")
    if output not in ("cartesian", "enu"):
        raise ValueError(f"output must be 'cartesian' or 'enu'; got {output!r}.")
    unit_xyz = jnp.asarray(unit_xyz)
    if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
        raise ValueError(f"unit_xyz must be (N, 3); got shape {unit_xyz.shape}.")

    def harmonics(point: Float[Array, " 3"]) -> Float[Array, " M"]:
        return real_spherical_harmonics(point[None, :], l_max)[0, 1:]

    grad = jax.vmap(jax.jacfwd(harmonics))(unit_xyz)  # (N, M, 3) ambient gradient
    radial = einx.dot("n m d, n d -> n m", grad, unit_xyz)
    psi = grad - einx.multiply("n m, n d -> n m d", radial, unit_xyz)
    phi = jnp.cross(unit_xyz[:, None, :], psi)

    degrees = np.asarray(harmonic_degrees(l_max)[1:])
    if normalized:
        scale = jnp.asarray(1.0 / np.sqrt(degrees * (degrees + 1.0)), dtype=psi.dtype)
        psi = einx.multiply("n m d, m -> n m d", psi, scale)
        phi = einx.multiply("n m d, m -> n m d", phi, scale)
    if output == "enu":
        lon = jnp.arctan2(unit_xyz[:, 1], unit_xyz[:, 0])
        lat = jnp.arcsin(jnp.clip(unit_xyz[:, 2], -1.0, 1.0))
        east_north = _enu(lon, lat)[:, :2, :]  # (N, 2, 3)
        psi = einx.dot("n m d, n c d -> n m c", psi, east_north)
        phi = einx.dot("n m d, n c d -> n m c", phi, east_north)
    return psi, phi, jnp.asarray(degrees)
