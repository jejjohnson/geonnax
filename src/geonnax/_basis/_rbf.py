r"""Placeable radial-basis / compact-support (Wendland) bases.

A radial basis places a kernel at a centre $c_a$ with width $\ell_a$ and
evaluates $\varphi_a(x) = K(\lVert x - c_a \rVert / \ell_a)$. Unlike the
spectral bases in this package (whose columns are global eigenfunctions with
fixed geometry), these columns are *local, placeable* bumps — put one at a
river mouth or a front and leave the rest of the domain untouched.

A Gaussian kernel gives a smooth global bump; the compactly supported Wendland
kernels give columns that are exactly zero past their width, so the basis (and
its Gram matrix) is sparse by construction. An RBF basis has no
eigendecomposition; the prior variance per centre is prescribed downstream, so
the only geometry it needs is the caller's own ``centers`` / ``widths``.

References:
    Wendland (2004), *Scattered Data Approximation*.
    Buhmann (2003), *Radial Basis Functions*.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float


# A square-root that is exact (returns 0) at ``sq == 0`` yet has finite
# gradients there: the inner ``where`` keeps the sqrt input away from 0 on the
# backward pass, while the outer ``where`` restores the exact 0 forward value.
# Unlike adding a fixed epsilon, this never shifts the value — important for
# narrow atoms whose width is comparable to a softening constant.
def _safe_sqrt(sq: Float[Array, "..."]) -> Float[Array, "..."]:
    positive = sq > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)


# arccos whose gradient is finite at the endpoints ``|c| == 1``: ``arccos`` has
# an infinite derivative there, so a coincident point/centre (``c == 1``, zero
# geodesic distance) would otherwise produce NaN cotangents even though the
# forward value is the finite peak. The inner ``where`` keeps the backward pass
# off the singularity; the outer ``where`` restores the exact value (0 at c=1,
# pi at c=-1). The spherical analogue of ``_safe_sqrt``.
def _safe_arccos(c: Float[Array, "..."]) -> Float[Array, "..."]:
    interior = jnp.abs(c) < 1.0
    safe_c = jnp.where(interior, c, 0.0)
    return jnp.where(interior, jnp.arccos(safe_c), jnp.where(c > 0.0, 0.0, jnp.pi))


def wendland_c2(r: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Wendland $C^2$ kernel, positive definite in dimensions $d \le 3$.

    $$K(r) = (1 - r)_+^4 (4r + 1).$$

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import wendland_c2
        >>> bool(wendland_c2(jnp.array(0.0)) == 1.0)  # peak at r = 0
        True
        >>> bool(wendland_c2(jnp.array(1.5)) == 0.0)  # zero past the support
        True
    """
    return jnp.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)


def wendland_c4(r: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Wendland $C^4$ kernel, positive definite in dimensions $d \le 3$.

    $$K(r) = (1 - r)_+^6 (35 r^2 + 18 r + 3) / 3.$$
    """
    return jnp.where(
        r < 1.0, (1.0 - r) ** 6 * (35.0 * r**2 + 18.0 * r + 3.0) / 3.0, 0.0
    )


def rbf_basis(
    x: Float[Array, "N d"],
    centers: Float[Array, "M d"],
    widths: Float[Array, " M"],
    *,
    kernel: Literal["gaussian", "wendland_c2", "wendland_c4"] = "gaussian",
) -> Float[Array, "N M"]:
    r"""Evaluate a placeable radial basis at given centres and widths.

    Column $a$ is $\varphi_a(x) = K(\lVert x - c_a \rVert / \ell_a)$.
    Returns only the basis matrix $\Phi$; the prior variance per centre is a
    downstream choice and the relevant geometry (``centers``, ``widths``) is
    already the caller's. Compactly supported (Wendland) columns are exactly
    zero past their width.

    Args:
        x: Evaluation points, shape ``(N, d)``.
        centers: Atom centres, shape ``(M, d)``.
        widths: Per-atom width $\ell_a$ (Gaussian length scale, or Wendland
            support radius), shape ``(M,)``.
        kernel: ``"gaussian"`` (smooth global bump) or ``"wendland_c2"`` /
            ``"wendland_c4"`` (compact support, positive definite for $d \le 3$).

    Returns:
        Basis matrix of shape ``(N, M)``.

    Raises:
        ValueError: If ``kernel`` is not a recognised name.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import rbf_basis
        >>> x = jnp.zeros((5, 2))
        >>> c = jnp.ones((3, 2))
        >>> rbf_basis(x, c, jnp.ones(3), kernel="wendland_c2").shape
        (5, 3)
    """
    if kernel not in ("gaussian", "wendland_c2", "wendland_c4"):
        raise ValueError(f"unknown kernel {kernel!r}.")
    diff = einx.subtract("n d, m d -> n m d", x, centers)
    sq = einx.sum("n m d -> n m", diff**2)  # squared distance
    if kernel == "gaussian":
        # Use the squared distance directly — no sqrt, so grads are clean at r=0.
        z2 = einx.divide("n m, m -> n m", sq, widths**2)
        return jnp.exp(-0.5 * z2)
    r = _safe_sqrt(sq)
    z = einx.divide("n m, m -> n m", r, widths)
    return wendland_c2(z) if kernel == "wendland_c2" else wendland_c4(z)


def spherical_rbf_basis(
    unit_xyz: Float[Array, "N 3"],
    centers: Float[Array, "M 3"],
    widths: Float[Array, " M"],
    *,
    kernel: Literal["gaussian", "wendland_c2", "wendland_c4"] = "gaussian",
) -> Float[Array, "N M"]:
    r"""Placeable radial basis on the unit 2-sphere, in geodesic distance.

    The spherical counterpart of `rbf_basis`: column $a$ is $\varphi_a(x) =
    K(d(x, c_a) / \ell_a)$ with $d$ the **great-circle** (geodesic) distance
    $d(x, c) = \arccos(\langle x, c \rangle)$ rather than the Euclidean (chordal)
    distance, so an atom's support is a geodesic cap and is rotation-invariant on
    the sphere. Put one at a coastline or an eddy and leave the rest of the globe
    untouched. Inputs are assumed to lie on the unit sphere (not normalised
    here). As with `rbf_basis`, only $\Phi$ is returned; the prior variance per
    centre is a downstream choice.

    Args:
        unit_xyz: Evaluation directions, shape ``(N, 3)``, on the unit sphere.
        centers: Atom centres, shape ``(M, 3)``, on the unit sphere.
        widths: Per-atom width $\ell_a$ (geodesic length scale or Wendland cap
            radius, in radians), shape ``(M,)``.
        kernel: ``"gaussian"`` (smooth cap) or ``"wendland_c2"`` /
            ``"wendland_c4"`` (compact geodesic support).

    Returns:
        Basis matrix of shape ``(N, M)``.

    Raises:
        ValueError: If ``kernel`` is not a recognised name.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import spherical_rbf_basis
        >>> x = jnp.eye(3)[:2]      # two points on the sphere, (N=2, 3)
        >>> c = jnp.eye(3)          # three centres, (M=3, 3)
        >>> spherical_rbf_basis(x, c, jnp.ones(3)).shape
        (2, 3)
    """
    if kernel not in ("gaussian", "wendland_c2", "wendland_c4"):
        raise ValueError(f"unknown kernel {kernel!r}.")
    # Great-circle distance via a gradient-safe arccos: exact at a coincident
    # centre (dot == 1, distance 0) yet with finite gradients there, and robust
    # to dot products that float just outside [-1, 1].
    cos_ang = einx.dot("n d, m d -> n m", unit_xyz, centers)
    dist = _safe_arccos(cos_ang)
    if kernel == "gaussian":
        z2 = einx.divide("n m, m -> n m", dist**2, widths**2)
        return jnp.exp(-0.5 * z2)
    z = einx.divide("n m, m -> n m", dist, widths)
    return wendland_c2(z) if kernel == "wendland_c2" else wendland_c4(z)


__all__ = ["rbf_basis", "spherical_rbf_basis", "wendland_c2", "wendland_c4"]
