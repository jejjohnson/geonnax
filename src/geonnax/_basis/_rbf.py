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


# Softening added under the square root so that gradients are finite when an
# evaluation point coincides with a centre (``r = 0``); bare ``sqrt`` has an
# infinite derivative there, which poisons reverse-mode through ``0 * inf``.
_EPS = 1e-12


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
    r = jnp.sqrt(sq + _EPS)
    z = einx.divide("n m, m -> n m", r, widths)
    return wendland_c2(z) if kernel == "wendland_c2" else wendland_c4(z)


__all__ = ["rbf_basis", "wendland_c2", "wendland_c4"]
