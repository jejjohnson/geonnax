r"""Divergence-free vector basis on the bounded box $[-L, L]^2$.

A 2D incompressible (divergence-free) velocity field is the skew gradient of a
scalar stream function, $\mathbf{u} = \nabla^\perp \psi = (-\partial_y \psi,
\partial_x \psi)$. Expanding $\psi$ in the box-Dirichlet Laplacian
eigenfunctions $\phi_j$ (see `fourier_basis`) gives a vector basis whose every
atom is divergence-free *by construction* — the natural reduced basis for ocean
currents, which are close to geostrophic and hence non-divergent.

Each atom is

$$
\boldsymbol{\varphi}_{j,k}(x, y) =
\big(-\phi_j(x)\, \phi_k'(y),\; \phi_j'(x)\, \phi_k(y)\big),
$$

so $\partial_x \varphi^{(x)} + \partial_y \varphi^{(y)} =
-\phi_j'(x)\phi_k'(y) + \phi_j'(x)\phi_k'(y) = 0$ identically. The eigenvalue of
the underlying stream function under $-\Delta$ is $\lambda_{j,k} = \lambda_j +
\lambda_k$, the same spectral tag the scalar Fourier basis returns, so a
consumer can impose a kinetic-energy spectral law over the modes.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax._basis._fourier import _to_tuple, fourier_eigenvalues


def _dirichlet_value_and_grad(
    x: Float[Array, " N"], num_basis: int, L: float
) -> tuple[Float[Array, "N M"], Float[Array, "N M"]]:
    r"""Return the 1D Dirichlet eigenfunctions and their derivatives on ``[-L, L]``.

    ``phi_j(x) = sin(j pi (x + L) / 2L) / sqrt(L)`` and its ``x``-derivative,
    each of shape ``(N, num_basis)``.
    """
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}.")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}.")
    j = jnp.arange(1, num_basis + 1, dtype=x.dtype)
    freq = j * (jnp.pi / (2.0 * L))  # (M,)
    arg = einx.dot("n, m -> n m", x + L, freq)  # (N, M)
    inv_sqrt_l = 1.0 / jnp.sqrt(L)
    value = jnp.sin(arg) * inv_sqrt_l
    grad = einx.multiply("n m, m -> n m", jnp.cos(arg), freq) * inv_sqrt_l
    return value, grad


def divfree_basis(
    xy: Float[Array, "N 2"],
    num_basis_per_dim: int | tuple[int, int],
    L: float | tuple[float, float],
) -> tuple[Float[Array, "N M two"], Float[Array, " M"]]:
    r"""Divergence-free vector basis from box-Dirichlet stream functions.

    Builds the skew-gradient atoms $\mathbf{u} = \nabla^\perp(\phi_j \otimes
    \phi_k)$ over $[-L, L]^2$. Every column is divergence-free analytically.

    Args:
        xy: Evaluation points of shape ``(N, 2)`` in $[-L, L]^2$.
        num_basis_per_dim: Per-axis number of 1D stream-function modes; an
            ``int`` is broadcast to both axes. Total atoms
            ``M = prod(num_basis_per_dim)``.
        L: Per-axis half-width; a scalar is broadcast to both axes.

    Returns:
        ``(Phi, lam)`` with ``Phi`` of shape ``(N, M, 2)`` (the last axis is the
        ``(u, v)`` velocity components) and ``lam`` of shape ``(M,)`` (the
        stream-function eigenvalues $\lambda_j + \lambda_k$, row-major over
        ``(j, k)``).

    Raises:
        ValueError: If ``xy`` is not ``(N, 2)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import divfree_basis
        >>> xy = jnp.zeros((5, 2))  # (N=5, 2)
        >>> Phi, lam = divfree_basis(xy, num_basis_per_dim=3, L=1.0)
        >>> (Phi.shape, lam.shape)  # M = 3 * 3 = 9
        ((5, 9, 2), (9,))
    """
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be 2D (N, 2); got shape {xy.shape}.")
    m_per = _to_tuple(num_basis_per_dim, 2, "num_basis_per_dim")
    l_per = _to_tuple(L, 2, "L")

    vx, gx = _dirichlet_value_and_grad(xy[:, 0], m_per[0], float(l_per[0]))  # (N, Mx)
    vy, gy = _dirichlet_value_and_grad(xy[:, 1], m_per[1], float(l_per[1]))  # (N, My)

    # u = -d/dy psi = -phi_j(x) phi_k'(y);  v = d/dx psi = phi_j'(x) phi_k(y).
    # Row-major flatten over (j, k) matches fourier_eigenvalues' layout.
    u = -einx.multiply("n a, n b -> n (a b)", vx, gy)
    v = einx.multiply("n a, n b -> n (a b)", gx, vy)
    phi = jnp.stack([u, v], axis=-1)  # (N, M, 2)

    lam = fourier_eigenvalues(num_basis_per_dim, L, 2, dtype=xy.dtype)
    return phi, lam


__all__ = ["divfree_basis"]
