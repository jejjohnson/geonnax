r"""Laplacian eigenfunctions on the bounded box $[-L, L]^D$.

Dirichlet eigenpairs of $-d^2/dx^2$ on $[-L, L]$:

$$
\phi_j(x) = \frac{1}{\sqrt{L}} \sin\!\left(\frac{j\pi(x + L)}{2L}\right),
\qquad
\lambda_j = \left(\frac{j\pi}{2L}\right)^2,
\qquad j = 1, 2, \ldots
$$


The 1D basis is $L^2([-L, L])$-orthonormal. On a $D$-dimensional
box the basis is the tensor product, indexed by a multi-index
$(j_1, \ldots, j_D)$; we flatten in row-major order. These eigenfunctions
are the engine of both VFF (#49, GP-side) and HSGP (#41, NN-side).

``boundary="neumann"`` and ``boundary="periodic"`` swap in the other two
classical boundary conditions (orthonormal on $[-L, L]$, eigenvalues of
$-d^2/dx^2$):

$$
\begin{aligned}
\text{Neumann:}\quad & \phi_0 = \tfrac{1}{\sqrt{2L}},\
  \phi_j = \tfrac{1}{\sqrt L}\cos\tfrac{j\pi(x+L)}{2L},\
  \lambda_j = \left(\tfrac{j\pi}{2L}\right)^2,\ j = 0, 1, \ldots \\
\text{Periodic:}\quad & \tfrac{1}{\sqrt{2L}},\
  \tfrac{1}{\sqrt L}\cos\tfrac{k\pi x}{L},\
  \tfrac{1}{\sqrt L}\sin\tfrac{k\pi x}{L},\
  \lambda = \left(\tfrac{k\pi}{L}\right)^2,\ k = 1, 2, \ldots
\end{aligned}
$$

The periodic columns are ordered constant, then ``cos 1, sin 1, cos 2, sin 2,
...``; every option lists its eigenvalues in non-decreasing order.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float


Boundary = Literal["dirichlet", "neumann", "periodic"]


def _check_boundary(boundary: str) -> None:
    if boundary not in ("dirichlet", "neumann", "periodic"):
        raise ValueError(
            f"boundary must be 'dirichlet', 'neumann' or 'periodic'; got {boundary!r}."
        )


def _frequencies_1d(
    num_basis: int, L: float, boundary: str, dtype
) -> Float[Array, " M"]:
    """Angular frequency of each 1D column (its eigenvalue is the square)."""
    if boundary == "dirichlet":
        return jnp.arange(1, num_basis + 1, dtype=dtype) * (jnp.pi / (2.0 * L))
    if boundary == "neumann":
        return jnp.arange(num_basis, dtype=dtype) * (jnp.pi / (2.0 * L))
    k = (jnp.arange(num_basis) + 1) // 2  # 0, 1, 1, 2, 2, ...
    return k.astype(dtype) * (jnp.pi / L)


def _basis_value_and_grad_1d(
    x: Float[Array, " N"], num_basis: int, L: float, boundary: str
) -> tuple[Float[Array, "N M"], Float[Array, "N M"]]:
    """1D eigenfunctions and their x-derivatives for any boundary condition."""
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}.")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}.")
    _check_boundary(boundary)
    freq = _frequencies_1d(num_basis, L, boundary, x.dtype)
    inv_sqrt_l = 1.0 / jnp.sqrt(L)
    if boundary == "periodic":
        arg = einx.dot("n, m -> n m", x, freq)
        is_cos = (jnp.arange(num_basis) % 2) == 1
        value = jnp.where(is_cos, jnp.cos(arg), jnp.sin(arg)) * inv_sqrt_l
        grad = (
            einx.multiply(
                "n m, m -> n m", jnp.where(is_cos, -jnp.sin(arg), jnp.cos(arg)), freq
            )
            * inv_sqrt_l
        )
        constant = jnp.arange(num_basis) == 0
    else:
        arg = einx.dot("n, m -> n m", x + L, freq)
        if boundary == "dirichlet":
            value = jnp.sin(arg) * inv_sqrt_l
            grad = einx.multiply("n m, m -> n m", jnp.cos(arg), freq) * inv_sqrt_l
        else:
            value = jnp.cos(arg) * inv_sqrt_l
            grad = einx.multiply("n m, m -> n m", -jnp.sin(arg), freq) * inv_sqrt_l
        constant = jnp.arange(num_basis) == (0 if boundary == "neumann" else -1)
    # The constant column (Neumann j = 0, periodic k = 0) is 1/√(2L).
    value = jnp.where(constant, 1.0 / jnp.sqrt(2.0 * L), value)
    grad = jnp.where(constant, 0.0, grad)
    return value, grad


def fourier_basis_1d(
    x: Float[Array, " N"],
    num_basis: int,
    L: float,
    *,
    boundary: Boundary = "dirichlet",
) -> Float[Array, "N M"]:
    r"""Evaluate the first ``num_basis`` 1D Laplacian eigenfunctions on ``[-L, L]``.

    $$
    \phi_j(x) = \frac{1}{\sqrt{L}} \sin\!\left(\frac{j\pi(x + L)}{2L}\right)
    $$


    Args:
        x: Inputs in ``[-L, L]``. Values outside the interval are
            extrapolated silently — callers wrap their own domain checks.
        num_basis: Number of basis functions ``M``.
        L: Half-width of the bounded domain (must be positive).
        boundary: ``"dirichlet"`` (default; the formula above),
            ``"neumann"`` or ``"periodic"`` (see the module docstring).

    Returns:
        ``(N, M)`` array whose ``(n, j-1)`` entry is $\phi_j(x_n)$ (for
        Dirichlet; column ``m`` is the ``m``-th eigenfunction in general).

    Raises:
        ValueError: If ``num_basis < 1`` or ``L <= 0``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import fourier_basis_1d
        >>> x = jnp.linspace(-1.0, 1.0, 5)  # (N=5,) in [-L, L]
        >>> fourier_basis_1d(x, num_basis=3, L=1.0).shape  # -> (N, M)
        (5, 3)
    """
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}.")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}.")
    _check_boundary(boundary)
    if boundary != "dirichlet":
        return _basis_value_and_grad_1d(x, num_basis, L, boundary)[0]
    j = jnp.arange(1, num_basis + 1, dtype=x.dtype)
    arg = einx.dot("n, m -> n m", x + L, j) * (jnp.pi / (2.0 * L))
    return jnp.sin(arg) / jnp.sqrt(L)


def fourier_eigenvalues_1d(
    num_basis: int,
    L: float,
    *,
    dtype: jnp.dtype = jnp.float32,
    boundary: Boundary = "dirichlet",
) -> Float[Array, " M"]:
    r"""Return $\lambda_j = (j\pi / (2L))^2$ for ``j = 1, ..., num_basis``.

    With ``boundary="neumann"`` / ``"periodic"`` the eigenvalues of those
    bases are returned instead (see the module docstring).

    Examples:
        >>> from geonnax._basis import fourier_eigenvalues_1d
        >>> fourier_eigenvalues_1d(num_basis=4, L=1.0).shape  # -> (M,)
        (4,)
    """
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}.")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}.")
    _check_boundary(boundary)
    if boundary != "dirichlet":
        return _frequencies_1d(num_basis, L, boundary, dtype) ** 2
    j = jnp.arange(1, num_basis + 1, dtype=dtype)
    return (j * jnp.pi / (2.0 * L)) ** 2


def _to_tuple(value: int | float | str | tuple, D: int, name: str) -> tuple:
    """Broadcast a scalar to a length-``D`` tuple, or validate an existing tuple."""
    if isinstance(value, tuple | list):
        out = tuple(value)
        if len(out) != D:
            raise ValueError(
                f"{name} must have length {D} (one per input dim); got {len(out)}."
            )
        return out
    return (value,) * D


def fourier_basis(
    x: Float[Array, "N D"],
    num_basis_per_dim: int | tuple[int, ...],
    L: float | tuple[float, ...],
    *,
    boundary: Boundary | tuple[Boundary, ...] = "dirichlet",
) -> tuple[Float[Array, "N M"], Float[Array, " M"]]:
    r"""Tensor-product Laplacian eigenpairs on $[-L, L]^D$ (Dirichlet by default).

    The $D$-dimensional eigenfunctions are products of 1D basis
    functions, with eigenvalues that *sum* across axes:

    $$
    \Phi_{(j_1, \ldots, j_D)}(x) = \prod_{d=1}^D \phi_{j_d}(x_d),
    \qquad
    \lambda_{(j_1, \ldots, j_D)} = \sum_{d=1}^D \lambda_{j_d}.
    $$


    The flattened index is row-major over the multi-index, i.e. the last
    dimension varies fastest. Total feature count is
    ``M = prod(num_basis_per_dim)``.

    Args:
        x: Inputs of shape ``(N, D)`` in $[-L, L]^D$.
        num_basis_per_dim: Per-axis number of 1D basis functions; an
            ``int`` is broadcast to all ``D`` axes.
        L: Per-axis half-width; an ``int``/``float`` is broadcast to all axes.
        boundary: Boundary condition, ``"dirichlet"``, ``"neumann"`` or
            ``"periodic"``; a single value applies to every axis, or pass one
            per axis (e.g. periodic in longitude, Dirichlet in latitude).

    Returns:
        ``(Phi, lam)`` with ``Phi`` of shape ``(N, M)`` and ``lam`` of
        shape ``(M,)``.

    Raises:
        ValueError: If ``num_basis_per_dim`` or ``L`` has wrong length.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import fourier_basis
        >>> x = jnp.zeros((4, 2))  # (N=4, D=2)
        >>> # M = prod(num_basis_per_dim) = 3 * 3 = 9
        >>> Phi, lam = fourier_basis(x, num_basis_per_dim=3, L=1.0)
        >>> (Phi.shape, lam.shape)
        ((4, 9), (9,))
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N, D); got shape {x.shape}.")
    D = x.shape[-1]
    M_per = _to_tuple(num_basis_per_dim, D, "num_basis_per_dim")
    L_per = _to_tuple(L, D, "L")
    B_per = _to_tuple(boundary, D, "boundary")

    # Per-axis bases and eigenvalues.
    bases = [
        fourier_basis_1d(x[:, d], M_per[d], float(L_per[d]), boundary=B_per[d])
        for d in range(D)
    ]  # each (N, M_d)
    lams = [
        fourier_eigenvalues_1d(
            M_per[d], float(L_per[d]), dtype=x.dtype, boundary=B_per[d]
        )
        for d in range(D)
    ]  # each (M_d,)

    # Tensor-product the bases (row-major flatten of the multi-index).
    Phi = bases[0]
    for b in bases[1:]:
        # Per-row outer product (n, a, b) then row-major flatten to (n, a·b).
        Phi = einx.multiply("n a, n b -> n a b", Phi, b)
        Phi = einx.id("n a b -> n (a b)", Phi)

    # Sum-of-eigenvalues across axes via pure JAX broadcasting — row-major
    # flatten matches the basis layout above.
    lam = _tensor_product_sum(lams, dtype=x.dtype)
    return Phi, lam


def _tensor_product_sum(
    lams: list[Float[Array, " M_d"]],
    *,
    dtype: jnp.dtype,
) -> Float[Array, " M"]:
    """Row-major flatten of the sum-of-eigenvalues tensor over D per-axis vectors.

    ``lams[d]`` has shape ``(M_d,)``; the result has shape ``(prod_d M_d,)``
    with entry ``(j_0, ..., j_{D-1})`` equal to ``sum_d lams[d][j_d]``. Uses
    pure JAX broadcasting — no Python-side enumeration of multi-indices.
    """
    if not lams:
        return jnp.zeros(0, dtype=dtype)
    D = len(lams)
    grid = jnp.zeros(tuple(lam.shape[0] for lam in lams), dtype=dtype)
    for d, lam_d in enumerate(lams):
        # Reshape lam_d to broadcast along axis d of the D-dim grid.
        shape = [1] * D
        shape[d] = lam_d.shape[0]
        grid = grid + jnp.reshape(lam_d, shape)
    return jnp.reshape(grid, (-1,))


def fourier_eigenvalues(
    num_basis_per_dim: int | tuple[int, ...],
    L: float | tuple[float, ...],
    D: int,
    *,
    dtype: jnp.dtype = jnp.float32,
    boundary: Boundary | tuple[Boundary, ...] = "dirichlet",
) -> Float[Array, " M"]:
    """Return the flattened sum-of-squares eigenvalues for a ``D``-dimensional box.

    Useful when only the eigenvalues are needed (e.g. building ``K_uu`` for
    inducing features without evaluating ``k_ux``).

    Examples:
        >>> from geonnax._basis import fourier_eigenvalues
        >>> # M = prod(num_basis_per_dim) = 3 * 3 = 9 for D=2
        >>> fourier_eigenvalues(num_basis_per_dim=3, L=1.0, D=2).shape
        (9,)
    """
    M_per = _to_tuple(num_basis_per_dim, D, "num_basis_per_dim")
    L_per = _to_tuple(L, D, "L")
    B_per = _to_tuple(boundary, D, "boundary")
    lams = [
        fourier_eigenvalues_1d(
            M_per[d], float(L_per[d]), dtype=dtype, boundary=B_per[d]
        )
        for d in range(D)
    ]
    return _tensor_product_sum(lams, dtype=dtype)
