"""Low-rank tensor factorizations for spectral-operator weights.

The learnable weights of a spectral convolution form a large tensor
``(in_channels, out_channels, *modes)`` (complex, hence a trailing real/imag
axis of size 2 here). Storing it densely dominates an FNO's parameter count,
so this module provides drop-in low-rank parameterisations — CP, Tucker, and
Tensor-Train — alongside the dense tensor. Each is a plain ``equinox.Module``
exposing `reconstruct`, which rebuilds the full real tensor; the spectral
layer then views the trailing axis as the real/imaginary parts.

References:
    Kossaifi et al. (2023), "Multi-Grid Tensorized Fourier Neural Operator".
    Kolda & Bader (2009), "Tensor Decompositions and Applications".
"""

from __future__ import annotations

import math
import string
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


Factorization = Literal["dense", "cp", "tucker", "tt"]


def _einsum_letters(n: int, *, skip: str = "") -> list[str]:
    """Return ``n`` distinct single-character einsum indices, avoiding ``skip``."""
    pool = [c for c in string.ascii_lowercase if c not in skip]
    if n > len(pool):
        raise ValueError(f"Cannot allocate {n} einsum indices.")
    return pool[:n]


def _resolve_rank(rank: float | int, full: int) -> int:
    """Resolve a fractional (``0 < rank <= 1``) or absolute rank against ``full``."""
    if isinstance(rank, float) and 0.0 < rank <= 1.0:
        return max(1, math.ceil(rank * full))
    r = int(rank)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {rank}.")
    return min(r, full)


class DenseTensor(eqx.Module):
    """A full, un-factorised tensor.

    Attributes:
        array: The stored tensor of the requested ``shape``.
    """

    array: Float[Array, "..."]

    @classmethod
    def init(
        cls, shape: tuple[int, ...], *, key: Array, scale: float = 1.0
    ) -> DenseTensor:
        """Construct a dense tensor with i.i.d. normal entries scaled by ``scale``."""
        return cls(array=scale * jax.random.normal(key, shape))

    def reconstruct(self) -> Float[Array, "..."]:
        return self.array


class CPTensor(eqx.Module):
    """Canonical-Polyadic (CP / PARAFAC) factorization.

    Represents ``T[i,j,...] = sum_r prod_k factor_k[index_k, r]``.

    Attributes:
        factors: One ``(dim_k, rank)`` factor per tensor mode.
        shape: The reconstructed tensor shape (static).
        rank: The CP rank (static).
    """

    factors: list[Float[Array, "dim rank"]]
    shape: tuple[int, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        shape: tuple[int, ...],
        *,
        key: Array,
        rank: float | int = 0.5,
        scale: float = 1.0,
    ) -> CPTensor:
        """Construct a CP tensor whose reconstruction has ~``scale`` element std."""
        r = _resolve_rank(rank, max(shape))
        n = len(shape)
        # Each reconstructed entry sums r products of n factor entries; choosing
        # per-factor std s gives reconstruction std ~ sqrt(r) * s**n.
        s = (scale / math.sqrt(r)) ** (1.0 / n)
        keys = jax.random.split(key, n)
        factors = [
            s * jax.random.normal(k, (d, r)) for d, k in zip(shape, keys, strict=True)
        ]
        return cls(factors=factors, shape=shape, rank=r)

    def reconstruct(self) -> Float[Array, "..."]:
        n = len(self.factors)
        modes = _einsum_letters(n, skip="z")
        lhs = ", ".join(f"{m}z" for m in modes)
        return jnp.einsum(f"{lhs} -> {''.join(modes)}", *self.factors)


class TuckerTensor(eqx.Module):
    """Tucker factorization: a small core contracted with per-mode factor matrices.

    Represents ``T = core ×_0 U_0 ×_1 U_1 ...`` with ``U_k`` of shape
    ``(dim_k, rank_k)``.

    Attributes:
        core: The core tensor of shape ``(rank_0, ..., rank_{n-1})``.
        factors: Per-mode factor matrices ``(dim_k, rank_k)``.
        shape: The reconstructed tensor shape (static).
    """

    core: Float[Array, "..."]
    factors: list[Float[Array, "dim rank"]]
    shape: tuple[int, ...] = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        shape: tuple[int, ...],
        *,
        key: Array,
        rank: float | int = 0.5,
        scale: float = 1.0,
    ) -> TuckerTensor:
        """Construct a Tucker tensor with orthonormal-ish factors."""
        ranks = tuple(_resolve_rank(rank, d) for d in shape)
        n = len(shape)
        keys = jax.random.split(key, n + 1)
        # Orthonormal columns keep the factor maps norm-preserving, so the core
        # std (scaled below) controls the reconstruction scale.
        factors = []
        for d, rk, k in zip(shape, ranks, keys[:-1], strict=True):
            mat = jax.random.normal(k, (d, rk))
            q, _ = jnp.linalg.qr(mat) if d >= rk else (mat, None)
            factors.append(q[:, :rk] if d >= rk else mat)
        core = scale * jax.random.normal(keys[-1], ranks)
        return cls(core=core, factors=factors, shape=shape)

    def reconstruct(self) -> Float[Array, "..."]:
        n = len(self.factors)
        core_idx = _einsum_letters(n, skip="")
        mode_idx = _einsum_letters(n, skip="".join(core_idx))
        operands = [self.core, *self.factors]
        terms = ["".join(core_idx)]
        terms += [f"{mode_idx[k]}{core_idx[k]}" for k in range(n)]
        out = "".join(mode_idx)
        return jnp.einsum(f"{', '.join(terms)} -> {out}", *operands)


class TTTensor(eqx.Module):
    """Tensor-Train (Matrix-Product) factorization.

    Represents ``T[i_0,...,i_{n-1}] = G_0[i_0] G_1[i_1] ... G_{n-1}[i_{n-1}]``
    with cores ``G_k`` of shape ``(r_k, dim_k, r_{k+1})`` and boundary ranks
    ``r_0 = r_n = 1``.

    Attributes:
        cores: The TT cores.
        shape: The reconstructed tensor shape (static).
        ranks: The full TT-rank sequence ``(1, r_1, ..., 1)`` (static).
    """

    cores: list[Float[Array, "r_left dim r_right"]]
    shape: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        shape: tuple[int, ...],
        *,
        key: Array,
        rank: float | int = 0.5,
        scale: float = 1.0,
    ) -> TTTensor:
        """Construct a TT tensor honouring the TT-rank upper bounds."""
        n = len(shape)
        # Maximal bond rank between split k|k+1 is min(prod left, prod right).
        bonds = [1]
        for k in range(1, n):
            left = math.prod(shape[:k])
            right = math.prod(shape[k:])
            bonds.append(_resolve_rank(rank, min(left, right)))
        bonds.append(1)
        ranks = tuple(bonds)
        keys = jax.random.split(key, n)
        # Per-core std chosen so the chained product keeps ~unit-ish scale.
        s = scale ** (1.0 / n)
        cores = [
            s
            * jax.random.normal(k, (ranks[i], shape[i], ranks[i + 1]))
            / math.sqrt(ranks[i])
            for i, k in enumerate(keys)
        ]
        return cls(cores=cores, shape=shape, ranks=ranks)

    def reconstruct(self) -> Float[Array, "..."]:
        # Sequentially contract the shared bond indices, accumulating mode axes.
        res = self.cores[0]  # (1, d0, r1)
        res = res.reshape(self.shape[0], self.ranks[1])
        for k in range(1, len(self.cores)):
            core = self.cores[k]  # (r_k, d_k, r_{k+1})
            r_left, d_k, r_right = core.shape
            res = res.reshape(-1, r_left) @ core.reshape(r_left, d_k * r_right)
            res = res.reshape(-1, r_right)
        return res.reshape(self.shape)


def init_factorized_tensor(
    shape: tuple[int, ...],
    factorization: Factorization,
    *,
    key: Array,
    rank: float | int = 0.5,
    scale: float = 1.0,
) -> DenseTensor | CPTensor | TuckerTensor | TTTensor:
    """Construct a factorized tensor of the requested kind.

    Args:
        shape: Target reconstructed shape.
        factorization: One of ``"dense"``, ``"cp"``, ``"tucker"``, ``"tt"``.
        key: PRNG key.
        rank: Fractional (``0 < rank <= 1``) or absolute rank for the low-rank
            forms; ignored for ``"dense"``.
        scale: Target element standard deviation of the reconstruction.

    Returns:
        The corresponding factorized-tensor module.

    Raises:
        ValueError: If ``factorization`` is unknown.

    Examples:
        All forms reconstruct the requested shape; the low-rank ones store
        far fewer parameters than dense:

        >>> import jax, jax.random as jr, equinox as eqx
        >>> from geonnax.layers import init_factorized_tensor
        >>> shape = (8, 8, 16, 16, 2)
        >>> def n_params(t):
        ...     leaves = jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))
        ...     return sum(x.size for x in leaves)
        >>> dense = init_factorized_tensor(shape, "dense", key=jr.PRNGKey(0))
        >>> cp = init_factorized_tensor(shape, "cp", key=jr.PRNGKey(0), rank=0.25)
        >>> dense.reconstruct().shape == cp.reconstruct().shape == shape
        True
        >>> n_params(cp) < n_params(dense)
        True
    """
    if factorization == "dense":
        return DenseTensor.init(shape, key=key, scale=scale)
    if factorization == "cp":
        return CPTensor.init(shape, key=key, rank=rank, scale=scale)
    if factorization == "tucker":
        return TuckerTensor.init(shape, key=key, rank=rank, scale=scale)
    if factorization == "tt":
        return TTTensor.init(shape, key=key, rank=rank, scale=scale)
    raise ValueError(
        f"factorization must be 'dense', 'cp', 'tucker', or 'tt', got "
        f"{factorization!r}."
    )


__all__ = [
    "CPTensor",
    "DenseTensor",
    "Factorization",
    "TTTensor",
    "TuckerTensor",
    "init_factorized_tensor",
]
