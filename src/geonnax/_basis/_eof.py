r"""Empirical-orthogonal-function (EOF / PCA) basis from a data matrix.

EOFs are the dominant spatial patterns of a dataset: the right singular vectors
of a ``(samples, grid)`` matrix, i.e. the eigenvectors of the spatial
covariance. Unlike the closed-form eigenbases elsewhere in this package, an EOF
basis is *data-driven* — its columns adapt to the covariance actually present in
the data (the DINEOF construction for ocean gap-filling; Beckers & Rixen 2003).

The "half a prior needs" is the singular-value spectrum: with a length-``T``
sample axis, mode ``a`` carries variance ``sigma_a^2 / (T - 1)``, so the prior
std is ``sigma_a / sqrt(T - 1)``. geonnax returns the singular values; the
variance convention stays with the consumer.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def eof_basis(
    data: Float[Array, "T N"],
    n_modes: int,
    *,
    center: bool = True,
) -> tuple[Float[Array, "N M"], Float[Array, " M"]]:
    r"""Leading ``n_modes`` empirical orthogonal functions of a data matrix.

    Computes the economy SVD ``X = U S V^\top`` of the (optionally
    sample-mean-centred) data ``X`` of shape ``(T, N)`` and returns the leading
    spatial patterns — the columns of ``V``, shape ``(N, M)`` — with their
    singular values. The columns are orthonormal; the singular values give the
    prior variance per mode, ``sigma_a^2 / (T - 1)``.

    Args:
        data: Data matrix of shape ``(T, N)`` — ``T`` samples (e.g. time
            snapshots) over ``N`` grid points.
        n_modes: Number of leading EOFs ``M`` to keep.
        center: If ``True`` (default), subtract the per-grid-point sample mean
            before the SVD, so the EOFs describe anomalies about the mean.

    Returns:
        ``(Phi, singular_values)`` with ``Phi`` of shape ``(N, M)`` (orthonormal
        columns) and ``singular_values`` of shape ``(M,)``, sorted descending.

    Raises:
        ValueError: If ``data`` is not 2D or ``n_modes`` is outside
            ``[1, min(T, N)]``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import eof_basis
        >>> X = jnp.arange(24.0).reshape(4, 6)  # (T=4, N=6)
        >>> Phi, s = eof_basis(X, n_modes=2)
        >>> (Phi.shape, s.shape)
        ((6, 2), (2,))
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (T, N); got shape {data.shape}.")
    n_samples, n_grid = data.shape
    max_modes = min(n_samples, n_grid)
    if n_modes < 1 or n_modes > max_modes:
        raise ValueError(f"n_modes must be in [1, {max_modes}]; got {n_modes}.")
    if center:
        data = data - jnp.mean(data, axis=0, keepdims=True)
    # Economy SVD: U (T, k), s (k,), vt (k, N) with k = min(T, N). The right
    # singular vectors (rows of vt) are the spatial EOFs.
    _, s, vt = jnp.linalg.svd(data, full_matrices=False)
    return vt[:n_modes].T, s[:n_modes]


__all__ = ["eof_basis"]
