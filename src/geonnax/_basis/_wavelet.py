r"""Orthonormal discrete-wavelet-transform (DWT) basis matrices.

The columns of the returned matrix are an orthonormal multiscale wavelet basis:
$\Phi = W^\top$ where $W$ is the periodic DWT *analysis* operator built by
cascading per-level convolve-and-downsample steps. Because the transform is
orthonormal, synthesis is the transpose ($\Phi^\top \Phi = I$) and a field is
reconstructed exactly as $\Phi (\Phi^\top f)$.

This is the **critically sampled, non-redundant** counterpart to the
overcomplete `gabor_frame` / `gabor_frame_grid`: same multiscale idea, but an
orthonormal basis with an exact inverse rather than a redundant frame. Scope:
periodic (circular) boundaries, power-of-two sizes, and the Haar / Daubechies
``db2`` / ``db4`` orthonormal filters. The basis matrix is built once at
setup time (small, data-independent), like the graph-Laplacian eigenpairs.

References:
    Mallat (2008), *A Wavelet Tour of Signal Processing*.
    Daubechies (1992), *Ten Lectures on Wavelets*.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# Orthonormal scaling (low-pass) filters, ``sum h = sqrt(2)``, ``sum h^2 = 1``
# (PyWavelets ``dec_lo`` conventions).
_FILTERS: dict[str, tuple[float, ...]] = {
    "haar": (0.7071067811865476, 0.7071067811865476),
    "db2": (
        -0.12940952255092145,
        0.22414386804185735,
        0.836516303737469,
        0.48296291314469025,
    ),
    "db4": (
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ),
}


def _qmf(h: np.ndarray) -> np.ndarray:
    """High-pass quadrature-mirror filter ``g[k] = (-1)^k h[L-1-k]``."""
    k = np.arange(h.shape[0])
    return ((-1.0) ** k) * h[::-1]


def _level_matrix(m: int, h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """One periodic convolve-and-downsample level as an ``(m, m)`` matrix.

    Rows ``0:m//2`` are the low-pass (approximation) outputs, rows ``m//2:m``
    the high-pass (detail) outputs; circular boundaries make the block
    orthonormal for an orthonormal QMF pair.
    """
    half = m // 2
    length = h.shape[0]
    w = np.zeros((m, m))
    for i in range(half):
        for k in range(length):
            col = (2 * i + k) % m
            w[i, col] += h[k]
            w[half + i, col] += g[k]
    return w


def wavelet_basis_1d(
    n: int,
    *,
    wavelet: str = "haar",
    levels: int | None = None,
) -> Float[Array, "n n"]:
    r"""Orthonormal 1D DWT synthesis-basis matrix on a periodic length-``n`` grid.

    Args:
        n: Grid length; must be a power of two ``>= 2``.
        wavelet: ``"haar"``, ``"db2"``, or ``"db4"``.
        levels: Number of decomposition levels; defaults to the full cascade
            ``log2(n)``. Capped so the coarsest scale stays ``>= 2``.

    Returns:
        ``(n, n)`` array whose columns are the orthonormal wavelet basis
        functions (``Phi.T @ Phi == I``).

    Raises:
        ValueError: If ``n`` is not a power of two ``>= 2``, ``wavelet`` is
            unknown, or ``levels < 1``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import wavelet_basis_1d
        >>> Phi = wavelet_basis_1d(8, wavelet="db2")
        >>> bool(jnp.allclose(Phi.T @ Phi, jnp.eye(8), atol=1e-6))
        True
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of two >= 2, got {n}.")
    if wavelet not in _FILTERS:
        raise ValueError(f"unknown wavelet {wavelet!r}; choose from {list(_FILTERS)}.")
    max_levels = n.bit_length() - 1  # log2(n) exactly, since n is a power of two
    levels = max_levels if levels is None else levels
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}.")
    levels = min(levels, max_levels)

    h = np.asarray(_FILTERS[wavelet], dtype=np.float64)
    g = _qmf(h)
    w = np.eye(n)
    size = n
    for _ in range(levels):
        if size < 2:
            break
        level = np.eye(n)
        level[:size, :size] = _level_matrix(size, h, g)
        w = level @ w
        size //= 2
    # Orthonormal transform: synthesis basis is the transpose of analysis.
    return jnp.asarray(w.T)


def wavelet_basis_2d(
    ny: int,
    nx: int | None = None,
    *,
    wavelet: str = "haar",
    levels: int | None = None,
) -> Float[Array, "nynx nynx"]:
    r"""Separable orthonormal 2D DWT basis as a Kronecker product.

    The 2D basis is $\Phi_y \otimes \Phi_x$, so its rows index a row-major
    flattened ``(ny, nx)`` grid (row ``iy * nx + ix``) and it stays orthonormal.

    Args:
        ny: Number of rows; a power of two ``>= 2``.
        nx: Number of columns; defaults to ``ny``. A power of two ``>= 2``.
        wavelet: ``"haar"``, ``"db2"``, or ``"db4"``.
        levels: Decomposition levels per axis; defaults to the full cascade.

    Returns:
        ``(ny * nx, ny * nx)`` orthonormal basis matrix.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import wavelet_basis_2d
        >>> Phi = wavelet_basis_2d(4, 4, wavelet="haar")
        >>> Phi.shape
        (16, 16)
    """
    nx = ny if nx is None else nx
    phi_y = wavelet_basis_1d(ny, wavelet=wavelet, levels=levels)
    phi_x = wavelet_basis_1d(nx, wavelet=wavelet, levels=levels)
    return jnp.kron(phi_y, phi_x)


__all__ = ["wavelet_basis_1d", "wavelet_basis_2d"]
