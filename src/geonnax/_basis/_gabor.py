r"""Fixed overcomplete multiscale Gabor frame.

A Gabor atom multiplies a Gaussian envelope by an oscillation,

$$
\varphi_a(x) = e^{-\lVert x - c_a \rVert^2 / (2 L_a^2)} \cos(\omega_a),
$$

with phase $\omega_a = k_a \lVert x - c_a \rVert$ (radial / isotropic) or
$\omega_a = k_a\, \theta_a^\top (x - c_a)$ (directional, unit orientation
$\theta_a$). A dyadic bank lays atoms across scales $L_s = L_0 2^s$ and
wavenumbers $k_s = 2\pi / L_s$ on a grid of locations per scale, giving a
*redundant frame* with more atoms than evaluation points — the multiscale
workhorse behind sea-surface-height mapping (Ubelmann et al.; Ardhuin et al.).

This is the redundant fixed-dictionary form, distinct from the trained network
form in ``geonnax.mfn`` and the orthonormal critically-sampled DWT in
``geonnax.layers``. A frame has no eigenvalues and no inverse; geonnax returns
only synthesis ($\Phi$) and the per-atom geometry (scales, wavenumbers) a
consumer needs to impose a spectral law such as $\sigma_a^2 \propto k_a^{-\alpha}$.
Recovering coefficients from a field is a regulariser-dependent least-squares /
frame-dual problem and stays downstream.

References:
    Mallat (2008), *A Wavelet Tour of Signal Processing*.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


# Softening for the orientation unit-normalisation (degenerate zero vectors).
_EPS = 1e-12


def _safe_sqrt(sq: Float[Array, "..."]) -> Float[Array, "..."]:
    """sqrt that is exact (0) at ``sq == 0`` with finite gradients there.

    The double ``where`` keeps the backward pass off the ``sqrt`` singularity at
    0 without shifting the forward value (unlike adding a fixed epsilon).
    """
    positive = sq > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)


def gabor_frame(
    x: Float[Array, "N d"],
    centers: Float[Array, "M d"],
    scales: Float[Array, " M"],
    wavenumbers: Float[Array, " M"],
    *,
    orientations: Float[Array, "M d"] | None = None,
) -> Float[Array, "N M"]:
    r"""Evaluate a bank of Gabor atoms (Gaussian envelope times oscillation).

    Radial (isotropic) atoms when ``orientations`` is ``None``; directional
    otherwise (the orientations are normalised to unit length internally).
    Returns only $\Phi$; the per-atom ``scales`` / ``wavenumbers`` the caller
    passed in are the metadata a consumer weights by.

    Args:
        x: Evaluation points, shape ``(N, d)``.
        centers: Atom centres, shape ``(M, d)``.
        scales: Gaussian envelope width $L_a$ per atom, shape ``(M,)``.
        wavenumbers: Oscillation wavenumber $k_a$ per atom, shape ``(M,)``.
        orientations: Optional unit orientations $\theta_a$, shape ``(M, d)``,
            for directional atoms; ``None`` gives radial atoms.

    Returns:
        Basis matrix of shape ``(N, M)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import gabor_frame
        >>> x = jnp.zeros((4, 2))
        >>> c = jnp.ones((6, 2))
        >>> gabor_frame(x, c, jnp.ones(6), jnp.ones(6)).shape
        (4, 6)
    """
    diff = einx.subtract("n d, m d -> n m d", x, centers)  # (N, M, d)
    sq = einx.sum("n m d -> n m", diff**2)  # squared distance
    env = jnp.exp(-0.5 * einx.divide("n m, m -> n m", sq, scales**2))
    if orientations is None:
        r = _safe_sqrt(sq)
        phase = einx.multiply("n m, m -> n m", r, wavenumbers)  # radial
    else:
        unit = orientations / (
            jnp.linalg.norm(orientations, axis=-1, keepdims=True) + _EPS
        )
        proj = einx.dot("n m d, m d -> n m", diff, unit)  # directional
        phase = einx.multiply("n m, m -> n m", proj, wavenumbers)
    return env * jnp.cos(phase)


def gabor_frame_grid(
    x: Float[Array, "N d"],
    bounds: Float[Array, "d 2"],
    *,
    n_scales: int,
    base_scale: float,
    oversample: float = 1.0,
) -> tuple[
    Float[Array, "N M"],
    Float[Array, "M d"],
    Float[Array, " M"],
    Float[Array, " M"],
]:
    r"""Build a dyadic radial-Gabor frame over a bounded box.

    Scales $L_s = \texttt{base\_scale} \cdot 2^s$ for $s = 0, \dots,
    n_\text{scales} - 1$; per scale, centres on a regular grid spaced by
    $L_s / \texttt{oversample}$ over ``bounds``; wavenumbers $k_s = 2\pi / L_s$.
    The centre layout is built statically (the number of scales is small and the
    layout is data-independent), so the only thing on the hot path is the single
    `gabor_frame` kernel.

    Args:
        x: Evaluation points, shape ``(N, d)``.
        bounds: Per-axis ``[lo, hi]`` box, shape ``(d, 2)``. A **build-time
            concrete** value (read in Python to lay out the grid), not a traced
            array.
        n_scales: Number of dyadic scales.
        base_scale: Finest scale $L_0$.
        oversample: Grid density per scale; spacing is ``L_s / oversample``
            (``> 1`` packs centres tighter than one envelope width).

    Returns:
        ``(Phi, centers, scales, wavenumbers)`` — ``Phi`` of shape ``(N, M)``
        and the per-atom geometry of shape ``(M, d)`` / ``(M,)`` / ``(M,)``, so
        the consumer can both synthesise ``Phi @ w`` and weight atoms by scale.

    Raises:
        ValueError: If ``n_scales < 1``, ``base_scale <= 0``, ``oversample <= 0``,
            or ``bounds`` is malformed.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax._basis import gabor_frame_grid
        >>> x = jnp.zeros((10, 2))
        >>> bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
        >>> Phi, c, L, k = gabor_frame_grid(x, bounds, n_scales=2, base_scale=0.25)
        >>> Phi.shape[0], c.shape[1], L.shape == k.shape
        (10, 2, True)
    """
    if n_scales < 1:
        raise ValueError(f"n_scales must be >= 1, got {n_scales}.")
    if base_scale <= 0:
        raise ValueError(f"base_scale must be > 0, got {base_scale}.")
    if oversample <= 0:
        raise ValueError(f"oversample must be > 0, got {oversample}.")
    bounds_np = np.asarray(bounds, dtype=np.float64)
    if bounds_np.ndim != 2 or bounds_np.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds_np.shape}.")
    if np.any(bounds_np[:, 1] <= bounds_np[:, 0]):
        raise ValueError("each bounds row must satisfy lo < hi.")

    d = bounds_np.shape[0]
    centers_blocks, scale_blocks, wavenumber_blocks = [], [], []
    for s in range(n_scales):
        length = base_scale * 2.0**s
        spacing = length / oversample
        # Floor the step count so every centre stays inside [lo, hi] even when
        # the spacing does not divide the interval (the +1e-9 catches an exact
        # endpoint that floating-point rounds just under).
        axes = [
            lo + spacing * np.arange(int(np.floor((hi - lo) / spacing + 1e-9)) + 1)
            for lo, hi in bounds_np
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
        centers_blocks.append(grid)
        scale_blocks.append(np.full(grid.shape[0], length))
        wavenumber_blocks.append(np.full(grid.shape[0], 2.0 * np.pi / length))

    centers = jnp.asarray(np.concatenate(centers_blocks), dtype=x.dtype)
    scales = jnp.asarray(np.concatenate(scale_blocks), dtype=x.dtype)
    wavenumbers = jnp.asarray(np.concatenate(wavenumber_blocks), dtype=x.dtype)
    phi = gabor_frame(x, centers, scales, wavenumbers)
    return phi, centers, scales, wavenumbers


__all__ = ["gabor_frame", "gabor_frame_grid"]
