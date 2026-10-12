"""Non-uniform Fourier transforms between point sets and a truncated rfft spectrum.

`nonuniform_rfft` computes the low Fourier modes of a field from its values at
arbitrary points of the latent torus $[0, 1)^d$; `nonuniform_irfft` evaluates
a truncated real Fourier series at arbitrary points. They are the
geometry-facing halves of Geo-FNO (Li, Huang, Liu and Anandkumar, *Fourier
Neural Operator with Learned Deformations for PDEs on General Geometries*,
JMLR 24, 2023), with one change: the forward transform takes **quadrature
weights** $w_n$, so it approximates the continuous Fourier coefficient and
converges as the number of points grows (the reference implementation sums
unweighted, so its output scales with the point count).

The spectrum uses the rfft corner layout of `geonnax.layers.SpectralConv`
(`_utils._corner_slices`): $2^{d-1}$ blocks of shape ``(C, *n_modes)``. On a
uniform grid of $S$ nodes with $w_n = 1/S$ the coefficients equal
``rfftn(u) / S`` on the retained modes.
"""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from geonnax.layers._utils import _corner_freqs


def _modes_spec(d: int) -> str:
    """einx axis names ``"m0 m1 …"`` for the ``d`` mode axes."""
    return " ".join(f"m{a}" for a in range(d))


def _validate_modes(n_modes: tuple[int, ...]) -> tuple[int, ...]:
    if len(n_modes) == 0:
        raise ValueError("n_modes must be non-empty.")
    if any(int(m) < 1 for m in n_modes):
        raise ValueError(f"all n_modes must be >= 1, got {n_modes}.")
    return tuple(int(m) for m in n_modes)


def _validate_points(xi: Array, n_modes: tuple[int, ...]) -> None:
    if xi.ndim != 2 or xi.shape[-1] != len(n_modes):
        raise ValueError(
            f"xi must have shape (N, {len(n_modes)}) to match n_modes={n_modes}, "
            f"got {xi.shape}."
        )


def _corner_phases(
    xi: Float[Array, "N d"], n_modes: tuple[int, ...]
) -> list[Float[Array, "N *n_modes"]]:
    """Phases $2\\pi\\,k\\cdot\\xi_n$ of every point against each corner's modes.

    $k\\cdot\\xi$ is reduced to $[-\\tfrac12, \\tfrac12]$ before the $2\\pi$
    scaling, which keeps float32 arguments small (the gradient is unchanged).
    """
    d = len(n_modes)
    modes = _modes_spec(d)
    two_pi = jnp.asarray(2.0 * math.pi, dtype=xi.dtype)
    phases = []
    for freqs in _corner_freqs(n_modes):
        table = np.stack(np.meshgrid(*freqs, indexing="ij"), axis=-1)  # (*m, d)
        k = jnp.asarray(table, dtype=xi.dtype)
        theta = einx.dot(f"n d, {modes} d -> n {modes}", xi, k)
        phases.append(two_pi * (theta - jax.lax.stop_gradient(jnp.round(theta))))
    return phases


def nonuniform_rfft(
    u: Float[Array, "C N"],
    xi: Float[Array, "N d"],
    weights: Float[Array, " N"],
    n_modes: tuple[int, ...],
) -> list[Complex[Array, "C *n_modes"]]:
    r"""Truncated Fourier coefficients of a field sampled at points of $[0, 1)^d$.

    A quadrature of the continuous Fourier coefficient on the unit torus, for
    every retained wavenumber $k \in \mathcal K$ (the rfft corner layout):

    $$\hat c_k = \sum_{n=1}^{N} w_n\, u_n\, e^{-2\pi i\, k\cdot\xi_n}
    \approx \int_{[0,1)^d} u(\xi)\, e^{-2\pi i\, k\cdot\xi}\, d\xi .$$

    The weights are used as given (never normalised): they should satisfy
    $w_n \ge 0$ and $\sum_n w_n \approx 1$ (cell areas / volumes of a mesh,
    or $1/N$ for a stratified cloud). A point with $w_n = 0$ contributes
    nothing, so ragged clouds can be zero-padded to a common $N$ and
    ``jax.vmap``-ed; padded coordinates must still be finite. On a uniform
    grid of $S$ nodes with $w_n = 1/S$ the result equals ``rfftn(u) / S`` on
    the retained modes.

    Cost: $O(N \cdot |\mathcal K| \cdot C)$ multiply-adds and an
    $N \times |\mathcal K|$ phase table per call, with
    $|\mathcal K| = 2^{d-1} \prod_a m_a$ (about 46 MB in float32 at
    $N = 20\,000$ and $12 \times 12$ modes). For very large clouds a
    non-uniform FFT such as ``jax-finufft`` computes the same sums in
    $O(N + |\mathcal K| \log |\mathcal K|)$; it is an optional, user-side
    substitute, not a geonnax dependency.

    Args:
        u: Field values, channel-first, shape ``(C, N)``.
        xi: Latent coordinates in $[0, 1)^d$ of the ``N`` points, shape
            ``(N, d)`` with ``d = len(n_modes)``.
        weights: Quadrature weights, shape ``(N,)``.
        n_modes: Retained modes per axis (as in `SpectralConv`).

    Returns:
        The $2^{d-1}$ corner blocks, each of shape ``(C, *n_modes)``, in the
        order of `_utils._corner_slices(n_modes)`.

    Raises:
        ValueError: If ``n_modes`` is empty or non-positive, or the shapes of
            ``u``, ``xi`` and ``weights`` disagree.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import nonuniform_rfft
        >>> xi = jr.uniform(jr.PRNGKey(0), (500, 2))
        >>> u = jnp.sin(2 * jnp.pi * xi[:, 0])[None]            # (C=1, N)
        >>> w = jnp.full((500,), 1 / 500)
        >>> blocks = nonuniform_rfft(u, xi, w, (4, 4))
        >>> len(blocks), blocks[0].shape
        (2, (1, 4, 4))
    """
    n_modes = _validate_modes(n_modes)
    _validate_points(xi, n_modes)
    n = xi.shape[0]
    if u.ndim != 2 or u.shape[-1] != n:
        raise ValueError(f"u must have shape (C, {n}) to match xi, got {u.shape}.")
    if weights.shape != (n,):
        raise ValueError(
            f"weights must have shape ({n},) to match xi, got {weights.shape}."
        )
    modes = _modes_spec(len(n_modes))
    uw = einx.multiply("c n, n -> c n", u, weights)
    blocks = []
    for phase in _corner_phases(xi, n_modes):
        re = einx.dot(f"c n, n {modes} -> c {modes}", uw, jnp.cos(phase))
        im = einx.dot(f"c n, n {modes} -> c {modes}", uw, jnp.sin(phase))
        blocks.append(jax.lax.complex(re, -im))
    return blocks


def nonuniform_irfft(
    coeffs: list[Complex[Array, "C *n_modes"]],
    xi: Float[Array, "M d"],
    n_modes: tuple[int, ...],
) -> Float[Array, "C M"]:
    r"""Evaluate a truncated real Fourier series at points of $[0, 1)^d$.

    The inverse of `nonuniform_rfft` on band-limited fields, with the
    Hermitian half-spectrum handled as ``irfftn`` does:

    $$u(\xi) = \sum_{k \in \mathcal K} \mu_k\,
    \mathrm{Re}\!\left[\hat c_k\, e^{2\pi i\, k\cdot\xi}\right],
    \qquad \mu_k = \begin{cases} 1 & k_d = 0 \\ 2 & k_d > 0 \end{cases}$$

    Taking the real part also discards the imaginary part of the $k_d = 0$
    plane, as ``irfftn`` does; no Nyquist term arises because the modes stay
    below half of any grid. On grid nodes $\xi = j/s$ this equals
    ``irfftn`` of the corners scattered into a spectrum and scaled by $S$.

    Cost: $O(M \cdot |\mathcal K| \cdot C)$ multiply-adds and an
    $M \times |\mathcal K|$ phase table, as for `nonuniform_rfft`
    (``jax-finufft`` is the optional user-side fast path).

    Args:
        coeffs: The $2^{d-1}$ corner blocks, each of shape ``(C, *n_modes)``,
            in the order of `_utils._corner_slices(n_modes)`.
        xi: Query coordinates in $[0, 1)^d$, shape ``(M, d)``.
        n_modes: Retained modes per axis.

    Returns:
        The real field at the query points, shape ``(C, M)``.

    Raises:
        ValueError: If ``n_modes`` is empty or non-positive, ``xi`` has the
            wrong number of columns, or ``coeffs`` has the wrong number or
            shape of blocks.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import nonuniform_irfft, nonuniform_rfft
        >>> s = 16
        >>> g = jnp.arange(s) / s
        >>> nodes = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        >>> u = jnp.cos(2 * jnp.pi * (nodes[:, 0] + 2 * nodes[:, 1]))[None]
        >>> blocks = nonuniform_rfft(u, nodes, jnp.full((s * s,), 1 / s**2), (4, 4))
        >>> query = jr.uniform(jr.PRNGKey(0), (300, 2))
        >>> v = nonuniform_irfft(blocks, query, (4, 4))
        >>> v.shape
        (1, 300)
        >>> exact = jnp.cos(2 * jnp.pi * (query[:, 0] + 2 * query[:, 1]))
        >>> bool(jnp.allclose(v[0], exact, atol=1e-5))
        True
    """
    n_modes = _validate_modes(n_modes)
    _validate_points(xi, n_modes)
    d = len(n_modes)
    if len(coeffs) != 2 ** (d - 1):
        raise ValueError(
            f"coeffs must hold 2**(d-1) = {2 ** (d - 1)} corner blocks for "
            f"n_modes={n_modes}, got {len(coeffs)}."
        )
    for c in coeffs:
        if c.ndim != d + 1 or tuple(c.shape[1:]) != n_modes:
            raise ValueError(
                f"each corner block must have shape (C, *{n_modes}), got {c.shape}."
            )
    modes = _modes_spec(d)
    last = f"m{d - 1}"
    mu = np.full(n_modes[-1], 2.0)
    mu[0] = 1.0
    total = jnp.zeros((coeffs[0].shape[0], xi.shape[0]), dtype=coeffs[0].real.dtype)
    for c, phase in zip(coeffs, _corner_phases(xi, n_modes), strict=True):
        mu_c = jnp.asarray(mu, dtype=c.real.dtype)
        re = einx.multiply(f"c {modes}, {last} -> c {modes}", c.real, mu_c)
        im = einx.multiply(f"c {modes}, {last} -> c {modes}", c.imag, mu_c)
        # Re[(a + ib)(cos + i sin)] = a cos - b sin
        part = einx.dot(f"c {modes}, n {modes} -> c n", re, jnp.cos(phase))
        part = part - einx.dot(f"c {modes}, n {modes} -> c n", im, jnp.sin(phase))
        total = total + part
    return total
