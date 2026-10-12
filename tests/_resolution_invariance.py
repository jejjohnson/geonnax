r"""Grid-resolution invariance harness (a test helper, not shipped).

A discretisation-invariant operator $G$ gives the same underlying function
whatever grid it is evaluated on. With a band-limited input $u$ sampled at a
coarse grid and ``up`` / ``down`` a band-limited resampling pair,

$$ \frac{\lVert G(u) - \mathrm{down}(G(\mathrm{up}(u)))\rVert}{\lVert G(u)\rVert}
\approx 0. $$

`resolution_gap` returns that ratio. Spectral (FFT zero-pad / truncate)
resampling is exact on band-limited periodic fields, so the gap isolates the
operator rather than the interpolation. This is the grid counterpart of
`tests/_point_invariance.py` (point clouds); `CNO` (#33) is asserted with the
same helper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import einx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


def spectral_resize(
    x: Float[Array, "C *spatial"], shape: Sequence[int]
) -> Float[Array, "C *spatial"]:
    """Resample a periodic field to ``shape`` by FFT zero-padding / truncation.

    Exact for fields band-limited below the smaller grid's Nyquist frequency.
    The Nyquist row of the smaller grid is dropped, so a field must stay
    strictly below it (``max_mode < min(n) / 2``).

    Args:
        x: Field of shape ``(C, *spatial)``.
        shape: Target spatial extents, one per spatial axis.

    Returns:
        The resampled field, shape ``(C, *shape)``, amplitude-preserving.
    """
    axes = tuple(range(1, x.ndim))
    xf = jnp.fft.fftshift(jnp.fft.fftn(x, axes=axes), axes=axes)
    out = xf
    for ax, (n, m) in enumerate(zip(x.shape[1:], shape, strict=True), start=1):
        if m > n:  # zero-pad the spectrum symmetrically
            lo = (m - n) // 2
            pad = [(0, 0)] * out.ndim
            pad[ax] = (lo, m - n - lo)
            out = jnp.pad(out, pad)
        elif m < n:  # keep the central m frequencies
            lo = (n - m) // 2
            out = jnp.take(out, jnp.arange(lo, lo + m), axis=ax)
    out = jnp.fft.ifftn(jnp.fft.ifftshift(out, axes=axes), axes=axes)
    scale = jnp.prod(jnp.asarray(shape)) / jnp.prod(jnp.asarray(x.shape[1:]))
    return out.real * scale


def band_limited_field(
    shape: Sequence[int], channels: int = 1, *, key: Array, max_mode: int = 3
) -> Float[Array, "C *spatial"]:
    """A smooth random periodic field on ``shape``, modes ``|k| <= max_mode``.

    A random-phase sum of sines/cosines per axis, so the field has no energy
    above ``max_mode`` and resampling it is exact.

    Returns:
        Array of shape ``(channels, *shape)`` with unit-order amplitude.
    """
    out = jnp.zeros((channels, *shape))
    for ax, n in enumerate(shape):
        k1, k2, key = jr.split(key, 3)
        coef = jr.normal(k1, (channels, max_mode))
        phase = jr.uniform(k2, (channels, max_mode), maxval=2 * jnp.pi)
        t = 2 * jnp.pi * jnp.arange(n) / n
        modes = jnp.arange(1, max_mode + 1)
        arg = einx.multiply("m, n -> m n", modes, t)
        wave = einx.sum(
            "c [m] n",
            einx.multiply(
                "c m, c m n -> c m n", coef, jnp.sin(arg[None] + phase[:, :, None])
            ),
        )
        shp = [1] * len(shape)
        shp[ax] = n
        out = out + wave.reshape(channels, *shp)
    return out


def resolution_gap(
    op: Callable[[Array], Array],
    x_lo: Float[Array, "C *spatial"],
    upsample: Callable[[Array], Array],
    downsample: Callable[[Array], Array],
) -> float:
    """Relative L2 gap between running ``op`` at low vs high resolution.

    Computes ``a = op(x_lo)`` and ``b = downsample(op(upsample(x_lo)))`` and
    returns ``||a - b|| / ||a||``; a discretisation-invariant operator gives
    a value near 0.

    Args:
        op: Single-example operator, ``(C, *spatial) -> (C', *spatial)``.
        x_lo: Band-limited input on the coarse grid.
        upsample: Coarse -> fine resampling (see `spectral_pair`).
        downsample: Fine -> coarse resampling, back to ``x_lo``'s grid.

    Returns:
        The relative gap as a Python float.
    """
    a = op(x_lo)
    b = downsample(op(upsample(x_lo)))
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))


def spectral_pair(
    lo: Sequence[int], factor: int = 2
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Spectral ``(upsample, downsample)`` pair between ``lo`` and ``factor * lo``."""
    hi = tuple(factor * n for n in lo)
    return (lambda x: spectral_resize(x, hi)), (lambda x: spectral_resize(x, tuple(lo)))
