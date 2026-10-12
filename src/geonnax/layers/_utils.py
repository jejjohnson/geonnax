"""Shared helpers for the `geonnax.layers` building blocks."""

from __future__ import annotations

import itertools
import math

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array


def act(x: Array) -> Array:
    """SiLU / swish activation, the default nonlinearity across the layers."""
    return jax.nn.silu(x)


def group_count(channels: int, groups: int = 8) -> int:
    """Largest divisor of ``channels`` not exceeding ``groups`` (via gcd).

    ``equinox.nn.GroupNorm`` requires ``channels % groups == 0``; clamping with
    ``gcd`` yields a valid group count for any channel width without raising.
    """
    return math.gcd(channels, groups)


def shuffle_patterns(num_spatial_dims: int) -> tuple[str, str]:
    """Return ``(space->depth, depth->space)`` einx patterns for factor-2 shuffle.

    ``space->depth`` folds a size-``2`` patch from each spatial axis into the
    channel axis (channels ``×2**d``, each spatial extent halved);
    ``depth->space`` is its exact inverse.
    """
    axes = " ".join(f"a{i}" for i in range(num_spatial_dims))
    grouped = " ".join(f"(a{i} p{i})" for i in range(num_spatial_dims))
    patches = " ".join(f"p{i}" for i in range(num_spatial_dims))
    down = f"c {grouped} -> (c {patches}) {axes}"
    up = f"(c {patches}) {axes} -> c {grouped}"
    return down, up


def shuffle_kwargs(num_spatial_dims: int) -> dict[str, int]:
    """Patch-size keyword args (all ``2``) for the shuffle einx patterns."""
    return {f"p{i}": 2 for i in range(num_spatial_dims)}


def pointwise(
    num_spatial_dims: int, in_channels: int, out_channels: int, *, key: Array
) -> eqx.nn.Conv:
    """A ``1x1`` (pointwise) convolution — a per-position channel-wise linear map."""
    return eqx.nn.Conv(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        key=key,
    )


# --- rfft corner layout (shared by SpectralConv and the non-uniform transforms) ---


def _corner_slices(n_modes: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Slices selecting the ``2**(d-1)`` low-frequency corners of an rfft spectrum.

    Every non-final axis keeps both its lowest positive (``[0:m]``) and lowest
    negative (``[-m:]``) frequencies; the final (rfft-halved) axis keeps only
    ``[0:m]``. The Cartesian product over axes enumerates the corner blocks.
    """
    d = len(n_modes)
    per_axis = [[slice(0, m), slice(-m, None)] for m in n_modes[:-1]]
    combos = itertools.product(*per_axis) if d > 1 else [()]
    last = slice(0, n_modes[-1])
    return [(*combo, last) for combo in combos]


def _corner_freqs(n_modes: tuple[int, ...]) -> list[tuple[np.ndarray, ...]]:
    """Integer wavenumbers of each corner block, in `_corner_slices` order.

    Corner ``c`` holds one integer array per axis: ``0..m-1`` or ``-m..-1`` on a
    non-final axis, ``0..m-1`` on the final one. These are exactly the
    frequencies that ``_corner_slices(n_modes)[c]`` selects from an rfft
    spectrum.
    """
    d = len(n_modes)
    per_axis = [[np.arange(m), np.arange(-m, 0)] for m in n_modes[:-1]]
    combos = itertools.product(*per_axis) if d > 1 else [()]
    last = np.arange(n_modes[-1])
    return [(*combo, last) for combo in combos]
