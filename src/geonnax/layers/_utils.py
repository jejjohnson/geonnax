"""Shared helpers for the :mod:`geonnax.layers` building blocks."""

from __future__ import annotations

import math

import jax
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
