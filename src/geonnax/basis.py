"""Pure-JAX feature helpers for Bayesian-Neural-Field-style models.

All functions are pure, stateless, and take only JAX arrays — no
``equinox.Module``, no NumPyro sites, no pandas. ``equinox.Module``
wrappers built atop these helpers (``Standardization``,
``FourierFeatures``, ``SeasonalFeatures``, ``InteractionFeatures``)
live with the consuming library.

Provides:

* `fourier_features` — cos/sin basis at frequencies $2\\pi
  \\cdot 2^d$ for $d = 0, \\dots, D-1$.
* `seasonal_frequencies` — flatten ``(periods, harmonics)`` pairs
  into a 1D frequency list.
* `seasonal_features` — cos/sin basis at multiples of
  $2\\pi / \\tau_p$ for each period $\\tau_p$.
* `interaction_features` — element-wise products on selected
  pairs of input columns.
* `standardize` / `unstandardize` — affine transform with a
  precomputed mean and std.

Implementation uses `einx` (``einx.id`` for broadcasts/reshapes,
``einx.prod`` for axis reductions) for any non-trivial reshaping, per the
project convention.
"""

from __future__ import annotations

from collections.abc import Sequence

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def fourier_features(
    x: Float[Array, " N"],
    max_degree: int,
    *,
    rescale: bool = False,
) -> Float[Array, "N two_max_degree"]:
    r"""Cos/sin Fourier basis at dyadic frequencies.

    For each input element and each degree $d \in \{0, \dots,
    D-1\}$, evaluates

    $$
    \phi_{d, \cos}(x) = \cos(2\pi \cdot 2^d \cdot x), \qquad
    \phi_{d, \sin}(x) = \sin(2\pi \cdot 2^d \cdot x).
    $$


    Returns the columns concatenated as ``[cos_0, ..., cos_{D-1},
    sin_0, ..., sin_{D-1}]``, matching Google's bayesnf layout.

    Args:
        x: Length-``N`` input vector.
        max_degree: Number of dyadic frequencies ``D``. Output has
            ``2 * max_degree`` columns.
        rescale: If ``True``, divide each ``(cos_d, sin_d)`` pair by
            ``d + 1`` to bias the prior toward lower-frequency basis
            functions.

    Returns:
        Array of shape ``(N, 2 * max_degree)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import fourier_features
        >>> fourier_features(jnp.linspace(0.0, 1.0, 5), max_degree=3).shape
        (5, 6)
    """
    degrees = jnp.arange(max_degree)
    # Broadcast x to (N, D) frequencies without an explicit reshape.
    z = einx.id("n -> n d", x, d=max_degree) * (2.0 * jnp.pi * 2.0**degrees)
    feats = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
    if rescale:
        denom = jnp.concatenate([degrees + 1, degrees + 1])
        feats = feats / denom
    return feats


def seasonal_frequencies(
    periods: Sequence[float],
    harmonics: Sequence[int],
) -> tuple[list[int], list[float]]:
    r"""Flatten ``(period, harmonic_count)`` pairs into Python lists.

    For each period $\tau_p$ with $H_p$ harmonics, emits
    frequencies $f_{p, h} = h / \tau_p$ for $h = 1, \dots,
    H_p$. The total length is $F = \sum_p H_p$.

    Inputs are **Python sequences**, not JAX arrays, so this helper
    runs at trace time and never triggers a concretization error under
    ``jax.jit``. Most callers won't use it directly; it's exposed for
    symmetry with `seasonal_features`.

    Args:
        periods: Period values.
        harmonics: Number of harmonics per period.

    Returns:
        ``(period_index, frequency)``: two Python lists of length
        $F = \sum_p H_p$.

    Examples:
        >>> from geonnax.basis import seasonal_frequencies
        >>> # periods (7, 365) with (1, 2) harmonics -> F = 1 + 2 = 3 freqs
        >>> idx, freqs = seasonal_frequencies([7.0, 365.0], [1, 2])
        >>> idx
        [0, 1, 1]
        >>> len(freqs)
        3
    """
    period_index: list[int] = []
    freqs: list[float] = []
    for p_idx, (period, n_h) in enumerate(zip(periods, harmonics, strict=True)):
        for h in range(1, int(n_h) + 1):
            period_index.append(p_idx)
            freqs.append(float(h) / float(period))
    return period_index, freqs


def seasonal_features(
    x: Float[Array, " N"],
    periods: Sequence[float],
    harmonics: Sequence[int],
    *,
    rescale: bool = False,
) -> Float[Array, "N two_F"]:
    r"""Cos/sin features at multiples of $2\pi / \tau_p$.

    For each period $\tau_p$ with $H_p$ harmonics, evaluates

    $$
    \phi_{p, h, \cos}(x) = \cos(2\pi h x / \tau_p), \qquad
    \phi_{p, h, \sin}(x) = \sin(2\pi h x / \tau_p),
    $$


    for $h = 1, \dots, H_p$. Returns the cos columns concatenated
    with the sin columns, length $F = \sum_p H_p$ each.

    ``periods`` and ``harmonics`` are **Python sequences** (tuples,
    lists, or 0-d JAX arrays wrapped at the call site). Keeping them as
    Python values lets the function run cleanly under ``jax.jit`` and
    ``lax.scan`` without triggering a concretization error.

    Args:
        x: Time/index input, shape ``(N,)``.
        periods: Period values.
        harmonics: Harmonics per period.
        rescale: If ``True``, divide each ``(cos, sin)`` pair by its
            within-period harmonic index, biasing the prior toward
            longer-wavelength modes within each period.

    Returns:
        Array of shape ``(N, 2 * F)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import seasonal_features
        >>> x = jnp.linspace(0.0, 10.0, 4)
        >>> # F = 1 + 2 = 3 frequencies -> 2 * F = 6 columns
        >>> seasonal_features(x, [7.0, 365.0], [1, 2]).shape
        (4, 6)
    """
    _, freq_list = seasonal_frequencies(periods, harmonics)
    if not freq_list:
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    freqs = jnp.asarray(freq_list, dtype=jnp.float32)
    z = einx.id("n -> n f", x, f=freqs.shape[0]) * (2.0 * jnp.pi * freqs)
    feats = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
    if rescale:
        # Rescale by within-period harmonic index (1, 2, ..., H_p).
        h_within_list: list[float] = []
        for n_h in harmonics:
            h_within_list.extend(range(1, int(n_h) + 1))
        h_within = jnp.asarray(h_within_list, dtype=jnp.float32)
        denom = jnp.concatenate([h_within, h_within])
        feats = feats / denom
    return feats


def interaction_features(
    x: Float[Array, "N D"],
    pairs: Int[Array, "K 2"],
) -> Float[Array, "N K"]:
    r"""Element-wise products on selected pairs of input columns.

    For each pair $(i, j)$ and each row $n$, computes
    $x_{n, i} \cdot x_{n, j}$.

    Args:
        x: Input matrix, shape ``(N, D)``.
        pairs: Index pairs, shape ``(K, 2)``. Empty pairs yield an
            ``(N, 0)`` output.

    Returns:
        Array of shape ``(N, K)`` of pairwise products.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import interaction_features
        >>> x = jnp.arange(6.0).reshape(2, 3)  # (N=2, D=3)
        >>> pairs = jnp.array([[0, 1], [0, 2]])  # (K=2, 2)
        >>> interaction_features(x, pairs).shape
        (2, 2)
    """
    if pairs.shape[0] == 0:
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    # x[:, pairs] has shape (N, K, 2); reduce the paired axis with prod.
    selected = x[:, pairs]
    return einx.prod("n k [two]", selected)


def standardize(
    x: Float[Array, "*shape"],
    mu: Float[Array, "*shape"],
    std: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    """Affine standardize: ``(x - mu) / std``.

    Broadcasts ``mu`` and ``std`` against ``x`` per the JAX broadcasting
    rules. ``std`` is *not* clamped; pass a positive value or guard
    upstream.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import standardize
        >>> x = jnp.array([1.0, 3.0, 5.0])
        >>> bool(jnp.allclose(standardize(x, 3.0, 2.0), jnp.array([-1, 0, 1])))
        True
    """
    return (x - mu) / std


def unstandardize(
    z: Float[Array, "*shape"],
    mu: Float[Array, "*shape"],
    std: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    """Inverse of `standardize`: ``z * std + mu``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import standardize, unstandardize
        >>> x = jnp.array([1.0, 3.0, 5.0])
        >>> # unstandardize undoes standardize for the same (mu, std).
        >>> z = standardize(x, 3.0, 2.0)
        >>> bool(jnp.allclose(unstandardize(z, 3.0, 2.0), x))
        True
    """
    return z * std + mu
