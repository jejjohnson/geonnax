"""Fourier spectral convolution (the core Fourier Neural Operator layer).

A `SpectralConv` mixes channels in the truncated Fourier domain: it
`rfft`s the input, keeps the lowest `n_modes` frequencies, applies a
learnable complex channel mixing there, and inverts the transform. Because the
spatial grid size is read from the input at call time (never baked into the
module), the same layer evaluates at any resolution above the mode count — the
resolution-invariance that defines the FNO (Li et al., 2021).

The complex mixing weights can be stored densely or in a low-rank factorised
form (CP / Tucker / TT) via `geonnax.layers._factorized`, which trades a
large parameter count for a compact, regularised one (Kossaifi et al., 2023).
"""

from __future__ import annotations

import itertools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from geonnax.layers._factorized import (
    CPTensor,
    DenseTensor,
    Factorization,
    TTTensor,
    TuckerTensor,
    init_factorized_tensor,
)


FactorizedTensor = DenseTensor | CPTensor | TuckerTensor | TTTensor


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


class SpectralConv(eqx.Module):
    """Fourier spectral convolution over ``num_spatial_dims = len(n_modes)`` axes.

    Operates on a single example of shape ``(in_channels, *spatial)``; ``vmap``
    over a batch. Each spatial extent must exceed twice its mode count so the
    low/high-frequency corner blocks do not overlap.

    The forward map is, per retained mode ``k``,
    ``ŷ[o, k] = Σ_i Ŵ[i, o, k] · x̂[i, k]`` where ``x̂ = rfft(x)`` and ``Ŵ`` is
    the (optionally factorized) complex weight.

    Attributes:
        weights: One factorized complex weight per spectral corner block; each
            reconstructs to ``(in_channels, out_channels, *n_modes, 2)`` (the
            trailing axis holds real/imag parts).
        bias: Optional spatial-domain bias of shape ``(out_channels, *ones)``.
        in_channels, out_channels: Channel counts (static).
        n_modes: Retained modes per spatial axis (static).
        factorization: Weight parameterisation (static).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import SpectralConv
        >>> sc = SpectralConv.init(4, 6, (8, 8), key=jr.PRNGKey(0))
        >>> sc(jnp.ones((4, 32, 32))).shape
        (6, 32, 32)
        >>> sc(jnp.ones((4, 64, 64))).shape   # resolution-invariant
        (6, 64, 64)

        With a Tensor-Train factorization of the spectral weights:

        >>> sc = SpectralConv.init(8, 8, (4, 4, 4), key=jr.PRNGKey(0),
        ...                        factorization="tt", rank=0.5)
        >>> sc(jnp.ones((8, 16, 16, 16))).shape
        (8, 16, 16, 16)
    """

    weights: list[FactorizedTensor]
    bias: Float[Array, "C_out *ones"] | None
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: tuple[int, ...] = eqx.field(static=True)
    factorization: Factorization = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        *,
        key: Array,
        factorization: Factorization = "dense",
        rank: float | int = 0.5,
        bias: bool = True,
        init_scale: float | None = None,
    ) -> SpectralConv:
        """Construct a spectral convolution.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            n_modes: Retained Fourier modes per spatial axis; its length is the
                spatial rank.
            key: PRNG key.
            factorization: ``"dense"``, ``"cp"``, ``"tucker"``, or ``"tt"``.
            rank: Fractional (``0 < rank <= 1``) or absolute rank for the
                factorised forms; ignored when ``factorization="dense"``.
            bias: Whether to add a learnable spatial-domain bias.
            init_scale: Std of the (reconstructed) weights; defaults to the
                fan-based ``sqrt(2 / (in + out))``.

        Raises:
            ValueError: If ``n_modes`` is empty or any entry is non-positive.
        """
        if len(n_modes) == 0:
            raise ValueError("n_modes must be non-empty.")
        if any(m < 1 for m in n_modes):
            raise ValueError(f"all n_modes must be >= 1, got {n_modes}.")
        n_modes = tuple(int(m) for m in n_modes)
        scale = (
            math.sqrt(2.0 / (in_channels + out_channels))
            if init_scale is None
            else init_scale
        )
        n_corners = 2 ** (len(n_modes) - 1)
        # Trailing 2 carries real/imag, factorised like any other tensor axis.
        weight_shape = (in_channels, out_channels, *n_modes, 2)
        keys = jax.random.split(key, n_corners)
        weights = [
            init_factorized_tensor(
                weight_shape, factorization, key=keys[c], rank=rank, scale=scale
            )
            for c in range(n_corners)
        ]
        bias_arr = jnp.zeros((out_channels, *(1,) * len(n_modes))) if bias else None
        return cls(
            weights=weights,
            bias=bias_arr,
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            factorization=factorization,
        )

    def _weight(self, corner: int) -> Complex[Array, "C_in C_out *modes"]:
        """Reconstruct corner ``corner``'s weights and view them as complex."""
        w = self.weights[corner].reconstruct()
        return w[..., 0] + 1j * w[..., 1]

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        d = len(self.n_modes)
        spatial = x.shape[1:]
        axes = tuple(range(1, d + 1))
        x_ft = jnp.fft.rfftn(x, axes=axes)

        out_ft = jnp.zeros((self.out_channels, *x_ft.shape[1:]), dtype=x_ft.dtype)
        for corner, sl in enumerate(_corner_slices(self.n_modes)):
            block = x_ft[(slice(None), *sl)]  # (in, *n_modes)
            mixed = jnp.einsum("i...,io...->o...", block, self._weight(corner))
            out_ft = out_ft.at[(slice(None), *sl)].set(mixed)

        out = jnp.fft.irfftn(out_ft, s=spatial, axes=axes)
        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = ["SpectralConv"]
