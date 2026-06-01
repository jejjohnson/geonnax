"""Reusable convolutional neural-network building blocks (pure ``equinox.Module``).

These primitives back `geonnax.unet` but are deliberately standalone so
other models can reuse them. Every layer operates on a *single* example with a
leading channel axis and ``num_spatial_dims`` trailing spatial axes — shape
``(C, *spatial)`` — and is meant to be ``jax.vmap``-ed over a batch. With
``num_spatial_dims=2`` (the common case) the input is a ``(C, H, W)`` field.

- `geonnax.layers._conv` — `StandardizedConv` (weight-standardized
  convolution) and the `Downsample` / `Upsample` pixel-shuffle
  resamplers.
- `geonnax.layers._norm` — `GlobalResponseNorm` (ConvNeXt-V2).
- `geonnax.layers._blocks` — `Block`, `ResnetBlock`,
  `ConvNeXtBlock`, `SqueezeExcitation`.
- `geonnax.layers._attention` — `Attention`,
  `LinearAttention`.
- `geonnax.layers._spectral` — `SpectralConv` (Fourier neural
  operator layer).
- `geonnax.layers._factorized` — `DenseTensor` / `CPTensor`
  / `TuckerTensor` / `TTTensor` low-rank weight parameterisations.
- `geonnax.layers._spherical_transform` —
  `SphericalHarmonicTransform`, `SphericalSpectralConv`.
"""

from geonnax.layers._attention import Attention, LinearAttention
from geonnax.layers._blocks import (
    Block,
    ConvNeXtBlock,
    ResnetBlock,
    SqueezeExcitation,
)
from geonnax.layers._conv import Downsample, StandardizedConv, Upsample
from geonnax.layers._factorized import (
    CPTensor,
    DenseTensor,
    Factorization,
    TTTensor,
    TuckerTensor,
    init_factorized_tensor,
)
from geonnax.layers._norm import GlobalResponseNorm
from geonnax.layers._spectral import SpectralConv
from geonnax.layers._spherical_transform import (
    SphericalHarmonicTransform,
    SphericalSpectralConv,
)


__all__ = [
    "Attention",
    "Block",
    "CPTensor",
    "ConvNeXtBlock",
    "DenseTensor",
    "Downsample",
    "Factorization",
    "GlobalResponseNorm",
    "LinearAttention",
    "ResnetBlock",
    "SpectralConv",
    "SphericalHarmonicTransform",
    "SphericalSpectralConv",
    "SqueezeExcitation",
    "StandardizedConv",
    "TTTensor",
    "TuckerTensor",
    "Upsample",
    "init_factorized_tensor",
]
