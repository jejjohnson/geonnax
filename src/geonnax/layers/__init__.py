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
- `geonnax.layers._spherical_wavelet` —
  `SphericalWaveletTransform`, `SphericalWaveletConv`.
- `geonnax.layers._wavelet` — `WaveletConv` plus the `dwt` / `idwt`
  discrete wavelet transforms.
- `geonnax.layers._wavelet_attention` — `WaveletAttention`
  (self-attention in the wavelet domain).
"""

from geonnax.layers._attention import (
    Attention,
    LinearAttention,
    WindowedAttention,
)
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
from geonnax.layers._spherical_wavelet import (
    SphericalWaveletConv,
    SphericalWaveletTransform,
)
from geonnax.layers._wavelet import Wavelet, WaveletConv, dwt, idwt
from geonnax.layers._wavelet_attention import WaveletAttention


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
    "SphericalWaveletConv",
    "SphericalWaveletTransform",
    "SqueezeExcitation",
    "StandardizedConv",
    "TTTensor",
    "TuckerTensor",
    "Upsample",
    "Wavelet",
    "WaveletAttention",
    "WaveletConv",
    "WindowedAttention",
    "dwt",
    "idwt",
    "init_factorized_tensor",
]
