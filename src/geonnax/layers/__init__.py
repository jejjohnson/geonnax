"""Reusable convolutional neural-network building blocks (pure ``equinox.Module``).

These primitives back `geonnax.unet` but are deliberately standalone so
other models can reuse them. Every layer operates on a *single* example with a
leading channel axis and ``num_spatial_dims`` trailing spatial axes — shape
``(C, *spatial)`` — and is meant to be ``jax.vmap``-ed over a batch. With
``num_spatial_dims=2`` (the common case) the input is a ``(C, H, W)`` field.

- `geonnax.layers._conv` — `StandardizedConv` (weight-standardized
  convolution) and the `Downsample` / `Upsample` pixel-shuffle
  resamplers.
- `geonnax.layers._antialias` — `band_limited_resize` (Fourier
  truncation / zero-padding resampling) and `AntiAliasedActivation`
  (up → σ → low-pass → down), the band-limited primitives of CNO.
- `geonnax.layers._norm` — `GlobalResponseNorm` (ConvNeXt-V2).
- `geonnax.layers._blocks` — `Block`, `ResnetBlock`,
  `ConvNeXtBlock`, `SqueezeExcitation`.
- `geonnax.layers._attention` — `Attention`,
  `LinearAttention`.
- `geonnax.layers._spectral` — `SpectralConv` (Fourier neural
  operator layer).
- `geonnax.layers._factorized` — `DenseTensor` / `CPTensor`
  / `TuckerTensor` / `TTTensor` low-rank weight parameterisations.
- `geonnax.layers._basis_conv` — `BasisTransform`,
  `BasisSpectralConv` (spectral convolution in any precomputed basis).
- `geonnax.layers._spherical_conv` — `SphericalSpectralConv`.
- `geonnax.layers._spherical_wavelet` — `SphericalWaveletConv`.
- `geonnax.layers._wavelet` — `WaveletConv` plus the `dwt` / `idwt`
  discrete wavelet transforms.
- `geonnax.layers._wavelet_attention` — `WaveletAttention`
  (self-attention in the wavelet domain).

`SphericalHarmonicTransform`, `nonuniform_rfft` and `nonuniform_irfft` moved
to `functax.transform` (jejjohnson/geonnax#153), and so did
`SphericalWaveletTransform` (jejjohnson/geonnax#158); their names still
resolve here for one release, with a `DeprecationWarning`.
"""

import functax.transform as _transform

from geonnax._deprecation import Moved, deprecated_dir, deprecated_getattr
from geonnax.layers._antialias import AntiAliasedActivation, band_limited_resize
from geonnax.layers._attention import (
    Attention,
    LinearAttention,
    WindowedAttention,
)
from geonnax.layers._basis_conv import BasisSpectralConv, BasisTransform
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
from geonnax.layers._spherical_conv import SphericalSpectralConv
from geonnax.layers._spherical_wavelet import SphericalWaveletConv
from geonnax.layers._wavelet import Wavelet, WaveletConv, dwt, idwt
from geonnax.layers._wavelet_attention import WaveletAttention


__all__ = [
    "AntiAliasedActivation",
    "Attention",
    "BasisSpectralConv",
    "BasisTransform",
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
    "SphericalSpectralConv",
    "SphericalWaveletConv",
    "SqueezeExcitation",
    "StandardizedConv",
    "TTTensor",
    "TuckerTensor",
    "Upsample",
    "Wavelet",
    "WaveletAttention",
    "WaveletConv",
    "WindowedAttention",
    "band_limited_resize",
    "dwt",
    "idwt",
    "init_factorized_tensor",
]

#: names that moved to functax, served with a DeprecationWarning until removal
_MOVED: dict[str, Moved] = {
    "SphericalHarmonicTransform": Moved(
        "functax.transform.SphericalHarmonicTransform",
        _transform.SphericalHarmonicTransform,
    ),
    "SphericalWaveletTransform": Moved(
        "functax.transform.SphericalWaveletTransform",
        _transform.SphericalWaveletTransform,
    ),
    "nonuniform_irfft": Moved(
        "functax.transform.nonuniform_irfft", _transform.nonuniform_irfft
    ),
    "nonuniform_rfft": Moved(
        "functax.transform.nonuniform_rfft", _transform.nonuniform_rfft
    ),
}

__getattr__ = deprecated_getattr(__name__, _MOVED)
__dir__ = deprecated_dir(__all__, _MOVED)
