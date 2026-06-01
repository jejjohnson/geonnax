"""Reusable convolutional neural-network building blocks (pure ``equinox.Module``).

These primitives back :mod:`geonnax.unet` but are deliberately standalone so
other models can reuse them. Every layer operates on a *single* example with a
leading channel axis and ``num_spatial_dims`` trailing spatial axes — shape
``(C, *spatial)`` — and is meant to be ``jax.vmap``-ed over a batch. With
``num_spatial_dims=2`` (the common case) the input is a ``(C, H, W)`` field.

- :mod:`geonnax.layers._conv` — :class:`StandardizedConv` (weight-standardized
  convolution) and the :class:`Downsample` / :class:`Upsample` pixel-shuffle
  resamplers.
- :mod:`geonnax.layers._norm` — :class:`GlobalResponseNorm` (ConvNeXt-V2).
- :mod:`geonnax.layers._blocks` — :class:`Block`, :class:`ResnetBlock`,
  :class:`ConvNeXtBlock`, :class:`SqueezeExcitation`.
- :mod:`geonnax.layers._attention` — :class:`Attention`,
  :class:`LinearAttention`.
"""

from geonnax.layers._attention import Attention, LinearAttention
from geonnax.layers._blocks import (
    Block,
    ConvNeXtBlock,
    ResnetBlock,
    SqueezeExcitation,
)
from geonnax.layers._conv import Downsample, StandardizedConv, Upsample
from geonnax.layers._norm import GlobalResponseNorm


__all__ = [
    "Attention",
    "Block",
    "ConvNeXtBlock",
    "Downsample",
    "GlobalResponseNorm",
    "LinearAttention",
    "ResnetBlock",
    "SqueezeExcitation",
    "StandardizedConv",
    "Upsample",
]
