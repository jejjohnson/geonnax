# U-Net

A modern, dimension-flexible U-Net (1D / 2D / 3D via `num_spatial_dims`) for gridded
fields. The encoder applies a stage block then a factor-2 downsample at each level; the
bottleneck is `ResnetBlock → Attention → ResnetBlock`; the decoder upsamples, fuses the
$1/\sqrt{2}$-scaled skip connection, and applies a stage block.

Each stage can optionally be a **nested residual U-Net** (a U²-Net-style block) for
extra in-stage refinement, and leaf blocks can be either residual or ConvNeXt-V2. The
constituent layers live in [Building Blocks](layers.md).

!!! warning "Not resolution-invariant"
    `UNet` (and its alias `XUNet`) is an ordinary CNN: its fixed `kernel_size=3`
    stencils span a different physical scale on a 32² grid than on a 64² one, and its
    pointwise activations alias under resampling. The same weights therefore do **not**
    give the same function at two resolutions. On a smooth band-limited field the
    relative L2 gap between `u(x)` and `down(u(up(x)))` is about 1.0 (an entirely
    different output), against under 0.5 % for `FNO` and about 2 % for `WNO`. This is
    pinned as a strict `xfail` in `tests/test_resolution_invariance.py`; use
    [`FNO`](operators.md) when evaluating at a different resolution from training. A
    discretisation-invariant convolutional operator (`CNO`) is tracked in
    [#33](https://github.com/jejjohnson/geonnax/issues/33).

## U-Net

Also exported under the long-form alias `geonnax.XUNet`.

::: geonnax.unet.UNet

## Nested residual stages

::: geonnax.unet.NestedResidualUNet

::: geonnax.unet.Stage
