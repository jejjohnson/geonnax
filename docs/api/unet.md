# U-Net

A modern, dimension-flexible U-Net (1D / 2D / 3D via `num_spatial_dims`) for gridded
fields. The encoder applies a stage block then a factor-2 downsample at each level; the
bottleneck is `ResnetBlock → Attention → ResnetBlock`; the decoder upsamples, fuses the
$1/\sqrt{2}$-scaled skip connection, and applies a stage block.

Each stage can optionally be a **nested residual U-Net** (a U²-Net-style block) for
extra in-stage refinement, and leaf blocks can be either residual or ConvNeXt-V2. The
constituent layers live in [Building Blocks](layers.md).

## U-Net

Also exported under the long-form alias `geonnax.XUNet`.

::: geonnax.unet.UNet

## Nested residual stages

::: geonnax.unet.NestedResidualUNet

::: geonnax.unet.Stage
