# Building Blocks

`geonnax.layers` collects the reusable convolutional and attention primitives that back
the [U-Net](unet.md) and the [operators](operators.md). They are deliberately
standalone — compose them into your own models. Each acts on a single `(C, *spatial)`
example; `jax.vmap` over a batch.

## Convolutions & resampling

::: geonnax.layers.StandardizedConv

::: geonnax.layers.Downsample

::: geonnax.layers.Upsample

## Residual & ConvNeXt blocks

::: geonnax.layers.Block

::: geonnax.layers.ResnetBlock

::: geonnax.layers.ConvNeXtBlock

::: geonnax.layers.SqueezeExcitation

## Normalisation

::: geonnax.layers.GlobalResponseNorm

## Attention

::: geonnax.layers.Attention

::: geonnax.layers.LinearAttention
