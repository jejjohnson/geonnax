# Building Blocks

`geonnax.layers` collects the reusable convolutional and attention primitives that back
the [U-Net](unet.md) and the [operators](operators.md). They are deliberately
standalone — compose them into your own models. Each acts on a single `(C, *spatial)`
example; `jax.vmap` over a batch.

## Convolutions & resampling

::: geonnax.layers.StandardizedConv

::: geonnax.layers.Downsample

::: geonnax.layers.Upsample

## Band-limited resampling & anti-aliased activation

`Downsample` / `Upsample` rearrange pixels into channels; these change the
sampling rate of the *same* periodic function instead. `band_limited_resize`
truncates or zero-pads the Fourier coefficients (an ideal low-pass), and
`AntiAliasedActivation` evaluates a pointwise nonlinearity on a finer grid and
low-passes it back (up → σ → down), the band-limit-preserving pieces of the
Convolutional Neural Operator (Raonić et al., NeurIPS 2023). They reduce
aliasing rather than eliminate it.

::: geonnax.layers.band_limited_resize

::: geonnax.layers.AntiAliasedActivation

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

## Non-uniform Fourier transforms

Weighted Fourier transforms between values at arbitrary points of the latent
torus $[0, 1)^d$ and the truncated rfft spectrum that
[`SpectralConv`](operators.md#geonnax.layers.SpectralConv) mixes (the
geometry-facing halves of Geo-FNO, with quadrature weights added). On a
uniform grid with weights $1/S$ they reproduce `rfftn / S` and `irfftn`.

::: geonnax.layers.nonuniform_rfft

::: geonnax.layers.nonuniform_irfft

## Spectral convolution in a precomputed basis

The FNO / SFNO pattern for any basis $\Phi$ sampled at $N$ points with
quadrature weights $w$ — region Slepians, EOFs, graph- or mesh-Laplacian
eigenvectors. `BasisTransform` holds the frozen analysis
$\Phi^\top \operatorname{diag}(w)$ and synthesis $\Phi_{\text{out}}$ tables;
`BasisSpectralConv` mixes channels with one matrix per mode. Each acts on a
single `(C, N)` example of point values.

::: geonnax.layers.BasisTransform

::: geonnax.layers.BasisSpectralConv
