# Operators (FNO / SFNO)

Neural operators learn mappings between discretised functions rather than between
fixed-size vectors. Because the spatial grid is read at call time, an FNO is
**resolution-invariant**: train on one grid and evaluate on another.

A Fourier layer mixes channels in the truncated frequency domain:

$$
\hat{y}[o, k] \;=\; \sum_i \hat{W}[i, o, k]\,\hat{x}[i, k],
\qquad \hat{x} = \mathcal{F}(x),
$$

keeping only the lowest `n_modes` frequencies `k`. The full block adds a local
pointwise skip, $\;\mathrm{act}\!\left(\mathcal{F}^{-1}(\hat{W}\hat{x}) + W x\right)$.

The **Spherical FNO** swaps the planar FFT for a Spherical Harmonic Transform, so the
operator lives on the sphere — the right geometry for global lon/lat fields.

## Fourier Neural Operator

::: geonnax.fno.FNO

::: geonnax.fno.FNOBlock

::: geonnax.fno.DomainPadding

## Spectral convolution

::: geonnax.layers.SpectralConv

## Factorized spectral weights

The spectral weight tensor dominates an FNO's parameter count. Low-rank
factorizations (CP, Tucker, Tensor-Train) shrink it by 10–100× at comparable
accuracy and act as a regulariser — this is the Tensorized FNO (Kossaifi et al., 2023).
Select one through the `factorization=` argument of [`SpectralConv`](#geonnax.layers.SpectralConv)
or [`FNO`](#geonnax.fno.FNO).

::: geonnax.layers.init_factorized_tensor

::: geonnax.layers.DenseTensor

::: geonnax.layers.CPTensor

::: geonnax.layers.TuckerTensor

::: geonnax.layers.TTTensor

## Spherical FNO

::: geonnax.sfno.SFNO

::: geonnax.sfno.SphericalFNOBlock

::: geonnax.layers.SphericalSpectralConv

::: geonnax.layers.SphericalHarmonicTransform

## Wavelet Neural Operator

The WNO swaps the FNO's Fourier transform for a multi-level discrete wavelet
transform, mixing channels in the wavelet domain. Where Fourier modes are
global, wavelet subbands are spatially localised across scales — better suited
to sharp or non-stationary fields. Analysis and synthesis are exact adjoints,
so an orthonormal wavelet reconstructs perfectly; each spatial extent must be
divisible by `2 ** level`.

::: geonnax.wno.WNO

::: geonnax.wno.WNOBlock

::: geonnax.layers.WaveletConv

::: geonnax.layers.dwt

::: geonnax.layers.idwt

## Spherical wavelets

A scale-discretised (needlet) wavelet transform on the sphere, built on the
spherical harmonic transform above. Each scale is a smooth harmonic band-pass
window; the squared windows partition unity ($\sum_j g_j(\ell)^2 = 1$), so
analysis followed by synthesis is exact. These axisymmetric needlets give
*localised* multi-scale analysis of global fields — the spherical counterpart
to the planar wavelet transform.

::: geonnax.layers.SphericalWaveletTransform

::: geonnax.layers.SphericalWaveletConv

## Wavelet attention (MSWT)

The Multi-Scale Wavelet Transformer is an *attention-based* operator: it runs
self-attention inside the wavelet domain so the high-frequency detail bands stay
explicit and cannot be smoothed away, mitigating the spectral bias of grid-space
attention on chaotic / long-rollout dynamics (Wang et al., 2026). `MSWT` is a
patch-tokenised U-net of `WaveletAttention` blocks with wavelet down/up-sampling;
`WaveletAttention` is the reusable layer. Attention is global (`Attention`) or,
for large grids, Swin-style `WindowedAttention` (``O(N · window^d)``).

::: geonnax.mswt.MSWT

::: geonnax.mswt.MSWTBlock

::: geonnax.layers.WaveletAttention

::: geonnax.layers.WindowedAttention
