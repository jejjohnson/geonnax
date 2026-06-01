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
