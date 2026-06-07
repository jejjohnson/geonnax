# Bases

Deterministic feature transforms and basis matrices used as inputs to, or
components of, the models above.

## The basis contract

Every basis exposes two things, so a downstream probabilistic library can build
a prior over any of them uniformly:

1. an **evaluation** that returns the basis matrix $\Phi \in \mathbb{R}^{N\times M}$
   from a set of $N$ input points (`name_basis` / `name_features`, or
   `name_frame` for overcomplete dictionaries); and
2. the **half a prior needs** — *eigenvalues* $\lambda$ for the spectral bases
   (the prior variance is the kernel spectral density at $\sqrt{\lambda}$), or
   *per-atom geometry* (centres/widths, or centres/scales/wavenumbers) for the
   localized and frame bases (variance prescribed, or following a spectral law
   in the per-atom wavenumber).

geonnax returns geometry; the variance, kernel spectral density, and sample
sites stay in the consuming library. Note the leading axis $N$ is the set of
evaluation points (the rows of $\Phi$), **not** a batch axis — these are pure
functions of arrays, so `jax.vmap` / `jax.jit` over any extra leading axis as
needed.

## Feature transforms

Fourier, seasonal, and interaction feature maps, the localized Gaussian-in-time
window, and standardisation helpers.

::: geonnax.basis

## Eigenfunction, localized & overcomplete bases

Closed-form and graph-based eigenpairs — 1D Dirichlet/Fourier modes, real
spherical harmonics $Y_\ell^m$, Slepian functions on a cap, and graph-Laplacian
($L = D - A$) eigenvectors — alongside the **placeable** radial bases
(`rbf_basis`, with Gaussian or compactly-supported Wendland kernels) and the
fixed **overcomplete** multiscale `gabor_frame` / `gabor_frame_grid`. The
spectral bases return eigenvalues; the localized and frame bases return per-atom
geometry instead.

::: geonnax._basis
