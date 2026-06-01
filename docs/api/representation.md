# Representation Networks

Coordinate-based networks that map low-dimensional inputs (positions, times) to field
values — the building blocks of implicit neural representations.

## SIREN

Sinusoidal representation networks (Sitzmann et al., 2020) use a sine activation,
$z_{i+1} = \sin(\omega\,(W_i z_i + b_i))$, with a frequency-aware initialisation so deep
sine networks stay well-conditioned.

::: geonnax.siren

## Multiplicative Filter Networks

MFNs (Fathony et al., 2021) build an output by repeatedly multiplying a linear map with
a Fourier or Gabor filter of the input, giving an analytically tractable basis.

::: geonnax.mfn
