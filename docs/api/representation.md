# Representation Networks

Coordinate-based networks that map low-dimensional inputs (positions, times) to field
values — the building blocks of implicit neural representations.

## SIREN

Sinusoidal representation networks (Sitzmann et al., 2020) use a sine activation,
$z_{i+1} = \sin(\omega\,(W_i z_i + b_i))$, with a frequency-aware initialisation so deep
sine networks stay well-conditioned.

::: geonnax.siren

## Multi-scale SIREN

A single `first_omega` fixes the frequency band a SIREN is biased toward.
`MultiScaleSIREN` runs $K$ SIRENs at log-spaced first-layer frequencies and sums their
readouts, $u(x) = \sum_k \mathrm{SIREN}_k(x)$, so mixed low- and high-frequency content
is covered without tuning one $\omega_0$.

Pick it over an MFN when the signal *decomposes spectrally* — PDE solutions with clean
low/high separation, multi-band audio, spatiotemporal fields whose axes differ in
frequency content. Pick an MFN when many scales must be resolved simultaneously at every
coordinate.

::: geonnax.multi_scale_siren

## Multiplicative Filter Networks

MFNs (Fathony et al., 2021) build an output by repeatedly multiplying a linear map with
a Fourier or Gabor filter of the input, giving an analytically tractable basis.

::: geonnax.mfn
