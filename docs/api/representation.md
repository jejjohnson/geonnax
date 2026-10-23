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

## Bayesian Neural Field core

`NeuralFieldMLP` is the network of Google's Bayesian Neural Field (Saad et al., 2024)
without its priors: learned per-input scales, gated feature blocks (rescaled inputs,
dyadic Fourier, seasonal, interactions and an optional pre-encoded block such as
spherical-harmonic features), a learned mix of `elu` and `tanh`, and NTK-scaled dense
layers with learned gains. Train it deterministically, or let pyrox-nn swap its named
fields for priors.

```python
import jax, jax.numpy as jnp, jax.random as jr
import geonnax

enc = geonnax.SphericalHarmonicEncoder(l_max=6, input_mode="lonlat")
field = geonnax.NeuralFieldMLP.init(
    (365.0, 180.0, 90.0), width=64, depth=2,           # inputs: (t, lon, lat)
    seasonality_periods=(365.25,), num_seasonal_harmonics=(3,),
    extra_features=49, key=jr.PRNGKey(0),
)
x = jnp.array([12.0, 3.5, 40.2])
y = field(x, enc(geonnax.geo.deg2rad(x[1:])))                # (1,)
```

::: geonnax.neural_field

## Multiplicative Filter Networks

MFNs (Fathony et al., 2021) build an output by repeatedly multiplying a linear map with
a Fourier or Gabor filter of the input, giving an analytically tractable basis.

::: geonnax.mfn

## Derivatives of fields

A neural field is differentiable in its coordinates, so geostrophic velocities,
elevation-change rates and PDE residuals come from automatic differentiation.
`geonnax.diffops` turns a single-example field into its gradient, Jacobian,
divergence, planar curl or Laplacian, and — for fields of a unit vector $n \in S^2$ —
its surface gradient (Cartesian or east/north) and Laplace–Beltrami operator, using
the radially constant extension $g(x) = f(x/\lVert x\rVert)$ so that
$\Delta_S f = \Delta g$ on the unit sphere. They are operators, not losses: build
residual costs from them in your own training code.

```python
import jax, jax.numpy as jnp, jax.random as jr
import geonnax
from geonnax import diffops

enc = geonnax.SphericalHarmonicEncoder(l_max=8, input_mode="cartesian")
net = geonnax.SIREN.init(in_features=81, hidden_features=64, out_features=1,
                         depth=3, key=jr.PRNGKey(0))
ssh = lambda n: net(enc(n))[0]                      # field on the unit sphere

lonlat = jnp.array([[3.0, 40.0], [5.5, 38.2]])
grad_en = jax.vmap(diffops.surface_grad_enu(ssh))(lonlat)   # (2, 2): east, north
```

::: geonnax.diffops
