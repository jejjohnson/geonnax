# geonnax

A pure-[Equinox](https://github.com/patrick-kidger/equinox) neural-network zoo for
geoscientific machine learning. Every model is a plain `equinox.Module` with an
`init(..., *, key)` constructor and no probabilistic-framework dependency.

## Installation

geonnax is not on PyPI yet; install it from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv add "geonnax @ git+https://github.com/jejjohnson/geonnax.git"
```

## Quickstart

Models act on a **single example** (leading channel axis, no batch dimension); use
`jax.vmap` to batch.

Three ways in: a coordinate network, a neural operator, and basis functions used as features in a model of your own.

```python
import equinox as eqx
import geonnax as gnx
import jax
import jax.numpy as jnp
from functax.basis.slepian import slepian_cap_basis
import jax.random as jr
from jaxtyping import Array, Float


# Shapes: N points, M = (l_max + 1)² harmonics, K Slepian modes, C channels, P × Q grid
lo, hi = jnp.array([-jnp.pi, -jnp.pi / 2]), jnp.array([jnp.pi, jnp.pi / 2])
lonlat: Float[Array, "N 2"] = jr.uniform(jr.key(0), (1_000, 2), minval=lo, maxval=hi)  # rad

# A · Coordinate network: one point in, one value out; vmap over the points
# zₘ(λ, φ) = Yₗᵐ(λ, φ),  l ≤ 10
enc = gnx.SphericalHarmonicEncoder(l_max=10, input_mode="lonlat")  # (2,) → (M,)  encoders
# f(z) = W₄ sin(ω W₃ sin(ω W₂ sin(ω₀ W₁ z + b₁) + b₂) + b₃) + b₄
net = gnx.SIREN.init(121, 128, 1, depth=3, key=jr.key(1))           # (M,) → (1,)  siren
model = eqx.nn.Sequential([eqx.nn.Lambda(enc), eqx.nn.Lambda(net)])
f: Float[Array, "N 1"] = jax.vmap(model)(lonlat)                    # (N, 2) → (N, 1)

# B · Neural operator: a whole field in, a field out
# v = 𝒢(u),  each block a convolution in the spherical-harmonic domain
op = gnx.SFNO.init(3, 1, key=jr.key(2), n_lat=32, n_lon=64, l_max=15)  # sfno
u: Float[Array, "C P Q"] = jnp.ones((3, 32, 64))                    # (C, P, Q)
v: Float[Array, "1 P Q"] = op(u)                                    # (C, P, Q) → (1, P, Q)

# C · Basis functions: features for any model (least squares, a GP, your own network)
# gₖ = argmax_g ∫_R g² / ∫_S² g²,  band-limited to l ≤ 20,  R a 40° cap at the pole
cap = slepian_cap_basis(20, 0.7, n_modes=40)                        # functax
phi: Float[Array, "N K"] = jax.vmap(gnx.SlepianEncoder(cap))(lonlat)  # (N, 2) → (N, K)  slepian
```

`f`, `v` and `phi` come out with shapes (1000, 1), (1, 32, 64) and (1000, 40).
`eqx.nn.Lambda` wraps the two modules because `eqx.nn.Sequential` passes a `key` to each layer; a two-field `eqx.Module`, as in the README's [example](https://github.com/jejjohnson/geonnax#example-map-methane-around-the-globe), does the same job.

## What's inside

- **[Operators](api/operators.md)** — Fourier (`FNO`) and Spherical (`SFNO`) neural
  operators with dense/CP/Tucker/TT spectral weights.
- **[U-Net](api/unet.md)** — dimension-flexible U-Net with nested residual stages.
- **[Building Blocks](api/layers.md)** — reusable conv/attention/spectral primitives.
- **[Representation Networks](api/representation.md)** — SIREN, multiplicative filter
  networks.
- **[Encoders & Geometry](api/encoders.md)**, **[Uncertainty & Bayesian](api/uncertainty.md)**,
  and **[Spectral Bases](api/bases.md)**.

## Links

- [API Reference](api/index.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/geonnax)
