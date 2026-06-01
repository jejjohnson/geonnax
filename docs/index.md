# geonnax

A pure-[Equinox](https://github.com/patrick-kidger/equinox) neural-network zoo for
geoscientific machine learning. Every model is a plain `equinox.Module` with an
`init(..., *, key)` constructor and no probabilistic-framework dependency.

## Installation

```bash
pip install geonnax        # or: uv add geonnax
```

## Quickstart

Models act on a **single example** (leading channel axis, no batch dimension); use
`jax.vmap` to batch.

```python
import jax, jax.numpy as jnp, jax.random as jr
import geonnax

# A resolution-invariant Fourier Neural Operator: (3, H, W) -> (1, H, W).
op = geonnax.FNO.init(
    in_channels=3, out_channels=1, n_modes=(16, 16),
    hidden_channels=32, n_layers=4, key=jr.PRNGKey(0),
)
y = op(jnp.ones((3, 64, 64)))            # (1, 64, 64)
ys = jax.vmap(op)(jnp.ones((8, 3, 64, 64)))   # (8, 1, 64, 64)
```

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
