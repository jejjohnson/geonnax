# geonnax

[![Tests](https://github.com/jejjohnson/geonnax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geonnax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geonnax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geonnax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geonnax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geonnax/actions/workflows/typecheck.yml)
[![codecov](https://codecov.io/gh/jejjohnson/geonnax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/geonnax)
[![PyPI version](https://img.shields.io/pypi/v/geonnax.svg)](https://pypi.org/project/geonnax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure-[Equinox](https://github.com/patrick-kidger/equinox) neural-network zoo
for geoscientific machine learning. Every model is a plain `equinox.Module`
with a `@classmethod init(..., *, key)` constructor and no probabilistic
dependencies — probabilistic libraries layer on top by wrapping a geonnax core
and swapping its named parameters for sample/param sites.

## Install

```bash
pip install geonnax          # or: uv add geonnax
```

For development, see [Contributing](#contributing).

## What's inside

| Module | Contents |
|--------|----------|
| `geonnax.unet` | `UNet` / `XUNet` — dimension-flexible (1D/2D/3D) U-Net with weight-standardized convs, ConvNeXt-V2 blocks, squeeze-excitation, cosine attention, and nested residual (U²-Net) stages |
| `geonnax.layers` | Reusable conv/attention building blocks: `StandardizedConv`, `ResnetBlock`, `ConvNeXtBlock`, `Attention`, `Downsample`/`Upsample`, … |
| `geonnax.siren` | `SIREN`, `SirenDense` — sinusoidal representation networks |
| `geonnax.mfn` | `FourierNet`, `GaborNet` — multiplicative filter networks |
| `geonnax.encoders` | Coordinate encoders: `Deg2Rad`, `CyclicEncoder`, `SphericalHarmonicEncoder`, … |
| `geonnax.slepian` | `SlepianEncoder`, `HybridSphericalSlepianEncoder` |
| `geonnax.randfeat` | `OrthogonalRandomFeatures` + RFF helpers |
| `geonnax.sngp` / `geonnax.vssgp` | SNGP / VSSGP Gaussian-process basis maps |
| `geonnax.conditioning` | `FiLM`, `HyperSIREN`, `ConditionedINR`, hypernetworks |
| `geonnax.basis` / `geonnax._basis` | Fourier / seasonal transforms and Dirichlet / spherical-harmonic / Slepian / graph-Laplacian eigenpairs |

## Quick start

Models operate on a **single example** (leading channel axis, no batch
dimension); use `jax.vmap` to batch.

```python
import jax, jax.numpy as jnp, jax.random as jr
import geonnax

# A 2D U-Net mapping a 3-channel (C, H, W) field to a 1-channel output.
model = geonnax.UNet.init(
    dim=32,
    channels=3,
    out_channels=1,
    num_spatial_dims=2,
    dim_mults=(1, 2, 4),
    key=jr.PRNGKey(0),
)

x = jnp.ones((3, 64, 64))           # (channels, height, width)
y = model(x)                        # (1, 64, 64)

batch = jnp.ones((8, 3, 64, 64))    # vmap over the batch axis
ys = jax.vmap(model)(batch)         # (8, 1, 64, 64)
```

The same `UNet` works for 1D profiles/series (`num_spatial_dims=1`) and 3D
volumes (`num_spatial_dims=3`). Set `nested_unet_depths=(2, 1, 1)` to enable
U²-Net-style nested stages, or `block_type="convnext"` for ConvNeXt-V2 blocks.
The reusable pieces in `geonnax.layers` can be composed into your own models.

## Contributing

Built with `uv`, `ruff`, `ty`, `pytest`, and MkDocs.

```bash
make install      # install all dependency groups + pre-commit hooks
make test         # run the test suite
make format       # auto-format and fix lint
make lint         # ruff check
make typecheck    # ty check
make docs-serve   # preview docs locally
```

Before committing, all four gates must pass: tests, `ruff check .`,
`ruff format --check .`, and `ty check src/geonnax`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) for the full
workflow, coding conventions, and the issue/epic model.

## License

MIT — see [LICENSE](LICENSE).

Author: [J. Emmanuel Johnson](https://jejjohnson.netlify.com) ·
Repo: <https://github.com/jejjohnson/geonnax>
