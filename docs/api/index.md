# API Reference

geonnax is a pure-[Equinox](https://github.com/patrick-kidger/equinox) neural-network
zoo for geoscientific machine learning. The reference is organised by theme rather
than dumped as one flat page:

| Section | What's inside |
|---------|---------------|
| [Operators (FNO / SFNO)](operators.md) | Fourier and Spherical Neural Operators, spectral convolutions, and factorized spectral weights |
| [U-Net](unet.md) | Dimension-flexible U-Net with nested residual (U²-Net) stages |
| [Building Blocks](layers.md) | Reusable `geonnax.layers` primitives — convolutions, blocks, attention, normalisation, resampling |
| [Representation Networks](representation.md) | SIREN and multiplicative filter networks |
| [Encoders & Geometry](encoders.md) | Coordinate encoders, lon/lat helpers, Slepian encoders |
| [Uncertainty & Bayesian](uncertainty.md) | SNGP, VSSGP, deep ensembles, heteroscedastic heads, conditioning |
| [Spectral Bases](bases.md) | Fourier / seasonal transforms and Dirichlet / spherical-harmonic / Slepian / graph-Laplacian eigenpairs |

## Conventions

A few patterns hold across the whole package:

- **Per-example, channel-first.** Modules act on a *single* example with a leading
  channel axis and trailing spatial axes — shape `(C, *spatial)`. Use
  [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) to map over
  a batch. For example a 2D field is `(channels, height, width)`.

- **`init` constructors.** Models are immutable `equinox.Module` pytrees built with a
  keyword-only PRNG key:

    ```python
    import jax.random as jr
    import geonnax

    model = geonnax.FNO.init(
        in_channels=3, out_channels=1, n_modes=(16, 16),
        hidden_channels=32, n_layers=4, key=jr.PRNGKey(0),
    )
    ```

- **Static vs. learnable.** Configuration (sizes, flags) is stored in
  `eqx.field(static=True)` fields; only array leaves are trained. Filter with
  `eqx.filter(model, eqx.is_inexact_array)` when building an optimiser.

- **Composability.** The pieces in [`geonnax.layers`](layers.md) back the U-Net and the
  operators but are standalone — compose them into your own models.

Every public class and function carries a runnable `Examples:` block in its docstring.
The snippets are real doctests — run `make doctest` (or
`pytest --doctest-modules src/geonnax`) to execute them all.
