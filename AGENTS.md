# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

geonnax is a zoo of **deterministic** neural-network building blocks for
geoscientific machine learning in JAX: plain `equinox.Module` cores (neural
operators, U-Nets, implicit neural representations, uncertainty-aware heads,
coordinate encoders) and pure-JAX basis functions, with no probabilistic
dependency. Two GeoML libraries build on it:

- **pyrox** (`pyrox-nn`, `pyrox-gp`) wraps geonnax cores and swaps their named
  parameter fields for NumPyro sample / param sites (`eqx.tree_at` on `W`,
  `proj`, `generator`, `Omega`, …), re-exports the deterministic cores from
  `pyrox_nn`, and re-exports `geonnax.basis` from `pyrox_gp._basis`;
- **kernellib** builds its random-feature maps on `geonnax.randfeat`
  (`rff_forward`, `orthogonal_blocks`) and uses `geonnax.basis`.

Both pin geonnax by git tag
(`geonnax @ git+https://github.com/jejjohnson/geonnax.git@vX.Y.Z`; it is not
on PyPI), so **the public names, constructor arguments and parameter field
names are their API**: a rename or removal needs a deprecation (see "The
public API"). New code composes what is here, and anything genuinely new
lands where the next person will find and reuse it.

It is one package, `src/geonnax/`. The public API is `geonnax.__all__` plus
the public submodules (each with its own `__all__`); names reachable only
from a submodule are the bases (`geonnax.basis`), the lon/lat helpers
(`geonnax.geo`), `geonnax.layers.dwt` / `idwt` / `Factorization` / `Wavelet`,
`geonnax.mswt.WaveletDownsample` / `WaveletUpsample` and
`geonnax.unet.BlockType`.

| Family | Modules | Owns |
|---|---|---|
| Bases (pure functions) | `_basis/` (implementation), `basis.py` (the public surface) | Dirichlet / Fourier eigenpairs on a box, divergence-free atoms, real spherical harmonics, Slepian cap bases, graph-Laplacian eigenpairs, RBF / Wendland and Gabor frames, DWT matrices, EOFs; Fourier / seasonal / interaction / time-window features, `standardize` |
| Geography | `geo.py`, `encoders.py`, `slepian.py` | lon/lat helpers (`deg2rad`, `lonlat_scale`, `lonlat_to_cartesian3d`, `cyclic_encode`, `spherical_harmonic_encode`); parameter-free encoder modules; `GeoContextEncoder`; Slepian encoders |
| Building blocks | `layers/` | `StandardizedConv`, `Downsample` / `Upsample`, `Block`, `ResnetBlock`, `ConvNeXtBlock`, `SqueezeExcitation`, `GlobalResponseNorm`, attention (`Attention`, `LinearAttention`, `WindowedAttention`, `WaveletAttention`), `SpectralConv` + factorized weights (`DenseTensor`, `CPTensor`, `TuckerTensor`, `TTTensor`), spherical harmonic / wavelet transforms and convs, `dwt` / `idwt`, `WaveletConv` |
| Neural operators | `fno.py`, `sfno.py`, `wno.py`, `mswt.py` | `FNO`, `SFNO`, `WNO`, `MSWT` and their blocks |
| U-Net | `unet.py` | `UNet` (alias `XUNet`), `Stage`, `NestedResidualUNet` |
| Implicit neural representations | `siren.py`, `multi_scale_siren.py`, `mfn.py`, `conditioning.py` | `SIREN`, `SirenDense`, the SIREN init rules, `MultiScaleSIREN`, `FourierNet` / `GaborNet`, conditioners (`ConcatConditioner`, `AffineModulation` = `FiLM`, `HyperLinear`), `ConditionedINR`, `HyperSIREN` |
| Uncertainty-aware cores | `randfeat.py`, `sngp.py`, `vssgp.py`, `spectral_norm.py`, `ensemble.py`, `heteroscedastic.py`, `mixture.py`, `crf.py`, `ncp.py` | Random Fourier features, the SNGP head and its Laplace covariance, the deep VSSGP core, spectral normalisation, BatchEnsemble layers, heteroscedastic heads, mixture-density head, linear-chain CRF, NCP input perturbation — each the deterministic half of a probabilistic layer pyrox builds |

Imports point one way: `_basis` imports nothing else in geonnax; `layers`
imports only itself and `_basis`; `geo` → `_basis`; the model modules import
`layers` / `_basis` / `geo`, and only these model-to-model edges exist:
`conditioning` and `multi_scale_siren` → `siren`, `vssgp` → `randfeat`,
`slepian` → `encoders` / `geo`. No test enforces this; keep it.

### Boundaries

- **No probabilistic framework.** geonnax never imports `numpyro` (or any
  other PPL); priors, sample sites, guides and KL terms live in pyrox
  (`tests/test_geonnax.py::test_no_numpyro_import_in_package` and the
  per-module `test_*_module_no_numpyro`). Stochastic *forward* passes are
  fine when the randomness comes from an explicit `key` argument
  (`NCPContinuousPerturb`, the heteroscedastic heads, dropout).
- **No kernels.** Kernel objects, spectral densities and kernel-aware feature
  maps live in kernellib (and `pyrox_gp`); geonnax returns kernel-free
  geometry (eigenvalues, centres and widths, frequency matrices).
- **Runtime dependencies** are jax, equinox, jaxtyping, einx and einops
  (`pyproject.toml`); NumPy (a jax dependency) is used only for eager,
  construction-time tables. No new runtime dependency without discussion.

## What geonnax is built on

| Library | geonnax uses it for | The rule it brings |
|---|---|---|
| **equinox** | Every model, layer and container is an `eqx.Module` pytree; `eqx.nn.Conv` / `Linear` / `GroupNorm` / `Dropout` inside layers; `eqx.tree_at` for functional updates | Modules are immutable: anything that "updates" (power-iteration state, the SNGP precision) returns a new module. Configuration (sizes, flags, activations, init constants) is `eqx.field(static=True)`; arrays are never static. No dataclasses, no mutable containers holding arrays. |
| **jax** | `jax.random`, `jit`, `vmap`, `grad`, FFTs | Explicit keys, split before reuse; pure forward passes; no Python control flow on traced values; one example per call, batch with `jax.vmap`. |
| **einx** | Named-axis contractions, transposes, broadcasts and reductions (`einx.dot`, `einx.id`, `einx.sum`, `einx.subtract`, …) | The house style for any non-trivial reshaping, as most modules already do; not enforced by lint or tests (older code mixes in `reshape`, `.T` and `axis=`), so match the module you are in and prefer einx in new code. |
| **jaxtyping** | Shape annotations (`Float[Array, "C *spatial"]`) | Annotate every public array argument and return; ruff ignores `F722` / `F821` / `UP037` in `src/` and `tests/` for them. |

geonnax builds on no other GeoML-stack library, so the capability index has
no upstream section.

## Reuse before you write

Before writing a helper, a layer, a basis or an init rule, find out whether it
exists:

1. **Search the capability index.**
   [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every public
   name in geonnax once, at its public home, grouped by the module that
   defines it, with a one-line summary; then the shared private helpers. It
   is generated (`make capabilities`) and checked by
   `tests/test_capabilities.py`, so it is current.
2. **Search the shared private helpers** (the end of the index): einx
   shuffle patterns, the GroupNorm group count, the dropout and key-splitting
   helpers, the rank resolver, the Fourier corner slices, wavelet filter
   banks, Glorot inits, the double-`where` `_safe_sqrt` / `_safe_arccos`,
   lon/lat validators.
3. **If it is missing, add it to the module that owns the concept** — a
   building block to `layers/`, a basis to `_basis/` (re-exported from
   `geonnax.basis`), a helper two modules need to the owner's `_utils` or the
   lowest module both already import — not inline in the one caller you have
   today. Copies already exist (`_pointwise` in `fno` / `sfno` / `wno`,
   `_glorot_uniform` in `ensemble` / `heteroscedastic`, `_require_positive`
   in `siren` / `mfn`, `_safe_sqrt` in `_rbf` / `_gabor`); don't add more.
4. **One object, one name.** A name means one thing across `geonnax` and its
   submodules (`tests/test_capabilities.py`); the deliberate exceptions are
   listed with a reason in `ALLOWED_SHARED_NAMES` in
   `scripts/capabilities.py`. Short aliases of a model (`FNO` /
   `FourierNeuralOperator`, `FiLM` / `AffineModulation`) bind the same object.

| You are about to write… | Use instead |
|---|---|
| A 1×1 conv, conv + norm + activation, a residual block | `eqx.nn.Conv` (see `fno._pointwise`), `geonnax.layers.Block`, `ResnetBlock`, `ConvNeXtBlock` |
| Pixel-shuffle down / up-sampling, space-to-depth | `geonnax.layers.Downsample` / `Upsample`; the patterns in `layers._utils.shuffle_patterns` |
| Self-attention over a grid | `geonnax.layers.Attention`, `LinearAttention`, `WindowedAttention`, `WaveletAttention` |
| An FFT conv keeping the low modes | `geonnax.layers.SpectralConv` (dense / CP / Tucker / TT weights via `init_factorized_tensor`) |
| A spherical harmonic transform on a lat/lon grid | `geonnax.layers.SphericalHarmonicTransform`, `SphericalSpectralConv`; for wavelets `SphericalWaveletTransform` |
| A discrete wavelet transform | `geonnax.layers.dwt` / `idwt`, `WaveletConv`; as matrices `geonnax.basis.wavelet_basis_1d` / `_2d` |
| Real spherical harmonics, Laplacian eigenfunctions, RBF / Gabor / Slepian columns, EOFs | `geonnax.basis` (`real_spherical_harmonics`, `fourier_basis`, `rbf_basis`, `gabor_frame_grid`, `slepian_cap_basis`, `eof_basis`, …) |
| Fourier / seasonal / interaction features | `geonnax.basis.fourier_features`, `seasonal_features`, `interaction_features`, `gaussian_window_features` |
| Degrees → radians, lon/lat → unit sphere, cyclic sin/cos | `geonnax.geo` functions; as modules `geonnax.Deg2Rad`, `Cartesian3DEncoder`, `CyclicEncoder`, `SphericalHarmonicEncoder`, `GeoContextEncoder` |
| `cos(xW/ℓ)`, `sin(xW/ℓ)` random features, orthogonal frequency blocks | `geonnax.rff_forward`, `rff_cosine_forward`, `orthogonal_blocks`, `OrthogonalRandomFeatures` |
| A SIREN layer or its `U(-a, a)` init | `geonnax.SirenDense`, `SIREN`, `siren_W_limit`, `build_siren_specs` |
| FiLM / concat / hypernetwork conditioning | `geonnax.AffineModulation` (`FiLM`), `ConcatConditioner`, `HyperLinear`, `ConditionedINR`, `HyperSIREN` |
| Spectral normalisation, power iteration | `geonnax.SpectralNormalization` |
| An SNGP head, an EMA Laplace precision | `geonnax.RandomFeatureGaussianProcess`, `LaplaceRandomFeatureCovariance` |
| BatchEnsemble rank-1 layers | `geonnax.DenseRank1`, `LayerNormEnsemble`, `MultiHeadAttentionBE`, `init_rank1_proj` / `apply_rank1_proj` |
| Glorot init, positive-argument checks, a safe `sqrt` / `arccos` | the private helpers in the index (`ensemble._glorot_uniform`, `siren._require_positive`, `_basis._rbf._safe_sqrt`, …) |
| `x.reshape`, `x.T`, `x[:, None] * y`, `jnp.sum(x, axis=…)` in new code | einx: `einx.id("a b -> b a", x)`, `einx.multiply("n, n d -> n d", …)`, `einx.sum("[n] d", x)` |

## The contracts

Everything composes because it keeps these contracts; pyrox and kernellib
rely on them. Break one and the code runs on your example, then fails under
`vmap`, inside a wrapper, or in someone else's pipeline.

### 1. Modules: Equinox cores with an `init` constructor

- **Subclass `eqx.Module`.** Declare array leaves first, then configuration
  as `eqx.field(static=True)` (sizes, ranks, flags, activations, wavelet
  names, init constants). An array is never static (it is unhashable and
  breaks `jit`); a Python number that should be learnable is stored as an
  array (`jnp.asarray(lengthscale)`).
- **Construct with a classmethod `init(..., *, key, ...)`** that validates its
  arguments (raise `ValueError` naming the bad value), splits the key once
  per sub-module or parameter (`jax.random.split(key, n)`), draws the
  initial weights and calls `cls(...)`. Modules without random weights take
  no key: they are constructed directly (the coordinate encoders,
  `NCPContinuousPerturb`, `DomainPadding`) or by a keyless classmethod
  (`SlepianEncoder.from_cap`, `SphericalHarmonicTransform.init`). Never sample
  at call time to build weights.
- **One example per call.** Inputs carry no batch axis: vectors `(D,)` for
  dense / INR / head modules, `(C, *spatial)` channel-first fields for
  layers, operators and the U-Net, `(C, n_lat, n_lon)` on the sphere. Batch
  with `jax.vmap`. Basis functions are the exception: they take the `N`
  evaluation points and return `(N, M)` (the rows are points, not a batch).
- **Dimension-flexible where it can be.** Layers and operators take
  `num_spatial_dims` (or infer it from `len(n_modes)`) and work in 1-D, 2-D
  and 3-D; the tests parametrise `nd` over `1, 2, 3` (3 marked `slow`).
- **Read grid sizes at call time.** Weights are tied to modes, wavelet
  levels or harmonic degrees, not to the grid, so `SpectralConv` / `FNO` are
  resolution-invariant (`test_spectral_conv_resolution_invariance`); a layer
  that needs a fixed grid builds its tables in `init` and validates them
  there (`SphericalHarmonicTransform`, hence `SFNO`).
- **Functional state.** A module that tracks state returns a new module:
  `SpectralNormalization.__call__` returns `(new_module, y)`;
  `RandomFeatureGaussianProcess.update_precision` and
  `LaplaceRandomFeatureCovariance.update` return updated copies via
  `eqx.tree_at`.
- **Frozen arrays are leaves behind `jax.lax.stop_gradient`** at use
  (`RandomFeatureGaussianProcess.W` / `bias`, the SHT `synth` / `analy`
  tables, the spectral-norm iterates), not static fields.
- **Stochastic forward passes take `key`.** `__call__(x, *, key)` for
  sampling heads; `key: Array | None = None` where randomness is optional
  (dropout in `ResnetBlock`, `ConvNeXtBlock`, `UNet`), with
  `eqx.nn.inference_mode` for deterministic evaluation.
- **Named parameters are API.** pyrox swaps fields by name; renaming `W`,
  `b`, `proj`, `generator`, `layers`, `Omega`, `phi`, `output_linear`, … is a
  breaking change (see "The public API").

### 2. Bases: evaluation plus the half a prior needs

The basis contract (long form in [`docs/api/bases.md`](docs/api/bases.md)):

- Every basis evaluates to a matrix `Φ ∈ ℝ^{N×M}` from `N` input points
  (`<name>_basis`, `<name>_features`, or `<name>_frame` for overcomplete
  dictionaries).
- It also returns **the half a prior needs**: eigenvalues for spectral bases
  (`fourier_basis` → `(Φ, λ)`, `divfree_basis`, `graph_laplacian_eigpairs`,
  `SlepianCapBasis.eigenvalues`), the spectrum for data-driven ones (`eof_basis`),
  or per-atom geometry for localized and frame bases (`gabor_frame_grid` →
  `(Φ, centers, scales, wavenumbers)`; `rbf_basis` takes the caller's
  `centers` / `widths`).
- Pure functions of arrays: no `eqx.Module` unless it caches a precomputed
  decomposition (`SlepianCapBasis`), no PRNG, no kernel. Anything that needs
  concrete values (an eigendecomposition with NumPy, a filter table) runs
  eagerly and says so; the evaluation itself is `jit` / `vmap` safe.
- Implementation in `src/geonnax/_basis/_<name>.py`, re-exported from
  `_basis/__init__.py` **and** `geonnax/basis.py` (`__all__`), so downstream
  code never imports the private package.
- Gradients are finite at coincident points: use the double-`where`
  helpers (`_safe_sqrt`, `_safe_arccos`) for distances.

### 3. Numerics and initialisation

- **Init scales are part of the method.** Keep the published
  initialisation (SIREN's three regimes via `siren_W_limit`, Glorot for
  dense heads, small `head_scale` / `scale_init_factor` for SNGP and
  heteroscedastic heads, rank-1 vectors centred on 1) and cite it in the
  docstring.
- **Dtypes follow the input.** Build constants with the input's dtype
  (`jnp.zeros(n, dtype=x.dtype)`); integer coordinates are promoted with
  `geo._promote_to_floating`. The suite runs in JAX's default float32 (no
  `conftest.py` enables x64), so tolerances there are float32 tolerances.
- **Traceability.** Validate configuration in `init` (concrete Python
  values); a forward pass may not branch in Python on array values. Shapes
  depend only on static fields and input shapes.
- **Stable formulas.** Floor norms before dividing (`_l2_normalize`,
  `_SIGMA_FLOOR` in `spectral_norm`), symmetrise and jitter before a
  Cholesky (`LaplaceRandomFeatureCovariance._chol`), work in log space for
  CRF recursions, and guard `sqrt` / `arccos` / `log` at their singular
  points with a double `where`.
- **Grids.** Spectral layers keep `n_modes` below half of each extent;
  wavelet layers need each extent divisible by `2**level`; the SHT grid must
  resolve `l_max` (`n_lat > l_max`, `n_lon > 2·l_max`, checked in
  `SphericalHarmonicTransform.init`).

### The public API

- **Export** a new public name from its module's `__all__`, and from
  `src/geonnax/__init__.py` (import and `__all__`) unless it belongs only to
  a submodule surface (a basis lives in `geonnax.basis` only). Ruff's
  `RUF022` keeps every `__all__` sorted.
- **Document** it: pages that render a whole module
  (`::: geonnax.basis`, `::: geonnax.siren`, … in `docs/api/bases.md`,
  `encoders.md`, `representation.md`, `uncertainty.md`) pick it up
  automatically; pages that list classes one by one (`layers.md`,
  `operators.md`, `unet.md`) need a `::: geonnax.<module>.<Name>` entry. Then
  run `make capabilities`.
- **Docstrings**: Google style; the model or formula in MathJax (`$…$`,
  `$$…$$`) or Unicode, never Sphinx / RST markup (`:math:`, `.. math::`,
  `:class:` roles, `::` literal blocks); shapes in the jaxtyping
  annotations and in the prose; a reference for a published method; and a
  plural `Examples:` section whose body is `>>>` doctest lines or a fenced
  block (`tests/test_docstrings.py`, `tests/test_docstrings_render.py`). The
  `>>>` examples are doctests: `make doctest` runs them all (CI does not, so
  run it when you touch a docstring); print shapes or rounded values.
- **Deprecate, don't break.** pyrox and kernellib pin a tag and import these
  names, construct with these arguments and swap these fields. To rename or
  remove one, keep the old spelling working for at least one release with a
  `DeprecationWarning` that names the replacement (a module-level
  `__getattr__` for a renamed function or class, a keyword shim for an
  argument), say so in the PR, and mention that pyrox / kernellib must
  follow when they bump the tag. There is no shared deprecation helper yet;
  add one in the first PR that needs it.

## What enforces them

| Test | Enforces |
|---|---|
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name bound to two objects; every listed private helper exists |
| `tests/test_plugin_skill.py` | The downstream plugin's `geonnax.*` names exist; its worked example runs and its claims hold (slow) |
| `tests/test_geonnax.py::test_no_numpyro_import_in_package`, `test_*_module_no_numpyro` | No numpyro anywhere in geonnax |
| `tests/test_docstrings.py`, `tests/test_docstrings_render.py` | No Sphinx / RST markup in source; `Examples:` (plural) with a `>>>` or fenced body |
| `make doctest` (`--doctest-modules src/geonnax`; not run in CI) | Every docstring example runs |
| `tests/test_spectral.py` | Factorized tensors, `SpectralConv` (shapes, 1–3-D, resolution invariance), `FNO`, the SHT round trip, `SFNO` |
| `tests/test_wavelet.py`, `test_wavelet_attention.py`, `test_mswt.py`, `test_spherical_wavelet.py` | `dwt` / `idwt` exact reconstruction, `WaveletConv`, `WNO`, wavelet attention, `MSWT`, the spherical wavelet transform |
| `tests/test_unet.py`, `test_unet_dropout.py` | U-Net shapes, batching, gradients; dropout key threading and `inference_mode` |
| `tests/test_spectral_norm.py` | Power-iteration state, the realised Lipschitz constant, the all-zero weight |
| `tests/test_heads.py` | Mixture-density head; CRF forward / Viterbi against brute-force enumeration |
| `tests/test_new_bases.py`, `test_localized_bases.py` | Orthonormality and exactness of the EOF, divergence-free, spherical-RBF, wavelet, RBF / Wendland and Gabor bases |
| `tests/test_multi_scale_siren.py`, `test_geo_context_encoder.py`, `test_geonnax.py` | `MultiScaleSIREN`; `GeoContextEncoder`; smoke and forward checks for the rest of the public surface |
| ruff (`RUF022`, …), ty | Sorted `__all__`, lint, types (`src/geonnax`) |

The closed-form eigenbases (`fourier_basis`, `real_spherical_harmonics`,
`slepian_cap_basis`, `graph_laplacian_eigpairs`) are covered only by their
doctests and a smoke import; a change to them needs tests.

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add a building block to `geonnax.layers` (conv, block, norm, attention, spectral / spherical / wavelet transform) | `add-layer` |
| Add a model (neural operator, U-Net variant, implicit neural representation) | `add-model` |
| Add a basis or feature transform (`_basis/`, re-exported from `geonnax.basis`) | `add-basis` |
| Add a coordinate encoder or lon/lat helper (`geo`, `encoders`, `slepian`) | `add-encoder` |
| Add the deterministic core of an uncertainty-aware layer (random features, GP heads, spectral norm, ensembles, output heads) | `add-uncertainty-core` |
| Add a conditioner or a conditioned INR (`conditioning`) | `add-conditioner` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `geonnax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `numerics-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `make gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

Downstream users get geonnax's guidance through the Claude Code plugin in
`plugins/geonnax/` (published by `.claude-plugin/marketplace.json`) and
`docs/llms.txt` (served at the site root); see `docs/agents.md`. When the
public API or the headline usage changes, update
`plugins/geonnax/skills/build-models-with-geonnax/` too:
`tests/test_plugin_skill.py` checks that every `geonnax.X` it and the plugin
reviewer name still exists, and runs its worked example (slow tier).

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test-fast            # fast tier, what CI runs (no coverage): -m "not slow and not integration" -n auto
make test                 # every tier, in parallel, no coverage
make test-cov             # every tier with the coverage report (uv run pytest -v)
make doctest              # every docstring example (pytest --doctest-modules src/geonnax)
make lint                 # ruff check .   (entire repo, notebooks included)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check src/geonnax (what CI runs)
make capabilities         # regenerate docs/api/capabilities.md
make docs                 # mkdocs build;  make docs-serve for a local preview
make precommit            # pre-commit run --all-files
```

Run one test with
`uv run pytest tests/test_spectral.py::test_spectral_conv_shapes -v -o addopts=`
(`addopts` turns coverage on; `-o addopts=` drops it for a partial run).
Without `-m`, pytest selects every tier, slow and integration included.

### Test tiers

The markers are registered in `pyproject.toml` (`--strict-markers`):

- **Unmarked (fast):** unit tests, shapes, validation errors, small
  forward checks.
- **`@pytest.mark.slow`:** heavyweight model compilation or broad
  parametrised sweeps; for a parametrised test mark only the expensive case
  (`pytest.param(3, marks=pytest.mark.slow)` for the 3-D variant).
- **`@pytest.mark.integration`:** end-to-end model build → `jit` / `vmap` /
  `grad` smoke checks.

CI ("Tests", `ci.yml`) runs `uv run pytest -m "not slow and not integration"
-n auto` on Python 3.12 and 3.13 with coverage (uploaded to Codecov, no
gate: `fail_under = 0`). **No workflow runs the slow and integration tiers or
the doctests**, so run them locally for what you touched:

```bash
uv run pytest -m "slow or integration" -n auto -o addopts= tests/test_<area>.py
make doctest
```

### Tests that assert on random draws

- **Incidental randomness** (any initialisation would do): a fixed key,
  `jr.PRNGKey(0)` or a module-level `KEY`, as the suite does.
- **Sampling behaviour under test** (random features approximating a kernel,
  Monte-Carlo heads, dropout ensembles, power iteration from a random start):
  bound the estimator by its own sampling distribution, not a tolerance tuned
  on one draw, and say in a comment where the bound came from.
- Float32 tolerances: `atol=1e-5`–`1e-6` for exact identities (orthonormality,
  perfect reconstruction); state the source of anything looser.

### Before every commit

All of these must pass, from the repo root:

1. `make test-fast` (what CI runs), plus the slow / integration tests of what
   you touched and `make doctest` if you changed a docstring.
2. `uv run --group lint ruff check .` — the **entire** repo, which includes
   `tests/`, `scripts/` and the notebook cells. Never lint a subdirectory:
   CI runs `ruff check .`.
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (`uv run --group typecheck ty check src/geonnax`).
5. After changing a public API: `make capabilities` (and the `docs/api` entry).
6. After changing docs, docstrings or `mkdocs.yml`:
   `uv run --group docs mkdocs build --strict` (no workflow builds the docs on
   a PR; `pages.yml` deploys from `main`).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; ask when unsure;
   push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor adjacent code or add docstrings to code you didn't
   change; remove only what your change made unused, and mention unrelated
   dead code instead of deleting it.
4. **Goal-driven.** Turn the task into a check (a failing test, a reproduced
   bug, a closed form or a reference implementation to match) and loop until
   it passes.

Also: Python 3.12+, `from __future__ import annotations` in every module,
type hints on every public function, Google-style docstrings, `eqx.Module`
(not dataclasses) for anything that holds arrays, `NamedTuple` for small
static records (`SirenLayerSpec`, `Rank1ProjInit`), pure functions with side
effects isolated and explicit. When you suggest a change, propose the code;
when you fix a bug, add the test that reproduces it.

## Git, commits and pull requests

- Never push to or merge into `main` unless explicitly told to ("push to
  main", "merge to main"). Work on a feature branch, commit locally, and push
  only when asked; "merge the branch" means push the feature branch, not
  merge into `main`. Confirm before anything that affects a shared branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(layers): add …`); CI validates PR titles. Types:
  `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
  `chore`, `revert`. Breaking changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please (`release-please-config.json`): it
  bumps `pyproject.toml` and `src/geonnax/__init__.py` and writes
  `CHANGELOG.md` from `feat`, `fix`, `perf` and `revert` commits; don't bump
  versions by hand.
- **Never replace or remove an existing PR title or description.** Read it
  first; only append checklist items or update their status, and change the
  title only to fix a Conventional Commits violation. An agent called for a
  small follow-up must not supply a fresh description scoped to its own work.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the
  label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

MkDocs + Material + mkdocstrings + mkdocs-jupyter (`mkdocs.yml`, sources in
`docs/`); `pages.yml` deploys to GitHub Pages on every push to `main`
(`mkdocs gh-deploy --force`).

- **API pages** (`docs/api/*.md`) are organised by theme (operators, U-Net,
  building blocks, representation networks, encoders, uncertainty, bases),
  with hand-written prose and `:::` entries; the capability index is
  `docs/api/capabilities.md` (generated).
- **Math** renders through MathJax (`pymdownx.arithmatex`,
  `docs/javascripts/mathjax.js`): `$…$` and `$$…$$` in docstrings and pages.
- **Notebooks** live in `docs/notebooks/` as executed `.ipynb` files
  (rendered with `execute: false`, so the committed outputs are what readers
  see) and are listed in the `nav` of `mkdocs.yml`. Author in jupytext
  percent format, convert, execute, delete the `.py`; figures inline with
  `plt.show()`, never `savefig` or committed PNGs. Full standards:
  `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues.
