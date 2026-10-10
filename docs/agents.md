# Building with agents

geonnax exists so that a neural operator, a U-Net, a sine network or a
spherical-harmonic encoder is one tested `equinox.Module`, with the right
initialisation and shape conventions, that probabilistic and kernel
libraries can build on. Coding agents tend to re-implement what they cannot
see — an FFT convolution that keeps the low modes, a Legendre recursion, a
SIREN layer with a guessed uniform init, `cos(xW)` random features, a
degrees-to-radians helper — and each copy loses a property (resolution
invariance, an exact transform, a variance-preserving init, gradients at a
singular point). geonnax ships three things that let agents find it.

## The capability index

The [capability index](api/capabilities.md) lists every public name in
`geonnax` and its submodules (`geonnax.basis`, `geonnax.geo`,
`geonnax.layers`, …), grouped by the module that defines it, with a
one-line summary; then the shared private helpers contributors reuse. It is
regenerated from the code and checked in the test suite, so it never drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/geonnax
/plugin install geonnax@geonnax
```

The plugin adds:

- **`build-models-with-geonnax`** (skill) — loads whenever a task builds or
  trains a neural network on gridded, spherical or coordinate data, writes
  an FFT / spherical-harmonic / wavelet layer, encodes lon/lat or time,
  evaluates a basis, or needs a deterministic core for an uncertainty-aware
  model: what lives where, the rules (one example per call, `init` with an
  explicit key, immutable modules, units and grids), a worked example, and
  a "don't write it — use geonnax" table.
- **`geonnax-reuse-reviewer`** (subagent) — a read-only check of a diff for
  network and basis code geonnax already provides, and for misuse.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/geonnax/llms.txt): a curated map
of geonnax and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on
geonnax:

```markdown
## Neural networks: build on geonnax

This project uses geonnax (Equinox neural operators, U-Nets, SIREN / MFN,
coordinate encoders, bases, random features and uncertainty-aware cores).
Before writing a layer, a network, an encoder, a basis or random features,
search the capability index
(https://jejjohnson.github.io/geonnax/api/capabilities/) or
`geonnax.__all__` / `geonnax.basis.__all__`, and compose what exists:

- Build models with `Model.init(..., key=k)` from freshly split keys; they
  act on one example (`(D,)` or channel-first `(C, *spatial)`): batch with
  `jax.vmap`.
- Keep the module a method returns (`sn, y = sn(x)`,
  `gp = gp.update_precision(features)`); never mutate a module.
- Use `geonnax.FNO` / `SFNO` / `WNO` / `UNet` for gridded fields,
  `geonnax.SIREN` / `FourierNet` for coordinate networks,
  `geonnax.geo` and the encoder modules for lon/lat (`[lon, lat]`; the
  spherical-harmonic encoder takes radians), and `geonnax.basis` for
  eigenfunctions, RBF / Gabor / wavelet frames and EOFs.
- Monte-Carlo heads and dropout take a `key` per call; evaluate dropout
  models under `eqx.nn.inference_mode`.
```

## Working on geonnax itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/geonnax/blob/main/AGENTS.md) in
the repository: the module map, the boundaries with pyrox and kernellib,
"reuse before you write", the contracts and the tests that enforce them,
and recipe skills for adding layers, models, bases, encoders,
uncertainty-aware cores, conditioners and notebooks.
