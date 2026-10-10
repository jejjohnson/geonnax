---
name: add-model
description: Add a network architecture to geonnax — a neural operator (FNO / SFNO / WNO / MSWT style), a U-Net variant or an implicit neural representation (SIREN / MFN style) — as a top-level module of eqx.Module classes with init(..., *, key), composed from geonnax.layers, exported, documented on an API page, indexed and tested (shapes, 1-D to 3-D, resolution, jit / vmap / grad). Use when asked to add, port or implement a model, operator, U-Net, INR or architecture in src/geonnax.
---

# Add a model

Read "1. Modules", "3. Numerics and initialisation" and "The public API" in
`AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md`: the operators (`FNO`, `SFNO`, `WNO`,
  `MSWT`), `UNet` (its options cover ConvNeXt blocks, nested U²-Net stages,
  dropout, multi-scale consolidation), the INRs (`SIREN`, `MultiScaleSIREN`,
  `FourierNet`, `GaborNet`) and conditioned INRs (`ConditionedINR`,
  `HyperSIREN`).
- A new option on an existing model (a factorization, a block type, a
  padding mode) is a change to that model, not a new one.
- Missing pieces (a new attention, transform or block) go to `layers/` first
  (the `add-layer` skill); the model composes them.
- An uncertainty-aware head or a probabilistic core is `add-uncertainty-core`.

## 2. Write it (`src/geonnax/<name>.py`)

- Module docstring: first sentence names the architecture and its paper (it
  becomes the capability-index group title); say what shapes it acts on.
- One `eqx.Module` per block plus the model; sub-modules and arrays as
  fields, configuration as `eqx.field(static=True)` (`num_spatial_dims`,
  `in_channels`, `out_channels`, `hidden_channels`, activation callables).
- `@classmethod init(cls, ..., *, key, ...)`: validate (`n_layers >= 1`,
  non-empty `n_modes`, divisibility of the grid by `2**levels`, the band
  limit against the grid) with `ValueError`s; split the key once per
  sub-module (`jax.random.split(key, n)`); count the keys you consume.
- `__call__` on **one example**: `(C_in, *spatial) -> (C_out, *spatial)` for
  operators and U-Nets, `(D_in,) -> (D_out,)` for INRs. Read grid extents at
  call time (resolution invariance), unless the model is bound to a grid it
  validated in `init` (as `SFNO` is to its SHT grid).
- Reuse: lift / project with 1×1 `eqx.nn.Conv` (`fno._pointwise`; don't add
  a fourth copy), `layers` blocks, `DomainPadding` for non-periodic inputs,
  `siren_W_limit` / `build_siren_specs` for SIREN-style inits.
- Optional randomness (dropout) takes `key: Array | None = None` and works
  under `eqx.nn.inference_mode`; nothing samples weights at call time.
- A long-form alias, if wanted, is the same object
  (`FourierNeuralOperator = FNO` followed by a `"""Verbose alias for `FNO`."""`
  string) and goes in `__all__` too.
- Docstring: the topology (formula in `$…$`), shapes, the paper, and a
  plural `Examples:` section whose `>>>` lines build a small model and print
  output shapes (single example and `jax.vmap`).

## 3. Export, document, index

- `__all__` in the new module; in `src/geonnax/__init__.py` add the module
  to the `from geonnax import (...)` block, import its public names, and add
  both the module and the names to `__all__` (RUF022 sorts it).
- Docs: add `::: geonnax.<name>.<Class>` entries under a new heading of
  `docs/api/operators.md` (operators), `unet.md` (U-Nets) or a
  `::: geonnax.<name>` block in `representation.md` (INRs). A new page also
  needs a row in `docs/api/index.md` and an entry in the `nav` of
  `mkdocs.yml`.
- `make capabilities`.

## 4. Tests (`tests/test_<name>.py`)

- Output shapes for the documented configurations, and every validation
  error.
- `@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])`
  for dimension-flexible models.
- The defining property against a reference: resolution invariance on two
  grids (`test_spectral_conv_resolution_invariance` resizes with
  `jax.image.resize` and bounds the relative error, with a comment on why),
  exact reconstruction, identity at init, agreement with a published value.
- `@pytest.mark.integration`: `jax.vmap` over a batch, `eqx.filter_jit`,
  finite `eqx.filter_grad` of a mean-squared loss.
- `@pytest.mark.slow` on broad sweeps (every factorization, every block type).

## 5. Verify

```bash
uv run pytest -o addopts= -n 2 tests/test_<name>.py        # every tier
uv run pytest -o addopts= --doctest-modules src/geonnax/<name>.py
uv run pytest -o addopts= tests/test_capabilities.py tests/test_docstrings.py tests/test_docstrings_render.py tests/test_geonnax.py
```

then the `pre-pr-check` skill (it includes the strict docs build).
