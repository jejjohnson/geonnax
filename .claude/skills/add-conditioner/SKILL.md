---
name: add-conditioner
description: Add a conditioning layer to geonnax — an AbstractConditioner subclass c(h, z) -> y (concatenation, feature-wise modulation, hypernetwork style) built by init(num_features=, cond_dim=, key=) so ConditionedINR can place it after every layer of an inner network, or a new conditioned-INR constructor like HyperSIREN — with docs, index and tests. Use when asked to add FiLM-like modulation, a hypernetwork, a conditional INR or latent / context conditioning in src/geonnax/conditioning.py.
---

# Add a conditioner

Read "1. Modules" in `AGENTS.md` and the module docstring of
`src/geonnax/conditioning.py` first; this is the step-by-step. pyrox-nn
wraps the conditioners by swapping `proj.weight` / `proj.bias` and
`generator.weight` / `generator.bias`, so field names are its API.

## 1. Make sure it does not exist yet

- `ConcatConditioner` (`Linear([h ‖ z])`), `AffineModulation` / `FiLM`
  (`γ(z) ⊙ h + β(z)`, with `gamma_activation` in `"one_plus_tanh"`, `"exp"`,
  `"softplus"`, `"identity"`; `log_det` in `"exp"` mode), `HyperLinear`
  (`(W, b) = g(z)`), the composite `ConditionedINR` (feature or input mode)
  and `HyperSIREN` / `GeneratedSiren` — see `docs/api/capabilities.md`.
- A new γ activation is a new `gamma_activation` value in `_apply_gamma`
  (and `_GAMMA_ACTIVATIONS`), not a new class. Context *encoders* (lon/lat,
  time, covariates) are the `add-encoder` skill.

## 2. Write it (`src/geonnax/conditioning.py`)

- `class Name(AbstractConditioner)`: inherits the static `num_features`
  (`h.shape[-1]`) and `cond_dim` (`z.shape[-1]`); add its parameters as
  fields — an `eqx.nn.Linear` generator or projection named like the
  existing ones (`proj`, `generator`) so wrappers can swap
  `.weight` / `.bias` — and its configuration as static fields.
- `@classmethod init(cls, num_features, cond_dim, *, key, **options)`: this
  exact keyword interface is what `ConditionedINR.init` calls through
  `_build_conditioner` (`HyperLinear` is special-cased there because it
  needs `target_in` / `target_out`; avoid adding another special case).
  Validate options with `ValueError`s.
- Initialise so the conditioner starts near the identity in `h`
  (`AffineModulation` zero-initialises its generator's bias, so with the
  default `"one_plus_tanh"` γ = 1 and β = 0 for a zero context;
  `test_affine_modulation_identity_at_init` checks it) — a large random
  modulation destroys a SIREN's initialisation.
- `__call__(self, h: Float[Array, " C"], z: Float[Array, " K"], /) -> Float[Array, " C"]`
  on one example; einx for the products and broadcasts.
- Docstring: the formula, the parameter count, the reference, and a plural
  `Examples:` section (single example; `jax.vmap` over a batch of `(h, z)`).

## 3. Export, document, index

- Add it to `__all__` in `conditioning.py` and to the import
  block and `__all__` of `src/geonnax/__init__.py`.
- `docs/api/uncertainty.md` renders `::: geonnax.conditioning` whole; extend
  the conditioning prose with when to pick it. Mention it in the
  `ConditionedINR.init` docstring's list of `conditioner_cls` values.
- `make capabilities`.

## 4. Tests (`tests/test_geonnax.py`, the "Tier B: conditioning" section)

- shapes on one example and under `jax.vmap`;
- identity (or the documented behaviour) at init;
- inside `ConditionedINR.init(SIREN.init(...), conditioner_cls=Name, ...)` in
  both `mode="feature"` and `mode="input"`;
- the `z.shape[-1] != cond_dim` error and option validation;
- the existing `test_conditioning_module_no_numpyro` keeps passing.

## 5. Verify

```bash
uv run pytest -o addopts= tests/test_geonnax.py -k "conditioner or conditioning or conditioned or modulation or hyper or film"
uv run pytest -o addopts= --doctest-modules src/geonnax/conditioning.py
uv run pytest -o addopts= tests/test_capabilities.py tests/test_docstrings.py tests/test_docstrings_render.py
```

then the `pre-pr-check` skill.
