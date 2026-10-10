---
name: geonnax-review
description: Review a change or pull request in geonnax against the repo's own rules — CODE_REVIEW.md, the module / basis / numerics contracts in AGENTS.md, the boundaries (no numpyro, no kernels), downstream compatibility with pyrox and kernellib, and reuse of existing layers, bases and helpers. Use when asked to review a diff, branch or PR in this repo.
---

# Review a geonnax change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which families it touches (bases, geography, layers, operators, U-Net,
   INRs and conditioning, uncertainty-aware cores).
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: hand-written
   1×1 convs, attention, FFT / wavelet / spherical-harmonic transforms,
   bases, Glorot inits, safe square roots and lon/lat conversions are the
   main way geonnax grows duplicates.
3. **Numerics and modules** — run the `numerics-reviewer` subagent in
   parallel with step 2: arrays in static fields, reused or missing keys,
   batch axes baked into `__call__`, dtype promotion, Python control flow on
   traced values, NaN gradients at coincident points, init scales, power
   iteration and Cholesky stability, grids that under-resolve a band limit.
4. **Boundaries** — no `numpyro` (or any PPL) import, no kernel objects or
   spectral densities; no new runtime dependency; imports keep pointing one
   way (`_basis` ← `geo` / `layers` ← models).
5. **Downstream compatibility** — for every renamed or removed public name,
   constructor argument or parameter field (`W`, `b`, `proj`, `generator`,
   `Omega`, `phi`, `layers`, `W_loc`, `output_linear`, …): is the old
   spelling kept with a `DeprecationWarning`? pyrox (`pyrox_nn`,
   `pyrox_gp._basis`) and kernellib (`geonnax.randfeat`, `geonnax.basis`)
   pin a tag and use them. A silent rename of a field pyrox swaps is
   Critical.
6. **Contracts** — for each family touched, the rules in `AGENTS.md`'s
   contracts: `init(..., *, key)`, one example per call, static
   configuration, functional state updates, `stop_gradient` on frozen
   leaves, the basis contract (`Φ` plus eigenvalues or geometry, re-exported
   from `geonnax.basis`), exports and `docs/api` entries, the capability
   index.
7. **Checklist** — the rest of `CODE_REVIEW.md` (docstrings with runnable
   `Examples:`, tests against references, tier markers), skipping what ruff,
   ty or the tests already enforce.
8. **Verify claims** — for anything you flag as a bug, run it:
   `uv run python -c "..."` with `jax.jit`, `jax.vmap`, `jax.grad` at the
   singular point, a float32 input, or the slow / integration tests and
   doctests of the touched module (no workflow runs them).

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
