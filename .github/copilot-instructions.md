# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the module map,
the boundaries with pyrox and kernellib, "reuse before you write", the
contracts, the tests that enforce them, commands, the pre-commit checklist,
the docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/geonnax/`: pure-JAX bases (`_basis/`, public through
  `geonnax.basis`), lon/lat helpers and encoders (`geo`, `encoders`,
  `slepian`), building blocks (`layers/`), neural operators (`fno`, `sfno`,
  `wno`, `mswt`), the U-Net, implicit neural representations (`siren`,
  `multi_scale_siren`, `mfn`, `conditioning`) and uncertainty-aware cores
  (`randfeat`, `sngp`, `vssgp`, `spectral_norm`, `ensemble`,
  `heteroscedastic`, `mixture`, `crf`, `ncp`). Search
  [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a
  helper.
- geonnax is deterministic: never import numpyro; priors and sample sites
  live in pyrox, kernels in kernellib. Both pin geonnax by git tag and use
  its public names, constructor arguments and parameter field names, so a
  rename needs a deprecation.
- Keep the contracts in `AGENTS.md`:
  - **modules**: `eqx.Module` with array leaves and
    `eqx.field(static=True)` configuration, built by a classmethod
    `init(..., *, key)`; one example per call (`(D,)` or `(C, *spatial)`),
    batched with `jax.vmap`; state updates return a new module; stochastic
    forward passes take an explicit `key`;
  - **bases**: pure functions returning `Φ (N, M)` plus eigenvalues or
    per-atom geometry, implemented in `_basis/` and re-exported from
    `geonnax.basis`;
  - **numerics**: published init scales, dtypes that follow the input,
    no Python control flow on traced values, double-`where` guards at
    singular points; prefer einx for reshapes, transposes and contractions.
- Docstrings: Google style, MathJax math, no Sphinx / RST markup, and a
  plural `Examples:` section with `>>>` doctests (`make doctest`).
- Before committing, from the repo root: `make test-fast`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` after a public API change.
- Behaviour: don't nitpick what ruff or ty catch; propose a test with every
  fix; never suggest a change without the code for it; keep changes
  surgical.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
