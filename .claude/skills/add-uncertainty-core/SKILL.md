---
name: add-uncertainty-core
description: Add the deterministic core of an uncertainty-aware or probabilistic layer to geonnax — random-feature maps, GP output heads (SNGP / VSSGP style), spectral normalisation, BatchEnsemble / rank-1 layers, heteroscedastic, mixture-density or structured (CRF) output heads, input perturbations — as an eqx.Module whose named parameter fields a pyrox-nn wrapper can swap for NumPyro sites, with no numpyro import, explicit keys for Monte-Carlo passes, functional state updates, docs, index and tests. Use when asked to add, port or implement such a core, head or layer in src/geonnax.
---

# Add an uncertainty-aware core

Read "Boundaries" and "1. Modules" in `AGENTS.md` first; this is the
step-by-step. The probabilistic half (priors, sample / param sites, guides,
KL terms) is a pyrox-nn wrapper that takes this core and swaps its fields
(`eqx.tree_at(lambda c: (c.W_loc, …), core, sampled)`); design the core so
that wrapper is a few lines (pyrox's `add-bayesian-layer` skill is the other
half).

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md`: `rff_forward` / `rff_cosine_forward` /
  `orthogonal_blocks` / `OrthogonalRandomFeatures`,
  `RandomFeatureGaussianProcess` + `LaplaceRandomFeatureCovariance`,
  `DeepVSSGPCore`, `SpectralNormalization`, `DenseRank1`,
  `LayerNormEnsemble`, `MultiHeadAttentionBE`, `init_rank1_proj` /
  `apply_rank1_proj`, `HeteroscedasticHead` (+ `MCSoftmaxDenseFA`,
  `MCSigmoidDenseFA`, `hetero_noisy_logits`), `MixtureOfGaussiansDenseHead`,
  `LinearChainCRF`, `NCPContinuousPerturb`, and MC dropout through
  `UNet(dropout=...)`.
- A prior, a sample site or a variational family is pyrox's, not geonnax's.
  A kernel (or its spectral density) is kernellib's; a core may only fix a
  standard-normal (RBF) frequency draw at init, as the SNGP head does.

## 2. Write it (`src/geonnax/<name>.py`, or extend the family's module)

- Module docstring: the method and paper, and one line on which part stays
  in the consuming probabilistic library (as `sngp.py` and `ncp.py` say).
- `class Name(eqx.Module)`: **every parameter a wrapper may sample is its
  own named array field** (`W`, `b`, `W_loc`, `b_loc`, `output_linear`,
  `W_freqs`, …) with a jaxtyping shape; configuration (`in_features`,
  `rank`, `num_mc_samples`, init constants such as `diag_init_bias`) is
  static. Those field names become pyrox's API: choose them once.
- `@classmethod init(cls, ..., *, key, ...)`: validate with `ValueError`s,
  split the key per draw, and use the published init with the private
  helpers (`ensemble._glorot_uniform`, `sngp._glorot_normal`, `ensemble._rs_init`
  centred on 1 for rank-1 vectors) — don't add another Glorot.
- `__call__` on **one example** (`(D_in,)`); ensemble outputs gain a leading
  member axis `(M, D_out)` as `DenseRank1` does.
- Monte-Carlo passes take `*, key` and a static `num_mc_samples`; return the
  MC average (and expose the raw samples through a function like
  `hetero_noisy_logits`).
- Frozen random draws (RFF frequencies, phases) are array leaves read
  through `jax.lax.stop_gradient`; a trainable lengthscale is an array.
- State (EMA precisions, power-iteration vectors) is updated by a method
  that returns a **new** module via `eqx.tree_at`
  (`update_precision`, `SpectralNormalization.__call__ -> (new, y)`).
- Numerics: symmetrise and add the ridge before a Cholesky; solve
  triangular systems instead of inverting; floor norms
  (`spectral_norm._l2_normalize`); log-space for normalisers
  (`logsumexp`, as `LinearChainCRF` does); softplus / exp for positive
  scales.
- No `numpyro`, no `jax.random` call without an explicit key argument.
- Docstring: the model in `$…$`, shapes, the reference, which fields a
  Bayesian wrapper samples, and a plural `Examples:` section.

## 3. Export, document, index

- `__all__` in the module; import the module and its names in
  `src/geonnax/__init__.py` and add them to `__all__`.
- `docs/api/uncertainty.md`: a heading and a `::: geonnax.<name>` block (or
  extend the family's section).
- `make capabilities`.

## 4. Tests

In `tests/test_geonnax.py` (the Tier C / D / E sections), or a dedicated
file as `tests/test_spectral_norm.py` and `tests/test_heads.py` are:

- shapes, validation errors, and `jax.jit` of the forward;
- **a `test_<name>_module_no_numpyro`** that reads the module source and
  asserts `"numpyro" not in src`, like `test_sngp_module_no_numpyro`;
- the defining property against a reference: brute-force enumeration (the
  CRF tests), the exact SVD for a spectral norm, a probability simplex for
  an MC softmax, identity at init for rank-1 vectors, the EMA formula for a
  precision update;
- Monte-Carlo and random-feature claims bounded by the estimator's own
  standard error, with the bound's source in a comment; fixed keys
  (`jr.PRNGKey(0)`) for incidental randomness;
- `eqx.filter_grad` finite and zero on frozen leaves.

## 5. Verify

```bash
uv run pytest -o addopts= -n 2 tests/test_geonnax.py tests/test_heads.py tests/test_spectral_norm.py
uv run pytest -o addopts= --doctest-modules src/geonnax/<name>.py
uv run pytest -o addopts= tests/test_capabilities.py tests/test_docstrings.py tests/test_docstrings_render.py
```

then the `pre-pr-check` skill. If pyrox-nn will wrap the core, say in the PR
which fields the wrapper should sample.
