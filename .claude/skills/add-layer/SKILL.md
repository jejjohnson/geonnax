---
name: add-layer
description: Add a reusable building block to geonnax.layers — a convolution, residual / ConvNeXt block, normalisation, attention variant, spectral / spherical / wavelet transform or conv, or a factorized spectral-weight form — as a channel-first (C, *spatial) eqx.Module with an init(..., *, key) constructor, exported, documented, indexed and tested in 1-D to 3-D. Use when asked to add, port or implement a layer, block, attention or transform in src/geonnax/layers.
---

# Add a building block to `geonnax.layers`

Read "1. Modules" and "3. Numerics and initialisation" under "The contracts"
in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (the `geonnax.layers._*` groups and the
  shared private helpers) for the block and its synonyms: pixel shuffle is
  `Downsample` / `Upsample`; "pre-norm cosine attention" is `Attention`;
  "Swin window attention" is `WindowedAttention`; "Tensorized FNO weights"
  are `init_factorized_tensor`.
- A composition of existing blocks (a `Block` + `SqueezeExcitation`, an
  `eqx.nn.Sequential`) needs no new class. A standard Equinox layer
  (`eqx.nn.Conv`, `Linear`, `GroupNorm`, `LayerNorm`, `Dropout`) is used
  directly, not wrapped.
- A whole network (an operator, a U-Net variant) is the `add-model` skill.

## 2. Where it goes

| It is… | File | Exemplar |
|---|---|---|
| A convolution or resampler | `src/geonnax/layers/_conv.py` | `StandardizedConv`, `Downsample` |
| A residual / gated block | `layers/_blocks.py` | `ResnetBlock`, `ConvNeXtBlock` |
| A normalisation | `layers/_norm.py` | `GlobalResponseNorm` |
| Attention over a grid | `layers/_attention.py`, `_wavelet_attention.py` | `Attention`, `WindowedAttention` |
| An FFT-domain layer | `layers/_spectral.py` (+ `_factorized.py` for a weight form) | `SpectralConv` |
| A transform on the sphere | `layers/_spherical_transform.py`, `_spherical_wavelet.py` | `SphericalHarmonicTransform` |
| A wavelet-domain layer | `layers/_wavelet.py` | `WaveletConv`, `dwt` / `idwt` |

A new family gets its own `layers/_<name>.py` with a module docstring whose
first sentence names it (the capability index uses it as the group title).

## 3. Write it

- `class Name(eqx.Module)`: array leaves (and sub-modules) first, then
  configuration as `eqx.field(static=True)` (`num_spatial_dims`, channel
  counts, flags, activation callables).
- `@classmethod init(cls, ..., *, key, ...)`: validate arguments with a
  `ValueError` naming the value; `jax.random.split(key, n)` once per
  sub-module or draw; build sub-layers with their own `init`.
- `__call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C_out *spatial"]`
  on **one example**; take `num_spatial_dims` (or infer it from
  `len(n_modes)`) so it works in 1-D, 2-D and 3-D. If the layer has
  dropout, take `key: Array | None = None` and build it with
  `_blocks._dropout_layer(rate)` (and `unet._split_keys` to thread keys).
- Reuse the helpers: `layers._utils.act` (SiLU), `group_count` for
  `eqx.nn.GroupNorm`, `shuffle_patterns` / `shuffle_kwargs` for space-to-
  depth, `_factorized._resolve_rank`, `_spectral._corner_slices`,
  `_wavelet._filter_bank`.
- Fixed tables (quadrature grids, filter banks, transform matrices) are
  built in `init` (NumPy is fine there), stored as array leaves and read
  through `jax.lax.stop_gradient`; validate that the grid resolves what the
  layer needs (as `SphericalHarmonicTransform.init` does).
- einx for reshapes, transposes and contractions in new code; dtypes follow
  the input.
- Docstring: what it computes (formula in `$…$`), the shapes, the published
  source, and a plural `Examples:` section with `>>>` lines that print a
  shape (`tests/test_docstrings_render.py` checks it renders; `make doctest`
  runs it).

## 4. Export, document, index

- Add it to `layers/__init__.py`: the import, `__all__`, and the bullet list
  in the module docstring.
- Re-export it from `src/geonnax/__init__.py` (import from `geonnax.layers`
  and `__all__`) unless it is a submodule-only helper (as `dwt` / `idwt`
  are). Ruff (`RUF022`) keeps both `__all__` lists sorted.
- Add `::: geonnax.layers.Name` under the right heading of
  `docs/api/layers.md` (convs, blocks, norms, attention) or
  `docs/api/operators.md` (spectral, spherical, wavelet layers).
- `make capabilities`.

## 5. Tests

In the file for its family (`tests/test_unet.py` for convs, blocks, norms and
attention; `tests/test_spectral.py`, `test_wavelet.py`,
`test_wavelet_attention.py`, `test_spherical_wavelet.py` for the transforms):

- output shape on a `(C, *spatial)` input, and the validation errors
  (`pytest.raises(ValueError, match=...)`);
- dimension flexibility:
  `@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])`;
- a property against a reference: exact reconstruction for a transform
  pair, resolution invariance for an FFT layer, equivalence to a dense /
  unfactorised form, identity at init where the design promises it;
- `@pytest.mark.integration`: `eqx.filter_jit`, `jax.vmap` over a batch and
  finite `eqx.filter_grad` (the `_grads_finite` helper in
  `tests/test_spectral.py`).

## 6. Verify

```bash
uv run pytest -o addopts= -n 2 tests/test_unet.py   # or the family's test file: every tier
uv run pytest -o addopts= --doctest-modules src/geonnax/layers
uv run pytest -o addopts= tests/test_capabilities.py tests/test_docstrings.py tests/test_docstrings_render.py
```

then the `pre-pr-check` skill.
