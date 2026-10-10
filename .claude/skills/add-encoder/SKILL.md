---
name: add-encoder
description: Add a coordinate encoder or geographic helper to geonnax — a pure lon/lat function in geonnax.geo on (N, 2) arrays plus its parameter-free single-example eqx.Module wrapper in geonnax.encoders (or a Slepian / spherical encoder in geonnax.slepian), with lon-then-lat ordering, explicit units, validation, docs, index and tests. Use when asked to add, port or implement a positional / coordinate / lon-lat / time / context encoding.
---

# Add a coordinate encoder

Read "1. Modules" in `AGENTS.md` first; this is the step-by-step. pyrox-nn
re-exports the encoders and `geonnax.geo` helpers, so their names and
argument conventions are its API.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (`geonnax.geo`, `geonnax.encoders`,
  `geonnax.slepian`): `deg2rad`, `lonlat_scale`, `lonlat_to_cartesian3d`,
  `cyclic_encode`, `spherical_harmonic_encode`; the modules `Deg2Rad`,
  `LonLatScale`, `Cartesian3DEncoder`, `CyclicEncoder`,
  `SphericalHarmonicEncoder`, `GeoContextEncoder` (lon/lat + day of year +
  covariates in one vector), `SlepianEncoder`,
  `HybridSphericalSlepianEncoder`.
- Composing two encoders (`eqx.nn.Sequential`, or concatenating outputs) is
  not a new encoder; a new block of `GeoContextEncoder` is an option there.
- A basis evaluated at the coordinates (RBF, Gabor, eigenfunctions) is the
  `add-basis` skill; a learnable embedding is a layer or a model.

## 2. Write the function (`src/geonnax/geo.py`)

- `def name(lonlat: Num[Array, "N 2"], *, ...) -> Float[Array, "N F"]`:
  batched over `N` rows, columns **`[lon, lat]`** in that order.
- Units are explicit: an `input_unit: Literal["degrees", "radians"]`
  keyword checked with `_validate_input_unit`, or documented degrees with a
  range argument checked with `_validate_range`; `(N, 2)` shapes with
  `_validate_lonlat_shape`; integer inputs through `_promote_to_floating`.
- Pure `jnp`, einx for broadcasts; no clipping of out-of-range values
  (shape-only validation keeps it `jit`-safe) — say so in the docstring.
- Spherical features reuse `geonnax._basis.real_spherical_harmonics`; never
  re-derive harmonics.

## 3. Write the module (`src/geonnax/encoders.py`)

- `class NameEncoder(eqx.Module)`: configuration only, as
  `eqx.field(static=True, default=...)`; no learnable parameters, no key,
  constructed directly (`NameEncoder()`), validated in `__post_init__` (as
  `Cartesian3DEncoder` does) or `__check_init__`.
- `__call__` on **one example** (`(2,)` lon/lat, a scalar, or `(D,)`):
  lift to `(1, …)`, call the `geo` function, drop the row (see
  `Cartesian3DEncoder.__call__`); scalars through `_as_scalar`. Batch with
  `jax.vmap`.
- If the output width depends on configuration, expose it (as
  `SphericalHarmonicEncoder.num_features` / `GeoContextEncoder.output_dim`).
- Docstrings: the feature formula in `$…$`, the units, the output width, and
  a plural `Examples:` section (single example and `jax.vmap`).

## 4. Export, document, index

- `__all__` of `geo.py` (functions; they stay submodule-only) and of
  `encoders.py`; re-export the module class from `src/geonnax/__init__.py`
  (import and `__all__`).
- `docs/api/encoders.md` renders `::: geonnax.encoders`, `::: geonnax.slepian`
  and `::: geonnax.geo` whole; add a sentence to the matching section.
- `make capabilities`.

## 5. Tests

`tests/test_geonnax.py` (the "Tier A: deterministic encoders" section) or
`tests/test_geo_context_encoder.py` for context-vector blocks:

- known values (`(lon, lat) = (0, 0)` → `+x`; poles; the dateline; a
  period wrap) and unit norms where promised;
- the module agrees with the function (`jax.vmap(module)(lonlat)` equals
  `geo.name(lonlat)`), degrees and radians agree after conversion;
- validation errors; integer inputs promoted;
- the output width equals the documented / exposed width.

## 6. Verify

```bash
uv run pytest -o addopts= tests/test_geonnax.py tests/test_geo_context_encoder.py
uv run pytest -o addopts= --doctest-modules src/geonnax/geo.py src/geonnax/encoders.py src/geonnax/slepian.py
uv run pytest -o addopts= tests/test_capabilities.py tests/test_docstrings.py tests/test_docstrings_render.py
```

then the `pre-pr-check` skill.
