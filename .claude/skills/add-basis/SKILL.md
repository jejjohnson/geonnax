---
name: add-basis
description: Add a basis or feature transform to geonnax — a spectral eigenbasis (with eigenvalues), a localized or overcomplete frame (with per-atom geometry), a data-driven basis (with its spectrum) or a feature transform — as a pure JAX function in src/geonnax/_basis re-exported from geonnax.basis, keeping the basis contract, with closed-form tests, docs and the capability index. Use when asked to add, port or implement a basis, eigenfunctions, frame, dictionary, positional / spectral features or inducing-feature geometry.
---

# Add a basis

Read "2. Bases" under "The contracts" in `AGENTS.md` and "The basis
contract" in `docs/api/bases.md` first; this is the step-by-step. pyrox-gp
re-exports `geonnax.basis` and kernellib builds Hilbert-space features on
`fourier_basis`, so the contract and the names are their API.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (the `geonnax.basis` section): Dirichlet
  / Fourier eigenpairs (`fourier_basis`, `fourier_basis_1d`), real spherical
  harmonics, Slepian caps, graph-Laplacian eigenpairs, divergence-free atoms,
  RBF / Wendland (Euclidean and geodesic), Gabor frames, DWT matrices, EOFs,
  and the feature transforms (`fourier_features`, `seasonal_features`,
  `gaussian_window_features`, `interaction_features`).
- A kernel spectral density, a prior variance or a random-feature draw is
  **not** a basis: it lives in kernellib / pyrox. geonnax returns geometry.
- A change of domain (a different `L`, a cap radius) is an argument, not a
  new basis.

## 2. Pick its kind

| It is… | Returns | Exemplar |
|---|---|---|
| Spectral (eigenfunctions of an operator) | `(Φ (N, M), λ (M,))` | `fourier_basis`, `divfree_basis`, `graph_laplacian_eigpairs` (eigvals first) |
| Localized / placeable | `Φ (N, M)` from the caller's `centers`, `widths` | `rbf_basis`, `spherical_rbf_basis` |
| Overcomplete frame | `(Φ, centers, scales, wavenumbers)` | `gabor_frame_grid` (and the evaluator `gabor_frame`) |
| Data-driven | `(Φ, spectrum)` | `eof_basis` |
| Feature transform | `(N, F)` features of 1-D or tabular inputs | `fourier_features`, `seasonal_features` (in `basis.py` itself) |

## 3. Write it

- Spatial bases: `src/geonnax/_basis/_<name>.py` with a module docstring
  whose first sentence names the basis (it is the capability-index group
  title), the formula and the reference. Feature transforms go in
  `src/geonnax/basis.py`.
- A pure function of arrays: inputs `x: Float[Array, "N D"]` (the rows are
  evaluation points, not a batch), keyword-only options, no PRNG, no
  `eqx.Module` (unless it caches a decomposition, like `SlepianCapBasis`).
- Validate shapes and arguments with `ValueError` (concrete Python values
  only). Work that needs concrete values (a NumPy eigendecomposition, filter
  tables) runs eagerly and the docstring says so; the evaluation is `jit` /
  `vmap` safe.
- Finite gradients at coincident points: distances through
  `_basis._rbf._safe_sqrt` / `_safe_arccos` (import them; don't copy them).
- Dtypes follow the input (`dtype=x.dtype` for constants); einx for
  tensor products and broadcasts (`einx.multiply("n a, n b -> n a b", …)`).
- Docstring: formula in `$…$` / `$$…$$`, `Args:` / `Returns:` with shapes,
  `Raises:`, a reference, and a plural `Examples:` section importing from
  `geonnax.basis` and printing shapes.

## 4. Export, document, index

- `__all__` of the new `_basis/_<name>.py`; import it in
  `_basis/__init__.py` (and its `__all__` and module-docstring list); then
  import it in `src/geonnax/basis.py` from `geonnax._basis` and add it to
  `basis.__all__`. The public home is `geonnax.basis` only (not
  `geonnax.__all__`).
- `docs/api/bases.md` renders `::: geonnax.basis` whole, so the function
  appears automatically; extend the prose section for its kind (spectral,
  localized / overcomplete, data-driven) with one sentence on what it is for.
- `make capabilities`.

## 5. Tests

In `tests/test_new_bases.py` (data-driven, vector, geodesic, wavelet bases)
or `tests/test_localized_bases.py` (RBF, Gabor, time windows), importing
from `geonnax.basis`:

- shapes and every validation error;
- the defining identity against a closed form, in float32 tolerances
  (`atol=1e-5`–`1e-6`): orthonormal columns (`Φᵀ Φ = I` for an orthonormal
  basis, a quadrature-weighted version for function bases), the
  eigen-equation or the known eigenvalues, exact reconstruction, zero
  divergence, rotation invariance, compact support;
- finite `jax.grad` at a coincident point / centre;
- `jax.jit` of the evaluation; mark anything over about a second `slow`.

## 6. Verify

```bash
uv run pytest -o addopts= tests/test_new_bases.py tests/test_localized_bases.py   # every tier
uv run pytest -o addopts= --doctest-modules src/geonnax/_basis src/geonnax/basis.py
uv run pytest -o addopts= tests/test_capabilities.py tests/test_docstrings.py tests/test_docstrings_render.py
```

then the `pre-pr-check` skill.
