---
name: add-basis
description: Route a new basis, eigenfunction family, frame, sphere grid or feature transform to functax, where geonnax's bases moved in jejjohnson/geonnax#153, and wire it into a geonnax network afterwards. Use when asked to add, port or implement a basis, eigenfunctions, frame, dictionary, positional / spectral features or inducing-feature geometry while working in geonnax.
---

# Add a basis

geonnax no longer holds bases. Since jejjohnson/geonnax#153 the bases, sphere grids, frames, field derivatives and transforms live in functax, and geonnax imports them. A new basis therefore goes to functax, and geonnax only consumes it.

## 1. Make sure it does not exist yet

- Search functax's capability index (`docs/api/capabilities.md` in jejjohnson/functax) and its "Reuse before you write" table in `AGENTS.md`. The families are `functax.basis.fourier`, `vector`, `sphere`, `slepian`, `local`, `eigen`, `polynomial` and `features`, plus `functax.geometry` for grids and `functax.quad` for rules.
- A kernel spectral density, a prior variance or a random-feature draw is not a basis: it lives in kernellib or pyrox.
- A change of domain (a different `L`, a cap radius) is an argument, not a new basis.

## 2. Add it in functax

Open an issue or branch in jejjohnson/functax and follow its `AGENTS.md` and its `pre-pr-check` skill. The new basis goes into the family submodule that owns it, never into a flat `functax.basis`.

## 3. Use it from geonnax

- Bump the functax pin in `pyproject.toml` (`functax[jax] @ git+https://github.com/jejjohnson/functax.git@vX.Y.Z`) to the release that contains it, and run `uv lock`.
- Import it from its functax family module (`from functax.basis.sphere import real_spherical_harmonics`), never through `geonnax.basis`, whose moved names are deprecated aliases.
- The feature-engineering helpers that stay in geonnax (`interaction_features`, `standardize`, `unstandardize`) live in `src/geonnax/basis.py`. A new helper of that kind, one that is not an expansion, may go there.

## 4. Verify

Run the tests of the network that uses the basis, then the `pre-pr-check` skill.
