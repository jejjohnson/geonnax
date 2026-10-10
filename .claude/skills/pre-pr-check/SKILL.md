---
name: pre-pr-check
description: Run geonnax's full pre-PR verification — ruff lint and format on the whole repo, ty on src/geonnax, the fast test tier as CI runs it, the slow and integration tiers and doctests of what changed (no workflow runs them), the capability index, the lockfile and the strict MkDocs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run --group lint ruff check .          # entire repo: src/, tests/, scripts/, notebook cells
uv run --group lint ruff format --check .
make typecheck                            # ty check src/geonnax (what CI runs)
make test-fast                            # -m "not slow and not integration" -n auto, as CI
```

CI runs the fast tier with coverage on Python 3.12 and 3.13
(`uv run pytest -m "not slow and not integration" -n auto`); `make test-fast`
is the same selection without coverage. On a machine short of memory, lower
the worker count (`-n 2`): JAX compilation in each xdist worker takes
gigabytes.

## The tiers and doctests of what you touched

No workflow runs the `slow` and `integration` tiers or the doctests, so this
is the only place they run:

```bash
uv run pytest -o addopts= -n 2 -m "slow or integration" tests/test_<area>.py
uv run pytest -o addopts= --doctest-modules src/geonnax/<module>.py   # or the package dir
```

`make test` runs every tier of the whole suite; `make doctest` every
docstring example. A layer or operator change runs `tests/test_unet.py`,
`tests/test_spectral.py` and the wavelet tests; a basis change
`tests/test_new_bases.py` and `tests/test_localized_bases.py`; an
uncertainty core `tests/test_geonnax.py`, `test_heads.py`,
`test_spectral_norm.py`.

## When the public API changed

```bash
make capabilities
uv run pytest -o addopts= tests/test_capabilities.py tests/test_geonnax.py
```

…plus the `::: geonnax.<module>.<Name>` entry on pages that list classes one
by one (`docs/api/layers.md`, `operators.md`, `unet.md`). A rename or removal
of a public name, constructor argument or parameter field keeps the old
spelling with a `DeprecationWarning` (pyrox and kernellib pin geonnax by tag
and use them); say so in the PR.

## When dependencies changed

`uv lock` and commit `uv.lock`; no new runtime dependency without
discussion, and never numpyro (`test_no_numpyro_import_in_package`).

## When docs or docstrings changed

- `uv run pytest -o addopts= tests/test_docstrings.py tests/test_docstrings_render.py`
  (no RST markup; plural `Examples:` with a `>>>` or fenced body) and the
  doctests of the modules you touched.
- `uv run --group docs mkdocs build --strict` (no workflow builds the docs on
  a PR; `pages.yml` deploys from `main`). If the build stalls at
  mkdocs-jupyter's `files` step, a system `pandoc` is converting every
  Markdown page through jupytext (very slow); hide it for the build:

  ```bash
  mkdir -p /tmp/nopandoc && printf '#!/bin/sh\nexit 1\n' > /tmp/nopandoc/pandoc && chmod +x /tmp/nopandoc/pandoc
  PATH=/tmp/nopandoc:$PATH uv run --group docs mkdocs build --strict
  ```

## Before pushing

- `git status` shows no stray files (`.plans/`, notebook `.py` drafts,
  `site/`, `coverage.xml`).
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
