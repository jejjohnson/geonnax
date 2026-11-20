---
name: add-notebook
description: Add or update an executed example notebook in geonnax's docs — jupytext percent .py while drafting, an executed .ipynb as the committed source that mystmd renders without running it, a Colab badge and install cell, an entry in the toc of docs/myst.yml, ruff on the notebook cells, and a link-clean docs build. Use when asked to write, extend, re-run or fix a tutorial / example / demo notebook under docs/notebooks.
---

# Add or update a notebook

The full standard is `.github/instructions/docs-examples.instructions.md`;
read it first. This is the checklist.

## 1. Plan it

- One question per notebook (train an FNO on a toy PDE, fit a SIREN to a
  field, encode lon/lat for a downstream model, compare bases); check
  `docs/notebooks/` and the `Examples` section of the `toc` in
  `docs/myst.yml` so you extend rather than duplicate.
- Use the public API the way a user would (`import geonnax`,
  `from geonnax.basis import ...`); no private names. A training loop may use
  plain `jax.grad` + `eqx.apply_updates`; optax is not a dependency of any
  group, so don't import it unless you add it to the `docs` group (discuss
  first).

## 2. Draft in `.py`, commit the `.ipynb`

1. Write `docs/notebooks/<name>.py` in jupytext percent format (header in
   the instructions file); first markdown cell: title + Colab badge
   (`https://colab.research.google.com/github/jejjohnson/geonnax/blob/main/docs/notebooks/<name>.ipynb`);
   the install cell uses
   `geonnax @ git+https://github.com/jejjohnson/geonnax@main` (geonnax is
   not on PyPI).
2. Smoke-run: `uv run --group docs python docs/notebooks/<name>.py`.
3. Convert and execute:

   ```bash
   uv run --group docs jupytext --to notebook docs/notebooks/<name>.py
   uv run --group docs jupyter nbconvert --to notebook --execute \
     docs/notebooks/<name>.ipynb --inplace --ExecutePreprocessor.timeout=180
   ```

4. Delete the `.py`; the executed `.ipynb` is the source of truth. To change
   it later, regenerate the `.py` (`uv run --group docs jupytext --to py:percent docs/notebooks/<name>.ipynb`),
   edit, re-convert and **re-execute**; never hand-edit committed cells.

## 3. Content rules

- Figures inline with `plt.show()`: no `savefig`, no committed PNGs; keep the
  notebook small (fewer, smaller figures) — pre-commit's
  `check-added-large-files` rejects files over 500 kB by default.
- Math with `$…$` / `$$…$$` (MathJax via `pymdownx.arithmatex`); each
  markdown paragraph on one line.
- Random draws from explicit keys (`jr.key(0)`, split per use) so a re-run
  reproduces the outputs; one example per call, batch with `jax.vmap`.
- Code cells ≤ 88 characters: ruff lints the `.ipynb` cells
  (`ruff check .` includes notebooks).

## 4. Wire it in and verify

- Add `- file: notebooks/<name>.ipynb` with a short `title:` under
  `Examples` in the `toc` of `docs/myst.yml`. Its file name must be unique
  across the prose tree, because mystmd serves each page at a flat URL built
  from the file name.
- `uv run --group lint ruff format docs/notebooks/` and
  `uv run --group lint ruff check .`
- `make docs` must build and be link-clean; mystmd renders the committed
  outputs as they are, without running the notebook.
