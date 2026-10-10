# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. geonnax is a zoo of deterministic Equinox network cores and bases
that pyrox and kernellib build on: most defects worth finding are about
**modules and numerics** (an array in a static field, a reused key, a batch
axis baked into a layer, an init scale off by ω, a NaN gradient at a
coincident point, a grid that under-resolves the band limit) or
**boundaries** (a numpyro import, a re-implemented layer or basis, a renamed
field pyrox swaps), not style. Read "Boundaries", "Reuse before you write"
and "The contracts" in [`AGENTS.md`](AGENTS.md) first; this file is the
checklist and the report format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, `__all__` ordering, RST markup in docstrings); review what they cannot
see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against
  [`docs/api/capabilities.md`](docs/api/capabilities.md) and its shared
  private helpers. A re-implemented layer, attention block, spectral or
  wavelet transform, basis, encoder, init rule, random-feature map or helper
  (Glorot init, safe `sqrt`, shuffle pattern, key splitting) is a **High**
  finding, with the existing name to use.
- No `numpyro` (or any PPL) anywhere: priors, sample sites and KL terms
  belong in pyrox; kernels and spectral densities in kernellib.
- Imports keep pointing one way (`_basis` ← `geo` / `layers` ← models); a
  new model-to-model import is justified.
- No new runtime dependency; NumPy only for eager construction-time tables.

### 2. Modules and `init`

- `eqx.Module` with array leaves and configuration in
  `eqx.field(static=True)`; no array in a static field, no dataclass or
  mutable container holding arrays.
- A classmethod `init(..., *, key, ...)` that validates its arguments with a
  `ValueError` naming the bad value, splits the key once per draw, and
  returns `cls(...)`; parameter-free modules take no key.
- One example per call (`(D,)` or `(C, *spatial)`), batched by `jax.vmap`;
  no batch axis baked into `__call__`.
- Dimension-flexible layers take `num_spatial_dims` (or `len(n_modes)`) and
  are tested in 1-D, 2-D and 3-D (3-D marked `slow`).
- State updates return a new module (`eqx.tree_at`); frozen arrays are
  leaves read through `jax.lax.stop_gradient`.
- Stochastic forward passes take an explicit `key` (optional where the
  randomness is, like dropout, and then `eqx.nn.inference_mode` works).

### 3. Bases

- Evaluation returns `Φ (N, M)` from `N` points, plus eigenvalues, the
  spectrum or per-atom geometry — the basis contract in `docs/api/bases.md`.
- Implemented in `_basis/_<name>.py`, re-exported from `_basis/__init__.py`
  and `geonnax.basis` (`__all__`); no PRNG, no kernel, no `eqx.Module`
  unless it caches a decomposition.
- Orthonormality, eigen-equations or exact reconstruction checked against a
  closed form in the tests.

### 4. Numerics and initialisation

- The published init (SIREN regimes via `siren_W_limit`, Glorot, small
  head scales) with a citation; any change to an init scale is justified
  with the variance argument.
- Dtypes follow the input (`jnp.zeros(n, dtype=x.dtype)`); integer
  coordinates promoted before affine maps.
- No Python `if` / `float()` / `.item()` on traced values; shapes depend only
  on static fields and input shapes.
- Finite gradients at singular points (`_safe_sqrt`, `_safe_arccos`,
  `_l2_normalize`, log-space recursions); symmetrise and jitter before a
  Cholesky.
- Spectral layers keep `n_modes` below half of each extent; grid-bound
  tables (SHT) validate the grid in `init`.
- einx for named-axis contractions, transposes, reshapes and inserted-axis
  broadcasts in new code (house style; not lint-enforced).

### 5. Public API and downstream compatibility

- New names in the module's `__all__` and, unless submodule-only, in
  `src/geonnax/__init__.py`; a `::: geonnax.<module>.<Name>` entry on pages
  that list classes one by one; `docs/api/capabilities.md` regenerated.
- A rename or removal of a public name, constructor argument or parameter
  field (`W`, `b`, `proj`, `generator`, `Omega`, `layers`, …) keeps the old
  spelling working with a `DeprecationWarning` naming the replacement:
  pyrox and kernellib pin geonnax by tag and use them. Missing this is
  **Critical** for a field pyrox swaps, **High** otherwise.
- Docstrings: Google style, the formula, shapes, a reference for a
  published method, and a plural `Examples:` section whose `>>>` lines pass
  under `make doctest` (CI does not run it).

### 6. Tests

- Against a closed form, a brute-force reference (as the CRF tests do) or a
  published value; shapes alone are not enough for numerics.
- `jit`, `vmap` and `grad` exercised for a new model (`integration`), finite
  gradients at the singular points.
- Incidental randomness pinned (`jr.PRNGKey(0)`); sampling behaviour bounded
  by its own distribution, with the bound's source in a comment.
- Tier markers right: unmarked for fast checks, `slow` for heavy compilation
  or broad sweeps, `integration` for end-to-end `jit` / `vmap` / `grad`.

### 7. Modern Python

- Type hints on every public function; `X | None`; `from __future__ import
  annotations`; specific exceptions with `raise ... from ...`; guard clauses
  over deep nesting.

---

## geonnax-Specific Checks

### Arrays are leaves, configuration is static

```python
# ❌ An array in a static field: equinox warns, and eqx.filter / jax.grad
#    never see it, so it is silently never trained
class Scale(eqx.Module):
    weight: Array = eqx.field(static=True)


# ✅ Array leaf, static configuration, built by init(..., *, key)
class Scale(eqx.Module):
    weight: Float[Array, " D"]
    features: int = eqx.field(static=True)

    @classmethod
    def init(cls, features: int, *, key: Array) -> Scale:
        if features <= 0:
            raise ValueError(f"features must be > 0, got {features}.")
        return cls(weight=1.0 + 0.1 * jr.normal(key, (features,)), features=features)
```

### One key per draw

```python
# ❌ The same key twice: b equals the first row of W
W = jr.normal(key, (4, 8))
b = jr.normal(key, (8,))

# ✅
k_w, k_b = jr.split(key)
W, b = jr.normal(k_w, (4, 8)), jr.normal(k_b, (8,))
```

### One example per call

```python
# ❌ A batch axis baked into the layer
def __call__(self, x: Float[Array, "B D_in"]) -> Float[Array, "B D_out"]:
    return x @ self.W + self.b


# ✅ Per example; callers batch with jax.vmap(layer)(xs)
def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, " D_out"]:
    return einx.dot("i, i o -> o", x, self.W) + self.b
```

### Finite gradients at coincident points

```python
# ❌ d/dx sqrt(‖x − c‖²) is NaN at x = c
r = jnp.sqrt(jnp.sum((x - c) ** 2))

# ✅ Same value everywhere, zero gradient at x = c
from geonnax._basis._rbf import _safe_sqrt

r = _safe_sqrt(jnp.sum((x - c) ** 2))
```

### State updates return a new module

```python
# ❌ eqx.Module is frozen: this raises FrozenInstanceError
self.state = self.state + x


# ✅
def update(self, x: Float[Array, " D"]) -> Counter:
    return eqx.tree_at(lambda m: m.state, self, self.state + x)
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
