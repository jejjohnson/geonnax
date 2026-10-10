---
applyTo: "src/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `equinox.Module` for anything that holds arrays (immutable pytrees; a dataclass is not one), with configuration in `eqx.field(static=True)`; `NamedTuple` for small static records
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

No new runtime dependency without discussion; build on what geonnax already
depends on (see "Boundaries" in `AGENTS.md`). geonnax never imports numpyro.

| Purpose | Preferred Package |
|---------|-------------------|
| Modules / pytrees, standard layers | `equinox` (`eqx.nn.Conv`, `Linear`, `GroupNorm`, `Dropout`) |
| Arrays, randomness, transforms | `jax` (`jax.random` with explicit keys) |
| Axis-naming array ops | `einx` |
| Shape annotations | `jaxtyping` |
| Construction-time tables (eager only) | `numpy` |
| Path handling | `pathlib` (stdlib) |
| Testing | `pytest` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Public classes and functions include a plural `Examples:` section with `>>>` doctests (run by `make doctest`; print shapes or rounded values); math in `$…$`, never Sphinx / RST markup (`tests/test_docstrings.py`)
