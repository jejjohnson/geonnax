"""Generate, or check, the capability index: every public name in geonnax.

``docs/api/capabilities.md`` lists each public object of ``geonnax`` and of its
public submodules once, at its public home, grouped by the module that defines
it, with the first sentence of its docstring. Names reachable only from a
submodule (the bases in ``geonnax.basis``, the lon/lat helpers in
``geonnax.geo``, ``geonnax.layers.dwt``, …) are listed under that submodule.
It then lists the shared private helpers contributors reuse. Agents and people
search it before writing a helper ("Reuse before you write" in ``AGENTS.md``).

geonnax builds on no other GeoML-stack library (only jax, equinox, einx and
jaxtyping), so the index has no upstream section.

Usage::

    make capabilities                                   # rewrite the index
    uv run python scripts/capabilities.py --check       # fail if stale

``tests/test_capabilities.py`` runs the check in the fast tier.

``--check`` also fails when two geonnax homes export the same name for
different objects (unless ``ALLOWED_SHARED_NAMES`` gives the reason), and when
a helper listed in ``PRIVATE_HELPERS`` no longer exists.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
import types
import typing
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "api" / "capabilities.md"

# Public homes, in order. A name is listed at the first home that exports it,
# so the submodules contribute only what the top level does not re-export.
PUBLIC_MODULES: list[str] = [
    "geonnax",
    "geonnax.basis",
    "geonnax.geo",
    "geonnax.layers",
    "geonnax.conditioning",
    "geonnax.crf",
    "geonnax.encoders",
    "geonnax.ensemble",
    "geonnax.fno",
    "geonnax.heteroscedastic",
    "geonnax.mfn",
    "geonnax.mixture",
    "geonnax.mswt",
    "geonnax.multi_scale_siren",
    "geonnax.ncp",
    "geonnax.randfeat",
    "geonnax.sfno",
    "geonnax.siren",
    "geonnax.slepian",
    "geonnax.sngp",
    "geonnax.spectral_norm",
    "geonnax.unet",
    "geonnax.vssgp",
    "geonnax.wno",
]

# Names two homes may bind to different objects, and why. None today.
ALLOWED_SHARED_NAMES: dict[str, str] = {}

# Shared private helpers: plumbing several modules use (or should use). Not
# public API, so downstream code must not import them; inside geonnax, reuse
# them instead of writing another copy. ``--check`` fails if one disappears.
PRIVATE_HELPERS: dict[str, str] = {
    "geonnax.layers._utils.act": "SiLU, the default nonlinearity of the layers",
    "geonnax.layers._utils.group_count": (
        "a valid GroupNorm group count for any channel width (gcd-clamped)"
    ),
    "geonnax.layers._utils.shuffle_patterns": (
        "einx patterns for the factor-2 space-to-depth / depth-to-space shuffle"
        " in any number of spatial dims"
    ),
    "geonnax.layers._utils.shuffle_kwargs": "patch sizes for `shuffle_patterns`",
    "geonnax.layers._blocks._dropout_layer": (
        "an `eqx.nn.Dropout`, or `None` when the rate is 0"
    ),
    "geonnax.unet._split_keys": (
        "split an optional key into n subkeys, or n `None`s (dropout threading)"
    ),
    "geonnax.layers._factorized._resolve_rank": (
        "a fractional or absolute rank resolved against the full size"
    ),
    "geonnax.layers._spectral._corner_slices": (
        "the corner slices of the retained Fourier modes along each axis"
    ),
    "geonnax.layers._wavelet._filter_bank": (
        "the orthonormal analysis filters (low, high) of a `Wavelet`"
    ),
    "geonnax.fno._pointwise": "a 1x1 (pointwise) `eqx.nn.Conv` channel mix",
    "geonnax.ensemble._glorot_uniform": "Glorot-uniform `(D_in, D_out)` init",
    "geonnax.sngp._glorot_normal": "Glorot-normal `(F_in, F_out)` init",
    "geonnax.siren._require_positive": (
        "raise `ValueError` if any keyword value is not positive"
    ),
    "geonnax.spectral_norm._l2_normalize": (
        "`x / max(‖x‖, eps)`, finite gradients at 0 (power iteration)"
    ),
    "geonnax._basis._rbf._safe_sqrt": (
        "double-`where` sqrt: exact 0 at 0 with finite gradients"
    ),
    "geonnax._basis._rbf._safe_arccos": (
        "double-`where` arccos: finite gradients at c = ±1 (geodesic distance)"
    ),
    "geonnax.geo._validate_lonlat_shape": "require an `(N, 2)` lon/lat array",
    "geonnax.geo._validate_range": "require `min < max` for a range argument",
    "geonnax.geo._validate_input_unit": "require `'degrees'` or `'radians'`",
    "geonnax.geo._promote_to_floating": (
        "promote integer arrays to float before affine maps"
    ),
    "geonnax.encoders._as_scalar": (
        "coerce a scalar argument to a float 0-d array, rejecting batches"
    ),
}

_AUTOREF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# A full stop that ends a sentence (not "e.g." / "i.e." / "etc." / "al.").
_SENTENCE_END = re.compile(r"(?<!e\.g)(?<!i\.e)(?<!etc)(?<! al)\.\s+(?=[A-Z])")


def _first_line(text: str) -> str:
    """The docstring's summary: its first sentence, on one line."""
    paragraph = text.strip().split("\n\n")[0]
    line = " ".join(part.strip() for part in paragraph.splitlines())
    # mkdocs-autorefs links resolve only in the docs that wrote them.
    line = _AUTOREF.sub(r"\1", line)
    sentence = _SENTENCE_END.split(line, maxsplit=1)[0]
    if sentence != line:
        line = sentence + "."
    if len(line) > 160:
        line = line[:157].rstrip() + "..."
    return _cell(line)


def _cell(text: str) -> str:
    """Make ``text`` safe inside a Markdown table cell.

    A LaTeX norm ``\\|x\\|`` becomes ``‖x‖``; any other pipe is escaped.
    """
    return text.replace("\\|", "‖").replace("|", "\\|")


def _is_type_alias(obj: object) -> bool:
    return typing.get_origin(obj) is not None


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if _is_type_alias(obj):
        return "type"
    if callable(obj):
        return "function"
    return "constant"


def _origin(obj: object, name: str) -> str:
    """The module that defines ``obj``.

    Classes and functions carry ``__module__``. A type alias (a ``Literal``)
    does not, so take the deepest geonnax module that binds it under ``name``.
    """
    if inspect.isclass(obj) or (callable(obj) and not _is_type_alias(obj)):
        return obj.__module__
    homes = [
        mod_name
        for mod_name, mod in sys.modules.items()
        if mod_name.startswith("geonnax")
        and getattr(mod, name, None) is obj
        and mod_name != "geonnax"
    ]
    return max(homes, key=lambda m: m.count("."), default="geonnax")


def _summary(obj: object) -> str:
    if _is_type_alias(obj):
        return _cell(f"`{obj!r}`")
    if not callable(obj):
        value = repr(obj)
        return _cell(f"`{value if len(value) <= 60 else value[:57] + '...'}`")
    return _first_line(inspect.getdoc(obj) or "")


def _public(module: types.ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return [
        n
        for n in names
        if not n.startswith("__")
        and not isinstance(getattr(module, n, None), types.ModuleType)
    ]


def _module_title(name: str) -> str:
    module = sys.modules.get(name) or importlib.import_module(name)
    return _first_line(module.__doc__ or "")


Row = tuple[str, str, str, str]  # name, kind, summary, defining module


def collect() -> tuple[dict[str, list[Row]], list[str]]:
    """``{public home: [rows]}`` and the names bound to two objects."""
    index: dict[str, list[Row]] = defaultdict(list)
    owners: dict[str, dict[str, object]] = defaultdict(dict)
    # id -> the first public name bound to that object (aliases share an id).
    first_name: dict[int, str] = {}
    listed: set[tuple[int, str]] = set()
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in _public(module):
            obj = getattr(module, name)
            owners[name][f"{module_name}.{name}"] = obj
            if (id(obj), name) in listed:
                continue  # re-exported: listed at its first home
            listed.add((id(obj), name))
            canonical = first_name.setdefault(id(obj), name)
            summary = f"Alias of `{canonical}`." if canonical != name else _summary(obj)
            index[module_name].append(
                (name, _kind(obj), summary, _origin(obj, canonical))
            )
    clashes = []
    for name, homes in sorted(owners.items()):
        if name in ALLOWED_SHARED_NAMES:
            continue
        if len({id(o) for o in homes.values()}) > 1:
            clashes.append(f"{name}: {', '.join(sorted(homes))}")
    return index, clashes


def missing_helpers() -> list[str]:
    """Entries of ``PRIVATE_HELPERS`` that no longer resolve."""
    missing = []
    for path in PRIVATE_HELPERS:
        module_name, _, attr = path.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(path)
            continue
        if not hasattr(module, attr):
            missing.append(path)
    return missing


def _table(rows: list[Row]) -> list[str]:
    out = ["| Name | Kind | What it does |", "|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} |" for r in rows]
    return [*out, ""]


def _public_part() -> tuple[list[str], int]:
    index, _ = collect()
    out: list[str] = []
    total = 0
    for module_name in PUBLIC_MODULES:
        rows = index.get(module_name, [])
        if not rows:
            continue
        total += len(rows)
        out += [f"## `{module_name}`", ""]
        # Group by the module that defines each name: one concept per group.
        groups: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            groups[row[3]].append(row)
        if len(groups) == 1:
            out += _table(sorted(rows))
            continue
        for origin in sorted(groups):
            title = f"{_module_title(origin)} (`{origin}`)"
            out += [f"### {title}", "", *_table(sorted(groups[origin]))]
    return out, total


def _helpers_part() -> list[str]:
    out = [
        "## Shared private helpers",
        "",
        "Not public API: downstream code must not import these. Inside geonnax,",
        "reuse them instead of writing another copy.",
        "",
        "| Helper | What it does |",
        "|---|---|",
    ]
    out += [f"| `{path}` | {_cell(what)} |" for path, what in PRIVATE_HELPERS.items()]
    return [*out, ""]


def render() -> str:
    public_part, total = _public_part()
    out = [
        "# Capability index",
        "",
        "<!-- Generated by scripts/capabilities.py — do not edit by hand. -->",
        "",
        f"Every public name in geonnax ({total} of them), each listed once at its",
        "public home and grouped by the module that defines it, with the first",
        "sentence of its docstring; names reachable only from a submodule are",
        "listed under it. Then the shared private helpers. Search this page",
        "before writing a helper: if what you need is here, compose it; if it is",
        "almost here, extend it where it lives. Regenerate with",
        "`make capabilities` after changing a public API",
        "(`tests/test_capabilities.py` checks it is current).",
        "",
        *public_part,
        *_helpers_part(),
    ]
    return "\n".join(out).rstrip() + "\n"


def stale(current: str, text: str) -> bool:
    """Whether the committed index ``current`` differs from ``text``."""
    return current != text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Capability index.").splitlines()[0]
    )
    parser.add_argument("--check", action="store_true", help="fail if stale")
    args = parser.parse_args()
    text = render()
    status = 0
    _, clashes = collect()
    if clashes:
        print("Names bound to different objects in two homes:")
        print("\n".join(f"  {c}" for c in clashes))
        print("Rename one, or add it to ALLOWED_SHARED_NAMES with a reason.")
        status = 1
    missing = missing_helpers()
    if missing:
        print("PRIVATE_HELPERS entries that no longer exist:")
        print("\n".join(f"  {m}" for m in missing))
        status = 1
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if stale(current, text):
            print(f"{INDEX.relative_to(ROOT)} is stale; run `make capabilities`")
            status = 1
    else:
        INDEX.write_text(text, encoding="utf-8")
        print(f"wrote {INDEX.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
