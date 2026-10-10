"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/geonnax/skills/build-models-with-geonnax/SKILL.md`` is what agents
in downstream projects read before building networks on geonnax, so a stale
name or a broken example there teaches them the wrong API. These tests run
its worked example and check that every ``geonnax.X`` / ``geonnax.basis.X`` /
``geonnax.geo.X`` / ``geonnax.layers.X`` it and the plugin's reviewer name is
a current public name.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "geonnax"
SKILL = PLUGIN / "skills" / "build-models-with-geonnax" / "SKILL.md"
AGENT = PLUGIN / "agents" / "geonnax-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the repository root.
    pytest.skip("the plugin is not present", allow_module_level=True)

# Longest module first, so ``geonnax.basis.X`` is not read as ``geonnax.basis``.
_NAME = re.compile(
    r"(?<![\w./])(geonnax\.basis|geonnax\.geo|geonnax\.layers|geonnax)"
    r"\.([A-Za-z_]\w*)"
)
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.slow
def test_worked_example_runs_and_its_claims_hold():
    (block,) = [b for b in _BLOCKS.findall(SKILL.read_text()) if "fourier_basis(" in b]
    ns: dict = {}
    exec(block, ns)
    assert ns["y_64"].shape == (1, 64, 64)
    assert ns["y_128"].shape == (1, 128, 128)
    assert ns["batch"].shape == (4, 1, 64, 64)
    # "agree to about 1 %": measured 0.012. The bound is the one
    # test_spectral_conv_resolution_invariance uses for the same check — the
    # linear resize from 128 to 64 points adds interpolation error on top of
    # the operator's own discretisation error.
    assert float(ns["resolution_gap"]) < 0.05
    assert ns["sh"].shape == (3, 25)  # (l_max + 1)² = 25 harmonics per point
    assert ns["Phi"].shape == (2001, 8)
    assert ns["lam"].shape == (8,)
    # "orthonormal on the grid to within 10⁻⁶": the Dirichlet modes vanish
    # at ±L, so the Riemann sum is exact up to float32 rounding (measured
    # 3.6e-7); 1e-5 leaves room for a different summation order.
    assert float(ns["basis_gap"]) < 1e-5
    # Each off-diagonal entry of F Fᵀ is the mean of R = 1024 cosines
    # cos(wᵀ(x − y)/ℓ), whose variance is at most 1/2 (and lower for
    # orthogonal draws), so its sd is at most sqrt(1 / 2R) ≈ 0.022; the
    # maximum over the 4,950 pairs stays within about 5 sd. The diagonal is
    # exact (cos 0 = 1). The prose says "about 0.06" (measured 0.059).
    assert float(ns["feature_gap"]) < 5 * (1 / (2 * 1024)) ** 0.5
    # 50 power iterations converge for a 32 × 32 Gaussian weight (the gap
    # closes as (σ₂/σ₁)^(2k)), so the rescaled weight's top singular value
    # is the requested coefficient.
    assert float(ns["sigma"]) == pytest.approx(0.95, rel=1e-3)
    # "about ten times": far from the cluster the features are nearly
    # orthogonal to every training feature, so var ≈ ‖φ‖² / ridge ≈ 10,
    # while at the centre the precision is dominated by the data (measured
    # 9.7 vs 1.1).
    assert float(ns["var_far"]) > 5 * float(ns["var_near"])


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    stale = []
    for module_name, attr in set(_NAME.findall(path.read_text())):
        module = importlib.import_module(module_name)
        if attr in getattr(module, "__all__", dir(module)):
            continue
        try:  # a submodule path such as ``geonnax.layers``
            importlib.import_module(f"{module_name}.{attr}")
        except ImportError:
            stale.append(f"{module_name}.{attr}")
    assert not stale, f"{path.name} names objects that do not exist: {sorted(stale)}"
