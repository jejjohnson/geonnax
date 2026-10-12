"""Grid-resolution invariance of the operators, via ``tests/_resolution_invariance.py``.

FNO and WNO read the grid only at call time and must pass; ``UNet`` / ``XUNet``
(fixed-stencil CNNs) do not, which is pinned here as a strict ``xfail`` until
``CNO`` (#33) lands. SFNO is omitted: its spherical-harmonic transform is
built for one fixed ``(n_lat, n_lon)`` grid, so it cannot be evaluated at two
resolutions with the same module.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest
from _resolution_invariance import (
    band_limited_field,
    resolution_gap,
    spectral_pair,
    spectral_resize,
)

from geonnax.fno import FNO
from geonnax.unet import UNet
from geonnax.wno import WNO


KEY = jr.PRNGKey(0)
LO = (32, 32)
UP, DOWN = spectral_pair(LO)  # 32^2 <-> 64^2

# Tolerances sit ~10x above the gaps measured over three input keys at random
# init (FNO <= 4.4e-3, WNO <= 2.0e-2; both from GELU's weak non-linearity on a
# smooth field) and far below the U-Net gap (~1.0, i.e. an entirely different
# function). FNO uses the issue's 5e-2; WNO gets a looser 1e-1 because its
# wavelet filters are tied to the grid spacing, so it is only approximately
# invariant (measured ~5x the FNO gap).
FNO_TOL = 5e-2
WNO_TOL = 1e-1


def _x(seed: int = 0):
    return band_limited_field(LO, key=jr.PRNGKey(seed))


# ---- harness self-checks ------------------------------------------------------


def test_spectral_resize_roundtrip_is_exact_on_band_limited_field():
    x = _x()
    assert jnp.allclose(DOWN(UP(x)), x, atol=1e-5)  # float32 FFT round-off
    assert UP(x).shape == (1, 64, 64)


def test_resolution_gap_identity_is_zero():
    assert resolution_gap(lambda z: z, _x(), UP, DOWN) < 1e-5


def test_resolution_gap_flags_a_resolution_bound_operator():
    # Negative control: a fixed one-pixel shift is 1/32 of the domain on the
    # coarse grid but 1/64 on the fine one, so the harness must report a clear
    # gap (measured 0.124 on this input; threshold at 0.05, ~40x the 1e-5 floor
    # of an exact operator).
    assert resolution_gap(lambda z: jnp.roll(z, 1, -1), _x(), UP, DOWN) > 0.05


def test_spectral_resize_downsample_shape():
    assert spectral_resize(_x(), (16, 16)).shape == (1, 16, 16)


# ---- operators that must be invariant ----------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_fno_resolution_invariant(seed):
    fno = FNO.init(1, 1, (6, 6), key=KEY, hidden_channels=16, n_layers=2)
    assert resolution_gap(fno, _x(seed), UP, DOWN) < FNO_TOL


@pytest.mark.integration
def test_wno_resolution_invariant():
    wno = WNO.init(1, 1, 2, key=KEY, hidden_channels=16, n_layers=2)
    assert resolution_gap(wno, _x(), UP, DOWN) < WNO_TOL


# ---- the documented gap -------------------------------------------------------


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="UNet/XUNet use a fixed 3x3 stencil and aliasing activations, so "
    "they are not discretisation-invariant; expected to pass once CNO (#33) lands.",
)
def test_unet_not_resolution_invariant():
    # ``XUNet`` is an alias of ``UNet``, so this covers both.
    unet = UNet.init(8, key=KEY, channels=1, dim_mults=(1, 2))
    assert resolution_gap(unet, _x(), UP, DOWN) < FNO_TOL
