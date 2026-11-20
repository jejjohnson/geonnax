"""geonnax's Earth constants equal geophysix's.

geonnax sits below geophysix in the stack and must not import it at run time
(geophysix design decision 5), so it repeats the constants. This test, the only
place geonnax touches geophysix, keeps the copies equal. It is skipped where
geophysix is not installed.
"""

from __future__ import annotations

import inspect
import math

import pytest

import geonnax as gnx
from geonnax import _compat
from geonnax._earth import EARTH_MEAN_RADIUS_M, WGS84_A, WGS84_F


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12)


def test_wgs84_constants_equal_geophysix():
    earth = pytest.importorskip("geophysix.earth")
    assert _close(WGS84_A, earth.WGS84.semi_major)
    assert _close(WGS84_F, earth.WGS84.flattening)


def test_earth_mean_radius_equals_geophysix_sphere():
    earth = pytest.importorskip("geophysix.earth")
    assert _close(EARTH_MEAN_RADIUS_M, earth.SPHERE.semi_major)
    assert _close(EARTH_MEAN_RADIUS_M, earth.WGS84.mean_radius)


def test_public_constants_are_the_ones_tested():
    assert (gnx.geo.WGS84_A, gnx.geo.WGS84_F) == (WGS84_A, WGS84_F)
    assert gnx.geo.EARTH_MEAN_RADIUS_M == EARTH_MEAN_RADIUS_M


@pytest.mark.parametrize("name", ["lonlat_to_ecef", "ecef_to_lonlat"])
def test_sphere_radius_default_equals_geophysix(name):
    earth = pytest.importorskip("geophysix.earth")
    default = inspect.signature(getattr(_compat, name)).parameters["radius"].default
    assert _close(default, earth.SPHERE.semi_major)


def test_local_tangent_plane_radius_default_is_the_mean_radius_in_km():
    earth = pytest.importorskip("geophysix.earth")
    default = (
        inspect.signature(_compat.local_tangent_plane).parameters["radius"].default
    )
    assert _close(default * 1000.0, earth.SPHERE.semi_major)
