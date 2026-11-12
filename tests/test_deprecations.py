"""The deprecated geonnax paths of the maths that moved to functax (#153).

Each moved name must warn once per access, name its functax home, and return
either the functax object itself or a shim whose result equals functax's with
geonnax's old defaults spelled out.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import functax.basis.slepian as fs
import functax.diffops as fd
import functax.geometry as fg
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax import _compat
from geonnax._deprecation import REMOVED_IN
from geonnax._earth import EARTH_MEAN_RADIUS_M, WGS84_A, WGS84_F


MODULES = [
    "geonnax",
    "geonnax.basis",
    "geonnax.diffops",
    "geonnax.geo",
    "geonnax.layers",
    "geonnax.randfeat",
]
MOVED = [
    (module, name)
    for module in MODULES
    for name in importlib.import_module(module)._MOVED
]


def _resolve(path: str) -> object:
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(("module", "name"), MOVED)
def test_moved_name_warns_and_points_at_functax(module, name):
    mod = importlib.import_module(module)
    entry = mod._MOVED[name]
    with pytest.warns(DeprecationWarning, match=rf"moved to {entry.target}") as caught:
        obj = getattr(mod, name)
    assert len(caught) == 1
    assert f"removed in geonnax {REMOVED_IN}" in str(caught[0].message)
    assert obj is entry.obj
    # an alias is the functax object; a shim lives in geonnax._compat
    assert obj is _resolve(entry.target) or obj is getattr(_compat, name)


@pytest.mark.parametrize("module", MODULES)
def test_moved_names_are_listed_but_not_exported(module):
    mod = importlib.import_module(module)
    assert set(mod._MOVED).isdisjoint(mod.__all__)
    assert set(mod._MOVED) <= set(dir(mod))
    with pytest.raises(AttributeError, match="no attribute"):
        _ = mod.not_a_name


def test_star_import_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec("from geonnax.basis import *", {})
        exec("from geonnax.geo import *", {})


def test_import_geonnax_is_silent_with_warnings_as_errors():
    code = "import geonnax, geonnax.basis, geonnax.diffops, geonnax.layers"
    subprocess.run([sys.executable, "-W", "error", "-c", code], check=True)


LONLAT = jnp.array([[3.0, 40.0], [-120.5, -33.2], [179.0, 85.0]])
XYZ = fg.lonlat_to_unit(LONLAT, unit="degrees")


def _equal(a, b) -> None:
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_frame_shims_keep_degrees_and_wgs84_defaults():
    wgs84 = {"semi_major": WGS84_A, "flattening": WGS84_F}
    ecef = fg.lonlat_to_ecef(LONLAT, 10.0, unit="degrees", **wgs84)
    _equal(_compat.lonlat_to_ecef(LONLAT, 10.0), ecef)
    _equal(
        _compat.lonlat_to_ecef(LONLAT, ellipsoid="sphere"),
        fg.lonlat_to_ecef(LONLAT, semi_major=EARTH_MEAN_RADIUS_M, unit="degrees"),
    )
    _equal(
        _compat.ecef_to_lonlat(ecef),
        fg.ecef_to_lonlat(ecef, unit="degrees", **wgs84),
    )
    _equal(_compat.enu_basis(LONLAT), fg.enu_basis(LONLAT, unit="degrees"))
    uv = jnp.ones((3, 2))
    _equal(
        _compat.tangent_to_cartesian(LONLAT, uv),
        fg.tangent_to_cartesian(LONLAT, uv, unit="degrees"),
    )
    _equal(
        _compat.cartesian_to_tangent(LONLAT, XYZ),
        fg.cartesian_to_tangent(LONLAT, XYZ, unit="degrees"),
    )
    _equal(
        _compat.local_tangent_plane(LONLAT, LONLAT[0]),
        fg.local_tangent_plane(LONLAT, LONLAT[0], radius=6371.0088, unit="degrees"),
    )
    _equal(
        _compat.rotation_to_pole(LONLAT[0]),
        fg.rotation_to_pole(LONLAT[0], unit="degrees"),
    )


def test_lonlat_shim_keeps_input_unit_and_promotes_integers():
    # jejjohnson/geonnax#158: functax's lonlat_to_unit takes unit= and raises on
    # integers, so the shim maps input_unit= and promotes integers to float32
    radians = jnp.deg2rad(LONLAT)
    _equal(_compat.lonlat_to_cartesian3d(radians), fg.lonlat_to_unit(radians))
    _equal(_compat.lonlat_to_cartesian3d(LONLAT, input_unit="degrees"), XYZ)
    integers = jnp.array([[0, 0], [90, 45]])
    out = _compat.lonlat_to_cartesian3d(integers, input_unit="degrees")
    assert out.dtype == jnp.float32
    _equal(out, fg.lonlat_to_unit(integers.astype(jnp.float32), unit="degrees"))


def test_lonlat_shim_keeps_the_old_checks():
    with pytest.raises(ValueError, match=r"lonlat must be \(N, 2\)"):
        _compat.lonlat_to_cartesian3d(jnp.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="input_unit must be"):
        _compat.lonlat_to_cartesian3d(LONLAT, input_unit="grads")


def test_cartesian_encoder_still_rejects_a_batch():
    from geonnax.encoders import Cartesian3DEncoder

    with pytest.raises(ValueError, match=r"lonlat must be \(2,\)"):
        Cartesian3DEncoder()(jnp.zeros((2, 2)))


def test_callers_of_lonlat_to_unit_still_take_integers():
    from geonnax.encoders import Cartesian3DEncoder
    from geonnax.geo import spherical_harmonic_encode

    expected = fg.lonlat_to_unit(jnp.array([90.0, 0.0]), unit="degrees")
    out = Cartesian3DEncoder(input_unit="degrees")(jnp.array([90, 0]))
    np.testing.assert_allclose(out, expected, atol=1e-7)
    features = spherical_harmonic_encode(jnp.array([[0, 1]]), l_max=2)
    assert features.dtype == jnp.float32
    assert features.shape == (1, 9)


def test_unknown_ellipsoid_is_rejected():
    with pytest.raises(ValueError, match="ellipsoid"):
        _compat.lonlat_to_ecef(LONLAT, ellipsoid="mars")


@pytest.mark.parametrize(
    ("shim", "args"),
    [
        ("fibonacci_sphere", (20,)),
        ("gauss_legendre_grid", (4, 8)),
        ("latlon_grid", (4, 8)),
        ("healpix_grid", (1,)),
    ],
)
def test_grid_shims_keep_the_radians_default(shim, args):
    old, new = getattr(_compat, shim), getattr(fg, shim)
    _equal(old(*args), new(*args))
    _equal(old(*args, output="lonlat"), new(*args, output="lonlat", unit="radians"))
    _equal(
        old(*args, output="lonlat", output_unit="degrees"),
        new(*args, output="lonlat", unit="degrees"),
    )


def test_gauss_legendre_shim_returns_a_numpy_tuple():
    nodes, weights = _compat.gauss_legendre(5)
    assert isinstance(nodes, np.ndarray)
    assert isinstance(weights, np.ndarray)
    np.testing.assert_allclose(weights.sum(), 2.0, rtol=1e-14)


def test_polygon_mask_and_surface_gradient_keep_the_degrees_default():
    polygon = jnp.array([[0.0, 30.0], [10.0, 30.0], [10.0, 50.0], [0.0, 50.0]])
    _equal(
        _compat.spherical_polygon_mask(XYZ, polygon),
        fs.spherical_polygon_mask(XYZ, polygon, unit="degrees"),
    )

    def field(p):
        return jnp.sin(p[0]) * jnp.cos(p[1])

    _equal(
        _compat.surface_grad_enu(field)(LONLAT[0]),
        fd.surface_grad_enu(field, unit="degrees")(LONLAT[0]),
    )
