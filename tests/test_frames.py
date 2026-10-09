"""Tests for the coordinate-frame helpers (GN4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import slepian_cap_basis
from geonnax.geo import (
    WGS84_A,
    WGS84_F,
    cartesian_to_tangent,
    ecef_to_lonlat,
    enu_basis,
    local_tangent_plane,
    lonlat_to_cartesian3d,
    lonlat_to_ecef,
    rotation_to_pole,
    tangent_to_cartesian,
)


def _points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return np.stack([lon, lat], axis=-1)


def test_ecef_known_points():
    with jax.enable_x64(True):
        xyz = np.asarray(lonlat_to_ecef(jnp.array([[0.0, 0.0], [0.0, 90.0]])))
    np.testing.assert_allclose(xyz[0], [WGS84_A, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(xyz[1], [0.0, 0.0, WGS84_A * (1 - WGS84_F)], atol=1e-6)


@pytest.mark.parametrize("ellipsoid", ["wgs84", "sphere"])
def test_ecef_round_trip(ellipsoid):
    lonlat = _points()
    h = np.random.default_rng(1).uniform(-10_000.0, 10_000.0, len(lonlat))
    with jax.enable_x64(True):
        xyz = lonlat_to_ecef(jnp.asarray(lonlat), jnp.asarray(h), ellipsoid=ellipsoid)
        back, h_back = ecef_to_lonlat(xyz, ellipsoid=ellipsoid)
        again = lonlat_to_ecef(back, h_back, ellipsoid=ellipsoid)
    # Compare in ECEF: longitude is undefined at the poles.
    np.testing.assert_allclose(np.asarray(again), np.asarray(xyz), atol=1e-6)  # < 1 µm
    np.testing.assert_allclose(np.asarray(h_back), h, atol=1e-6)


def test_sphere_ecef_matches_unit_vectors():
    lonlat = _points()
    with jax.enable_x64(True):
        xyz = lonlat_to_ecef(jnp.asarray(lonlat), ellipsoid="sphere", radius=2.0)
        unit = lonlat_to_cartesian3d(jnp.asarray(lonlat), input_unit="degrees")
    np.testing.assert_allclose(np.asarray(xyz), 2.0 * np.asarray(unit), atol=1e-12)


def test_enu_is_orthonormal_and_right_handed():
    lonlat = np.concatenate([_points(), [[30.0, 90.0], [-60.0, -90.0]]])
    with jax.enable_x64(True):
        frame = np.asarray(enu_basis(jnp.asarray(lonlat)))
        unit = np.asarray(
            lonlat_to_cartesian3d(jnp.asarray(lonlat), input_unit="degrees")
        )
    gram = np.einsum("nij,nkj->nik", frame, frame)
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(3), gram.shape), atol=1e-12)
    np.testing.assert_allclose(
        np.cross(frame[:, 0], frame[:, 1]), frame[:, 2], atol=1e-12
    )
    np.testing.assert_allclose(frame[:, 2], unit, atol=1e-12)


def test_tangent_round_trip_and_tangency():
    lonlat = _points()
    uv = np.random.default_rng(2).normal(size=(len(lonlat), 2))
    with jax.enable_x64(True):
        vec = tangent_to_cartesian(jnp.asarray(lonlat), jnp.asarray(uv))
        back = cartesian_to_tangent(jnp.asarray(lonlat), vec)
        unit = lonlat_to_cartesian3d(jnp.asarray(lonlat), input_unit="degrees")
    np.testing.assert_allclose(np.asarray(back), uv, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(np.asarray(vec) * np.asarray(unit), axis=1), 0, atol=1e-12
    )


def test_azimuthal_equidistant_preserves_distance_from_origin():
    origin = np.array([12.0, -30.0])
    lonlat = _points()
    with jax.enable_x64(True):
        xy = np.asarray(
            local_tangent_plane(jnp.asarray(lonlat), jnp.asarray(origin), radius=1.0)
        )
        p = np.asarray(lonlat_to_cartesian3d(jnp.asarray(lonlat), input_unit="degrees"))
        o = np.asarray(
            lonlat_to_cartesian3d(jnp.asarray(origin[None]), input_unit="degrees")
        )[0]
    gc = np.arctan2(np.linalg.norm(np.cross(p, o), axis=1), p @ o)
    np.testing.assert_allclose(np.linalg.norm(xy, axis=1), gc, atol=1e-10)


def test_gnomonic_and_origin_gradient():
    origin = jnp.array([0.0, 0.0])
    xy = local_tangent_plane(
        jnp.array([[10.0, 0.0]]), origin, projection="gnomonic", radius=1.0
    )
    np.testing.assert_allclose(
        np.asarray(xy), [[np.tan(np.radians(10.0)), 0.0]], atol=1e-6
    )

    def f(ll):
        return jnp.sum(local_tangent_plane(ll, origin))

    assert jnp.isfinite(jax.grad(f)(jnp.array([[0.0, 0.0]]))).all()
    with pytest.raises(ValueError, match="projection"):
        local_tangent_plane(jnp.zeros((1, 2)), origin, projection="mercator")  # ty: ignore[invalid-argument-type]


def test_rotation_to_pole():
    centre = np.array([40.0, 25.0])
    with jax.enable_x64(True):
        R = np.asarray(rotation_to_pole(jnp.asarray(centre)))
        c = np.asarray(
            lonlat_to_cartesian3d(jnp.asarray(centre[None]), input_unit="degrees")
        )[0]
    np.testing.assert_allclose(R @ c, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_slepian_rotation_uses_the_same_frame():
    basis = slepian_cap_basis(4, 0.4, n_modes=3, eig_threshold=0.0).rotate_to(
        jnp.array([0.3, 0.5])
    )
    xyz = lonlat_to_cartesian3d(jnp.asarray(_points(5)), input_unit="degrees")
    R = rotation_to_pole(jnp.array([0.3, 0.5]), input_unit="radians")
    np.testing.assert_allclose(
        np.asarray(basis.centred_coordinates(xyz)),
        np.asarray(xyz) @ np.asarray(R).T,
        atol=1e-6,
    )
