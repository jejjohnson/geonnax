"""Smoke tests for the geonnax public surface."""

import jax.numpy as jnp

import geonnax


def test_version_exposed():
    assert isinstance(geonnax.__version__, str)


def test_geo_module_imports():
    from geonnax import geo

    assert hasattr(geo, "deg2rad")
    assert hasattr(geo, "lonlat_scale")
    assert hasattr(geo, "lonlat_to_cartesian3d")
    assert hasattr(geo, "cyclic_encode")
    assert hasattr(geo, "spherical_harmonic_encode")


def test_basis_module_imports():
    from geonnax import basis

    assert hasattr(basis, "fourier_features")
    assert hasattr(basis, "seasonal_features")
    assert hasattr(basis, "interaction_features")
    assert hasattr(basis, "standardize")
    assert hasattr(basis, "unstandardize")


def test_geo_deg2rad_smoke():
    from geonnax.geo import deg2rad

    out = deg2rad(jnp.array([0.0, 90.0, 180.0]))
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.array([0.0, jnp.pi / 2, jnp.pi]), atol=1e-6)


def test_basis_fourier_features_smoke():
    from geonnax.basis import fourier_features

    out = fourier_features(jnp.linspace(0.0, 1.0, 5), max_degree=3)
    assert out.shape == (5, 6)


def test_eigenbasis_submodule():
    from geonnax._basis import fourier_basis_1d, real_spherical_harmonics

    assert callable(fourier_basis_1d)
    assert callable(real_spherical_harmonics)
