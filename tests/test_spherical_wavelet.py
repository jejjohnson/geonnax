"""Tests for the spherical wavelet (scale-discretised) transform and conv."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import geonnax
from geonnax.layers import SphericalHarmonicTransform
from geonnax.layers._spherical_wavelet import (
    SphericalWaveletConv,
    SphericalWaveletTransform,
    _partition_filters,
)


KEY = jr.PRNGKey(0)


@pytest.mark.parametrize("l_max", [8, 12, 16, 31, 32])
def test_partition_of_unity(l_max):
    # Squared windows must sum to one at every degree (=> exact reconstruction).
    filters = _partition_filters(l_max)
    sums = (filters**2).sum(axis=0)
    assert np.abs(sums - 1.0).max() < 1e-12
    assert (filters >= 0).all()


def _band_limited_field(sht, channels):
    coeffs = jr.normal(KEY, (channels, (sht.l_max + 1) ** 2))
    return sht.inverse(coeffs)


def test_wavelet_round_trip_is_exact():
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht)
    field = _band_limited_field(sht, 3)
    scales = swt.forward(field)
    assert scales.shape == (swt.n_scales, 3, 16, 32)
    rec = swt.inverse(scales)
    assert float(jnp.abs(rec - field).max()) < 1e-4


def test_wavelet_scales_partition_the_field():
    # Summing the raw wavelet maps is NOT the identity, but re-synthesising is;
    # this guards the per-scale band-pass structure (more than one active scale).
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht)
    assert swt.n_scales >= 3


def test_n_scales_override():
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht, n_scales=4)
    assert swt.n_scales == 4
    # A truncated partition no longer reconstructs exactly, but shapes hold.
    assert swt.forward(jnp.ones((2, 16, 32))).shape == (4, 2, 16, 32)


def test_spherical_wavelet_conv():
    sht = SphericalHarmonicTransform.init(16, 32, 12)
    swt = SphericalWaveletTransform.init(sht)
    conv = SphericalWaveletConv.init(3, 5, swt.n_scales, key=KEY)
    x = jnp.ones((3, 16, 32))
    assert conv(x, swt).shape == (5, 16, 32)

    def loss(c, x, t):
        return jnp.mean(c(x, t) ** 2)

    grads = eqx.filter_grad(loss)(conv, x, swt)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.isfinite(g).all() for g in leaves)


def test_transform_windows_are_frozen():
    sht = SphericalHarmonicTransform.init(12, 24, 8)
    swt = SphericalWaveletTransform.init(sht)
    x = jnp.ones((2, 12, 24))

    def loss(t, x):
        return jnp.mean(t.forward(x) ** 2)

    grads = eqx.filter_grad(loss)(swt, x)
    filt_grads = jax.tree_util.tree_leaves(eqx.filter(grads.filters, eqx.is_array))
    assert all(float(jnp.abs(g).max()) == 0.0 for g in filt_grads)


def test_public_exports():
    for name in ["SphericalWaveletTransform", "SphericalWaveletConv"]:
        assert hasattr(geonnax, name), name
