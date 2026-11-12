"""Tests for the spherical wavelet conv. The transform it runs on moved to
functax (jejjohnson/geonnax#158), which tests the windows and the round trip;
these check that the conv composes with it and that its windows stay frozen.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from functax.transform import SphericalHarmonicTransform, SphericalWaveletTransform

import geonnax
from geonnax.layers import SphericalWaveletConv


KEY = jr.PRNGKey(0)


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


def test_conv_leaves_the_transform_windows_frozen():
    sht = SphericalHarmonicTransform.init(12, 24, 8)
    swt = SphericalWaveletTransform.init(sht)
    conv = SphericalWaveletConv.init(2, 2, swt.n_scales, key=KEY)
    x = jnp.ones((2, 12, 24))

    def loss(t, x):
        return jnp.mean(conv(x, t) ** 2)

    grads = eqx.filter_grad(loss)(swt, x)
    assert float(jnp.abs(grads.filters).max()) == 0.0


def test_public_exports():
    assert hasattr(geonnax, "SphericalWaveletConv")
    assert "SphericalWaveletConv" in geonnax.__all__
    # the transform is a deprecated alias now, not an export
    assert "SphericalWaveletTransform" not in geonnax.__all__
    assert "SphericalWaveletTransform" not in geonnax.layers.__all__
