"""Tests for spectral convolutions, FNO, the SHT, and SFNO."""

import equinox as eqx
import jax
import jax.image as jimage
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.fno import FNO, DomainPadding, FNOBlock, FourierNeuralOperator
from geonnax.layers._factorized import init_factorized_tensor
from geonnax.layers._spectral import SpectralConv
from geonnax.layers._spherical_transform import (
    SphericalHarmonicTransform,
    SphericalSpectralConv,
)
from geonnax.sfno import SFNO


KEY = jr.PRNGKey(0)
FACTORIZATIONS = ["dense", "cp", "tucker", "tt"]


def _grads_finite(model, *args):
    def loss(m, *a):
        return jnp.mean(m(*a) ** 2)

    grads = eqx.filter_grad(loss)(model, *args)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    return bool(leaves) and all(jnp.isfinite(g).all() for g in leaves)


# ---- factorized tensors -----------------------------------------------------


@pytest.mark.parametrize("fac", FACTORIZATIONS)
def test_factorized_tensor_reconstructs_shape(fac):
    shape = (4, 6, 8, 2)
    t = init_factorized_tensor(shape, fac, key=KEY, rank=0.5, scale=0.1)
    assert t.reconstruct().shape == shape


def test_factorizations_reduce_parameters():
    shape = (8, 8, 16, 16, 2)
    dense = init_factorized_tensor(shape, "dense", key=KEY)
    n_dense = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(dense, eqx.is_array))
    )
    for fac in ["cp", "tucker", "tt"]:
        t = init_factorized_tensor(shape, fac, key=KEY, rank=0.25)
        n = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array)))
        assert n < n_dense


def test_init_factorized_tensor_rejects_unknown():
    with pytest.raises(ValueError, match="factorization must be"):
        init_factorized_tensor((2, 2), "lowrank", key=KEY)


# ---- SpectralConv -----------------------------------------------------------


@pytest.mark.parametrize("fac", FACTORIZATIONS)
def test_spectral_conv_shapes(fac):
    sc = SpectralConv.init(4, 6, (8, 8), key=KEY, factorization=fac, rank=0.5)
    assert sc(jnp.ones((4, 32, 32))).shape == (6, 32, 32)


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_spectral_conv_dimension_flexible(nd):
    sc = SpectralConv.init(2, 3, (4,) * nd, key=KEY)
    x = jnp.ones((2,) + (16,) * nd)
    assert sc(x).shape == (3,) + (16,) * nd


def test_spectral_conv_resolution_invariance():
    # The same operator applied to a smooth field at two grid resolutions
    # agrees up to interpolation error — the defining FNO property.
    sc = SpectralConv.init(1, 1, (6, 6), key=KEY, bias=False)

    def field(n):
        xs = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
        return (jnp.sin(xs)[:, None] + jnp.cos(xs)[None, :])[None]

    y_lo = sc(field(64))
    y_hi = jimage.resize(sc(field(128)), y_lo.shape, "linear")
    rel = jnp.linalg.norm(y_lo - y_hi) / jnp.linalg.norm(y_lo)
    assert float(rel) < 0.05


def test_spectral_conv_grad_and_jit():
    sc = SpectralConv.init(2, 2, (6, 6), key=KEY)
    x = jnp.ones((2, 32, 32))
    assert _grads_finite(sc, x)
    assert eqx.filter_jit(sc)(x).shape == (2, 32, 32)


def test_spectral_conv_rejects_bad_modes():
    with pytest.raises(ValueError, match="n_modes must be non-empty"):
        SpectralConv.init(2, 2, (), key=KEY)
    with pytest.raises(ValueError, match="all n_modes must be"):
        SpectralConv.init(2, 2, (4, 0), key=KEY)


# ---- FNO --------------------------------------------------------------------


@pytest.mark.parametrize("fac", FACTORIZATIONS)
def test_fno_factorizations(fac):
    m = FNO.init(
        3,
        1,
        (8, 8),
        key=KEY,
        hidden_channels=16,
        n_layers=3,
        factorization=fac,
        rank=0.5,
    )
    assert m(jnp.ones((3, 32, 32))).shape == (1, 32, 32)


def test_fno_vmap_jit_grad():
    m = FNO.init(2, 2, (6, 6), key=KEY, hidden_channels=16, n_layers=2)
    x = jnp.ones((2, 32, 32))
    assert jax.vmap(m)(jnp.ones((4, 2, 32, 32))).shape == (4, 2, 32, 32)
    assert eqx.filter_jit(m)(x).shape == (2, 2, 32, 32)[1:]
    assert _grads_finite(m, x)


@pytest.mark.parametrize("nd", [1, 2, 3])
def test_fno_dimension_flexible(nd):
    m = FNO.init(2, 3, (4,) * nd, key=KEY, hidden_channels=8, n_layers=2)
    x = jnp.ones((2,) + (16,) * nd)
    assert m(x).shape == (3,) + (16,) * nd


def test_fno_domain_padding():
    m = FNO.init(
        1, 1, (6, 6), key=KEY, hidden_channels=8, n_layers=2, domain_padding=0.1
    )
    assert m(jnp.ones((1, 40, 40))).shape == (1, 40, 40)
    assert isinstance(m.domain_padding, DomainPadding)


def test_fno_no_domain_padding_by_default():
    m = FNO.init(1, 1, (6, 6), key=KEY, hidden_channels=8, n_layers=2)
    assert m.domain_padding is None


def test_fno_block_shape():
    blk = FNOBlock.init(8, (6, 6), key=KEY)
    assert blk(jnp.ones((8, 32, 32))).shape == (8, 32, 32)


def test_fno_rejects_bad_args():
    with pytest.raises(ValueError, match="n_layers must be"):
        FNO.init(1, 1, (4, 4), key=KEY, n_layers=0)
    with pytest.raises(ValueError, match="n_modes must be non-empty"):
        FNO.init(1, 1, (), key=KEY)


def test_fourier_neural_operator_alias():
    assert FourierNeuralOperator is FNO
    assert geonnax.FNO is FNO


# ---- spherical harmonic transform -------------------------------------------


def test_sht_round_trip_is_exact_on_band_limited():
    l_max, n_lat, n_lon = 8, 12, 24
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    coeffs = jr.normal(KEY, (3, (l_max + 1) ** 2))
    field = sht.inverse(coeffs)
    recovered = sht.forward(field)
    rel = jnp.linalg.norm(recovered - coeffs) / jnp.linalg.norm(coeffs)
    assert float(rel) < 1e-4


def test_sht_shapes():
    sht = SphericalHarmonicTransform.init(10, 20, 6)
    field = jnp.ones((2, 10, 20))
    coeffs = sht.forward(field)
    assert coeffs.shape == (2, 49)
    assert sht.inverse(coeffs).shape == (2, 10, 20)


def test_sht_rejects_negative_l_max():
    with pytest.raises(ValueError, match="l_max must be"):
        SphericalHarmonicTransform.init(8, 16, -1)


def test_sht_rejects_under_resolved_grid():
    with pytest.raises(ValueError, match="n_lat must exceed l_max"):
        SphericalHarmonicTransform.init(8, 64, 12)
    with pytest.raises(ValueError, match="n_lon must exceed 2"):
        SphericalHarmonicTransform.init(32, 16, 12)


def test_spherical_spectral_conv():
    l_max, n_lat, n_lon = 8, 12, 24
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    ssc = SphericalSpectralConv.init(3, 5, l_max, key=KEY)
    x = jnp.ones((3, n_lat, n_lon))
    assert ssc(x, sht).shape == (5, n_lat, n_lon)
    assert _grads_finite(ssc, x, sht)


# ---- SFNO -------------------------------------------------------------------


def test_sfno_shapes_vmap_jit_grad():
    m = SFNO.init(
        3, 2, key=KEY, n_lat=16, n_lon=32, l_max=10, hidden_channels=16, n_layers=3
    )
    x = jnp.ones((3, 16, 32))
    assert m(x).shape == (2, 16, 32)
    assert jax.vmap(m)(jnp.ones((4, 3, 16, 32))).shape == (4, 2, 16, 32)
    assert eqx.filter_jit(m)(x).shape == (2, 16, 32)
    assert _grads_finite(m, x)


def test_sfno_transform_matrices_are_frozen():
    m = SFNO.init(2, 1, key=KEY, n_lat=12, n_lon=24, l_max=8, hidden_channels=8)
    x = jnp.ones((2, 12, 24))

    def loss(model, x):
        return jnp.mean(model(x) ** 2)

    grads = eqx.filter_grad(loss)(m, x)
    sht_grads = jax.tree_util.tree_leaves(eqx.filter(grads.sht, eqx.is_array))
    assert all(float(jnp.abs(g).max()) == 0.0 for g in sht_grads)


def test_sfno_rejects_bad_layers():
    with pytest.raises(ValueError, match="n_layers must be"):
        SFNO.init(1, 1, key=KEY, n_lat=8, n_lon=16, l_max=4, n_layers=0)


def test_public_exports_present():
    for name in [
        "FNO",
        "SFNO",
        "SpectralConv",
        "SphericalHarmonicTransform",
        "SphericalSpectralConv",
        "DomainPadding",
        "CPTensor",
        "TuckerTensor",
        "TTTensor",
        "DenseTensor",
    ]:
        assert hasattr(geonnax, name), name
    assert hasattr(geonnax, "fno")
    assert hasattr(geonnax, "sfno")
