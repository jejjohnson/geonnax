"""Tests for spectral convolutions, FNO, the SHT, and SFNO."""

import einx
import equinox as eqx
import jax
import jax.image as jimage
import jax.numpy as jnp
import jax.random as jr
import pytest
from _point_invariance import assert_point_refinement_converges, grid_nodes

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


@pytest.mark.slow
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


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
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


@pytest.mark.integration
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


# ---- SpectralConv on point clouds ---------------------------------------------


def _with_random_bias(sc):
    return eqx.tree_at(lambda m: m.bias, sc, jr.normal(KEY, sc.bias.shape))


def _band_limited(xi):
    """A smooth field inside a (6, 6)-mode band, (1, N) from xi (N, 2)."""
    a, b = xi[:, 0], xi[:, 1]
    two_pi = 2 * jnp.pi
    return (
        jnp.sin(two_pi * a)
        + jnp.cos(two_pi * (a + 2 * b))
        + 0.5 * jnp.sin(two_pi * 3 * b)
    )[None]


_SLOW_FACS = [
    fac if fac == "dense" else pytest.param(fac, marks=pytest.mark.slow)
    for fac in FACTORIZATIONS
]


@pytest.mark.parametrize("fac", _SLOW_FACS)
def test_spectral_conv_from_points_matches_call(fac):
    # On the grid nodes with w = 1/S the quadrature is the DFT itself, so the
    # point path reproduces __call__ up to float32 round-off.
    grid = (16, 12)
    sc = _with_random_bias(
        SpectralConv.init(3, 4, (4, 3), key=KEY, factorization=fac, rank=0.5)
    )
    x = jr.normal(jr.PRNGKey(1), (3, *grid))
    s = x[0].size
    u = einx.id("c a... -> c (a...)", x)
    y = sc.from_points(u, grid_nodes(grid), jnp.full((s,), 1.0 / s), grid)
    assert y.shape == (4, *grid)
    assert jnp.allclose(y, sc(x), atol=1e-5)


@pytest.mark.parametrize("fac", _SLOW_FACS)
def test_spectral_conv_to_points_matches_call(fac):
    grid = (16, 12)
    sc = _with_random_bias(
        SpectralConv.init(3, 4, (4, 3), key=KEY, factorization=fac, rank=0.5)
    )
    x = jr.normal(jr.PRNGKey(1), (3, *grid))
    y = sc.to_points(x, grid_nodes(grid))
    assert y.shape == (4, x[0].size)
    assert jnp.allclose(y, einx.id("c a... -> c (a...)", sc(x)), atol=1e-5)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_spectral_conv_points_dimension_flexible(nd):
    grid = (10,) * nd
    sc = _with_random_bias(SpectralConv.init(2, 3, (3,) * nd, key=KEY))
    x = jr.normal(jr.PRNGKey(1), (2, *grid))
    s = x[0].size
    nodes = grid_nodes(grid)
    u = einx.id("c a... -> c (a...)", x)
    y_from = sc.from_points(u, nodes, jnp.full((s,), 1.0 / s), grid)
    y_to = sc.to_points(x, nodes)
    assert jnp.allclose(y_from, sc(x), atol=1e-5)
    assert jnp.allclose(y_to, einx.id("c a... -> c (a...)", sc(x)), atol=1e-5)


def test_spectral_conv_from_points_padding():
    # Zero-weight points contribute exact zeros, so padding a cloud with
    # arbitrary finite points leaves the output unchanged.
    sc = _with_random_bias(SpectralConv.init(2, 3, (4, 4), key=KEY))
    k1, k2, k3, k4 = jr.split(jr.PRNGKey(1), 4)
    n, n_pad = 300, 80
    xi = jr.uniform(k1, (n, 2))
    u = jr.normal(k2, (2, n))
    w = jnp.full((n,), 1.0 / n)
    xi_pad = jnp.concatenate([xi, jr.uniform(k3, (n_pad, 2))])
    u_pad = jnp.hstack([u, jr.normal(k4, (2, n_pad))])
    w_pad = jnp.concatenate([w, jnp.zeros((n_pad,))])
    y = sc.from_points(u, xi, w, (16, 16))
    y_pad = sc.from_points(u_pad, xi_pad, w_pad, (16, 16))
    assert jnp.array_equal(y, y_pad)


def _from_points_setup():
    sc = SpectralConv.init(1, 1, (6, 6), key=KEY, bias=False)
    grid = (64, 64)
    reference = sc(einx.id("c (a b) -> c a b", _band_limited(grid_nodes(grid)), a=64))
    return sc, grid, reference


def test_spectral_conv_point_resolution_invariance():
    # Jittered stratified clouds at N = 2025 / 8100 against the 64 x 64 grid
    # reference. Stratified quadrature converges ~N^-1: the theme-epic (#75)
    # prototype measured 0.11 / 0.024, and eight harness keys here give at
    # most 0.105 / 0.028.
    sc, grid, reference = _from_points_setup()
    assert_point_refinement_converges(
        lambda u, xi, w: sc.from_points(u, xi, w, grid),
        _band_limited,
        reference,
        (45, 90),
        d=2,
        key=KEY,
        max_rel_err=(0.15, 0.05),
    )


def test_spectral_conv_point_refinement_unweighted_fails():
    # Negative control: the reference Geo-FNO transform sums without
    # quadrature weights (w = 1), so its output grows ~N (rel. error ~2000 /
    # ~8000 here) instead of converging. This is why from_points takes weights.
    sc, grid, reference = _from_points_setup()
    with pytest.raises(AssertionError, match="does not converge"):
        assert_point_refinement_converges(
            lambda u, xi, w: sc.from_points(u, xi, jnp.ones_like(w), grid),
            _band_limited,
            reference,
            (45, 90),
            d=2,
            key=KEY,
            max_rel_err=(0.15, 0.05),
        )


def test_spectral_conv_from_points_rejects_underresolved_grid():
    sc = SpectralConv.init(2, 2, (4, 4), key=KEY)
    xi = jr.uniform(KEY, (10, 2))
    u, w = jnp.ones((2, 10)), jnp.full((10,), 0.1)
    with pytest.raises(ValueError, match="grid must have 2 extents"):
        sc.from_points(u, xi, w, (8, 16))
    with pytest.raises(ValueError, match="grid must have 2 extents"):
        sc.from_points(u, xi, w, (16, 16, 16))
    with pytest.raises(ValueError, match="grid must have 2 extents"):
        sc.to_points(jnp.ones((2, 16, 8)), xi)


@pytest.mark.integration
def test_spectral_conv_points_vmap_jit_grad():
    sc = SpectralConv.init(2, 2, (4, 4), key=KEY)
    batch, n = 3, 200
    k1, k2 = jr.split(jr.PRNGKey(1))
    xi = jr.uniform(k1, (batch, n, 2))
    u = jr.normal(k2, (batch, 2, n))
    # Ragged clouds of 200, 150 and 120 real points, zero-weight padded to 200.
    n_real = jnp.array([200, 150, 120])
    real = einx.less("n, b -> b n", jnp.arange(n), n_real)
    w = einx.where("b n, b, -> b n", real, 1.0 / n_real, 0.0)

    def apply(m, u, xi, w):
        return m.to_points(m.from_points(u, xi, w, (16, 16)), xi)

    y = eqx.filter_jit(jax.vmap(apply, in_axes=(None, 0, 0, 0)))(sc, u, xi, w)
    assert y.shape == (batch, 2, n)
    assert bool(jnp.isfinite(y).all())

    def loss(m, xi):
        return jnp.mean(apply(m, u[0], xi, w[0]) ** 2)

    g_m = eqx.filter_grad(loss)(sc, xi[0])
    leaves = jax.tree_util.tree_leaves(eqx.filter(g_m, eqx.is_array))
    assert leaves and all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert bool(jnp.isfinite(jax.grad(loss, argnums=1)(sc, xi[0])).all())


# ---- FNO --------------------------------------------------------------------


@pytest.mark.slow
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


@pytest.mark.integration
def test_fno_vmap_jit_grad():
    m = FNO.init(2, 2, (6, 6), key=KEY, hidden_channels=16, n_layers=2)
    x = jnp.ones((2, 32, 32))
    assert jax.vmap(m)(jnp.ones((4, 2, 32, 32))).shape == (4, 2, 32, 32)
    assert eqx.filter_jit(m)(x).shape == (2, 2, 32, 32)[1:]
    assert _grads_finite(m, x)


@pytest.mark.parametrize("nd", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
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


def test_sht_tables_are_built_in_float64():
    # #122: the tables are constants, so `init` evaluates the harmonics in float64
    # and casts once. analy @ synth = I is then exact to float32 rounding of the
    # tables: measured 4.9e-8 at l_max = 63 (5.3e-6 when the harmonics were
    # evaluated in float32); 5e-7 leaves margin and fails the float32 build.
    l_max = 63
    sht = SphericalHarmonicTransform.init(l_max + 1, 2 * l_max + 2, l_max)
    assert sht.synth.dtype == jnp.zeros(()).dtype  # default float, cast once
    with jax.enable_x64(True):
        gram = jnp.asarray(sht.analy, jnp.float64) @ jnp.asarray(sht.synth, jnp.float64)
        err = jnp.abs(gram - jnp.eye(gram.shape[0])).max()
    assert float(err) < 5e-7


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


@pytest.mark.integration
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


def test_fno_curvilinear_grid_with_coordinate_channels():
    """Case A: plain FNO on a 30x20 curvilinear grid with (x, y) channels."""
    from geonnax import FNO
    from geonnax.geo import local_tangent_plane

    n1, n2 = 30, 20
    s = jnp.linspace(0.0, 1.0, n1)[:, None]
    p = jnp.linspace(-1.0, 1.0, n2)[None, :]
    lon = 5.0 + 2.0 * s + 0.5 * p
    lat = 45.0 + 3.0 * s - 0.1 * p
    lonlat = jnp.stack([lon.ravel(), lat.ravel()], -1)
    enu = local_tangent_plane(lonlat, jnp.array([lon.mean(), lat.mean()]))
    coords = enu.T.reshape(2, n1, n2) / 100.0
    x = jnp.concatenate([jnp.sin(3.0 * s * p)[None], coords], axis=0)
    op = FNO.init(
        3,
        1,
        (4, 4),
        key=jr.PRNGKey(0),
        hidden_channels=8,
        n_layers=2,
        domain_padding=0.1,
    )
    y = op(x)
    assert y.shape == (1, n1, n2)
    assert bool(jnp.isfinite(y).all())
