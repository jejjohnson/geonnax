"""Smoke + forward-equality tests for the geonnax public surface."""

import jax
import jax.numpy as jnp
import jax.random as jr

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
    out = geonnax.geo.deg2rad(jnp.array([0.0, 90.0, 180.0]))
    assert out.shape == (3,)
    assert jnp.allclose(out, jnp.array([0.0, jnp.pi / 2, jnp.pi]), atol=1e-6)


def test_basis_fourier_features_smoke():
    out = geonnax.basis.fourier_features(jnp.linspace(0.0, 1.0, 5), max_degree=3)
    assert out.shape == (5, 6)


def test_eigenbasis_submodule():
    from geonnax._basis import fourier_basis_1d, real_spherical_harmonics

    assert callable(fourier_basis_1d)
    assert callable(real_spherical_harmonics)


# ---- Tier A: deterministic encoders -----------------------------------------


def test_deg2rad_module():
    enc = geonnax.Deg2Rad()
    out = enc(jnp.array([0.0, 90.0, 180.0]))
    assert jnp.allclose(out, jnp.array([0.0, jnp.pi / 2, jnp.pi]), atol=1e-6)


def test_lonlat_scale_module():
    enc = geonnax.LonLatScale()
    out = enc(jnp.array([[-180.0, -90.0], [0.0, 0.0], [180.0, 90.0]]))
    expected = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    assert jnp.allclose(out, expected)


def test_cartesian3d_encoder_module():
    enc = geonnax.Cartesian3DEncoder()
    out = enc(jnp.array([[0.0, 0.0]]))
    assert jnp.allclose(out, jnp.array([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_cyclic_encoder_module():
    enc = geonnax.CyclicEncoder()
    out = enc(jnp.array([0.0, jnp.pi]))
    assert out.shape == (2, 2)
    assert jnp.allclose(out[:, 0], jnp.array([1.0, -1.0]), atol=1e-5)


def test_spherical_harmonic_encoder_cartesian():
    enc = geonnax.SphericalHarmonicEncoder(l_max=3)
    xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = enc(xyz)
    assert out.shape == (2, 16)
    assert enc.num_features == 16


def test_spherical_harmonic_encoder_lonlat():
    enc = geonnax.SphericalHarmonicEncoder(l_max=2, input_mode="lonlat")
    out = enc(jnp.array([[0.0, 0.0], [jnp.pi / 2, 0.0]]))
    assert out.shape == (2, 9)


# ---- Tier A: stochastic-via-key cores ---------------------------------------


def test_mcdropout_keeps_shape_and_scales():
    drop = geonnax.MCDropout(rate=0.5)
    x = jnp.ones((4, 8))
    out = drop(x, key=jr.PRNGKey(0))
    assert out.shape == x.shape
    # Survivors are scaled by 2.0; zeros are dropped.
    assert jnp.all((out == 0.0) | (out == 2.0))


def test_ncp_continuous_perturb_changes_input():
    perturb = geonnax.NCPContinuousPerturb(scale=0.5)
    x = jnp.zeros((4, 3))
    out = perturb(x, key=jr.PRNGKey(0))
    assert out.shape == x.shape
    assert jnp.any(out != 0.0)


# ---- Tier A: SIREN ----------------------------------------------------------


def test_siren_dense_init_and_call():
    layer = geonnax.SirenDense.init(3, 8, key=jr.PRNGKey(0), layer_type="first")
    out = layer(jnp.ones((5, 3)))
    assert out.shape == (5, 8)
    # Sine activation keeps outputs in [-1, 1].
    assert jnp.max(jnp.abs(out)) <= 1.0


def test_siren_full_network():
    net = geonnax.SIREN.init(2, 16, 1, depth=4, key=jr.PRNGKey(0))
    out = net(jnp.zeros((10, 2)))
    assert out.shape == (10, 1)
    assert len(net.layers) == 4


def test_siren_W_limit_regimes():
    assert geonnax.siren_W_limit("first", 4, omega=30.0) == 1.0 / 4
    # Hidden divides by omega so high omega gives small limit.
    hidden = geonnax.siren_W_limit("hidden", 4, omega=30.0, c=6.0)
    assert hidden < geonnax.siren_W_limit("last", 4, omega=30.0, c=6.0)


# ---- Tier A: random features ------------------------------------------------


def test_orthogonal_random_features_shape():
    orf = geonnax.OrthogonalRandomFeatures.init(
        4, 8, key=jr.PRNGKey(0), lengthscale=1.0
    )
    out = orf(jnp.ones((3, 4)))
    # rff_forward concatenates cos and sin so output has 2 * n_features columns.
    assert out.shape == (3, 16)


def test_rff_forward_helper_matches_orf():
    key = jr.PRNGKey(1)
    orf = geonnax.OrthogonalRandomFeatures.init(2, 4, key=key, lengthscale=1.0)
    x = jr.normal(jr.PRNGKey(2), (3, 2))
    direct = geonnax.rff_forward(orf.W, orf.lengthscale, orf.n_features, x)
    assert jnp.allclose(orf(x), direct)


# ---- Tier A: Slepian --------------------------------------------------------


def test_slepian_encoder_from_cap_runs():
    enc = geonnax.SlepianEncoder.from_cap(
        l_max=3,
        cap_radius_deg=20.0,
        cap_centre_lonlat_deg=(0.0, 0.0),
        eig_threshold=0.0,
        n_modes=4,
    )
    out = enc(jnp.array([[0.0, 0.0], [0.1, 0.05]]))
    assert out.shape == (2, enc.num_features)


def test_hybrid_spherical_slepian_concatenates():
    enc = geonnax.HybridSphericalSlepianEncoder.from_cap(
        sh_l_max=1,
        slepian_l_max=3,
        cap_radius_deg=20.0,
        cap_centre_lonlat_deg=(0.0, 0.0),
        eig_threshold=0.0,
        n_modes=2,
    )
    out = enc(jnp.array([[0.0, 0.0]]))
    assert out.shape == (1, enc.num_features)


# ---- Composition test: everything sits in eqx.nn.Sequential -----------------


def test_encoders_compose_by_hand():
    deg = geonnax.Deg2Rad()
    cart = geonnax.Cartesian3DEncoder(input_unit="radians")
    out = cart(deg(jnp.array([[0.0, 0.0]])))
    assert out.shape == (1, 3)


def test_no_numpyro_import_in_package():
    """geonnax must remain pure-Equinox — no numpyro symbols anywhere."""
    import sys

    # Force a clean re-import path check: confirm geonnax does not pull numpyro.
    for mod_name in list(sys.modules):
        if mod_name.startswith("geonnax"):
            mod = sys.modules[mod_name]
            for attr in dir(mod):
                obj = getattr(mod, attr, None)
                module_name = getattr(obj, "__module__", "") or ""
                assert "numpyro" not in module_name, (
                    f"{mod_name}.{attr} leaks numpyro symbol "
                    f"(__module__={module_name!r})"
                )


def test_jit_round_trip_siren():
    """Sanity-check a JIT'd SIREN forward — exercise the static fields."""
    net = geonnax.SIREN.init(2, 8, 1, depth=3, key=jr.PRNGKey(0))
    f = jax.jit(lambda m, x: m(x))
    out = f(net, jnp.zeros((4, 2)))
    assert out.shape == (4, 1)
