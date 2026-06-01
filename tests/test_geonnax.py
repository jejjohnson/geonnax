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


# ---- Tier B: MFN ------------------------------------------------------------


def test_fourier_filter_shape():
    f = geonnax.FourierFilter.init(3, 8, key=jr.PRNGKey(0))
    out = f(jnp.ones((4, 3)))
    assert out.shape == (4, 8)
    assert jnp.max(jnp.abs(out)) <= 1.0  # sin is bounded


def test_gabor_filter_shape():
    g = geonnax.GaborFilter.init(2, 6, key=jr.PRNGKey(0))
    out = g(jnp.zeros((3, 2)))
    assert out.shape == (3, 6)


def test_fourier_net_forward():
    net = geonnax.FourierNet.init(2, 16, 1, depth=3, key=jr.PRNGKey(0))
    out = net(jnp.zeros((4, 2)))
    assert out.shape == (4, 1)


def test_gabor_net_forward():
    net = geonnax.GaborNet.init(2, 16, 1, depth=3, key=jr.PRNGKey(0))
    out = net(jnp.zeros((4, 2)))
    assert out.shape == (4, 1)


def test_mfn_squeeze_single_point():
    net = geonnax.FourierNet.init(2, 4, 3, depth=2, key=jr.PRNGKey(0))
    out = net(jnp.zeros(2))
    assert out.shape == (3,)  # 1-D input -> 1-D output


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


# ---- Tier C: heteroscedastic FA heads --------------------------------------


def test_mc_softmax_dense_fa_init_and_forward():
    layer = geonnax.MCSoftmaxDenseFA.init(
        in_features=4, num_classes=3, rank=2, key=jr.PRNGKey(0)
    )
    x = jnp.ones((5, 4))
    probs = layer(x, key=jr.PRNGKey(1))
    assert probs.shape == (5, 3)
    # MC-averaged softmax rows must still sum to 1.
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_mc_sigmoid_dense_fa_init_and_forward():
    layer = geonnax.MCSigmoidDenseFA.init(
        in_features=4, num_classes=3, rank=2, key=jr.PRNGKey(0)
    )
    x = jnp.ones((5, 4))
    probs = layer(x, key=jr.PRNGKey(1))
    assert probs.shape == (5, 3)
    # MC-averaged sigmoid stays in [0, 1] per element.
    assert jnp.all((probs >= 0.0) & (probs <= 1.0))


def test_hetero_noisy_logits_shape():
    layer = geonnax.MCSoftmaxDenseFA.init(
        in_features=4,
        num_classes=3,
        rank=2,
        key=jr.PRNGKey(0),
        num_mc_samples=7,
    )
    x = jnp.ones((5, 4))
    logits = geonnax.hetero_noisy_logits(layer, x, key=jr.PRNGKey(2))
    assert logits.shape == (7, 5, 3)


def test_heteroscedastic_module_no_numpyro():
    import pathlib

    import geonnax.heteroscedastic as het

    src = pathlib.Path(het.__file__).read_text()
    assert "numpyro" not in src


# ---- Tier C: SNGP ----------------------------------------------------------


def test_laplace_random_feature_covariance_init_and_update():
    cov = geonnax.LaplaceRandomFeatureCovariance.init(4, momentum=0.9, ridge=1.0)
    assert cov.precision.shape == (4, 4)
    # init is `ridge * I` so variance_at on standard basis is 1 / (ridge + ridge) = 0.5
    feats = jnp.eye(4)
    var = cov.variance_at(feats)
    assert var.shape == (4,)
    # After update the returned container is a fresh instance.
    new_cov = cov.update(feats)
    assert new_cov is not cov
    assert new_cov.precision.shape == (4, 4)


def test_random_feature_gaussian_process_mean_shape():
    layer = geonnax.RandomFeatureGaussianProcess.init(
        in_features=3, num_features=16, out_features=2, key=jr.PRNGKey(0)
    )
    x = jnp.ones((5, 3))
    mean = layer(x)
    assert mean.shape == (5, 2)


def test_random_feature_gaussian_process_return_cov():
    layer = geonnax.RandomFeatureGaussianProcess.init(
        in_features=3, num_features=16, out_features=2, key=jr.PRNGKey(0)
    )
    x = jnp.ones((5, 3))
    mean, var = layer(x, return_cov=True)
    assert mean.shape == (5, 2)
    assert var.shape == (5,)


def test_random_feature_gaussian_process_update_precision():
    layer = geonnax.RandomFeatureGaussianProcess.init(
        in_features=3, num_features=8, out_features=1, key=jr.PRNGKey(0)
    )
    x = jnp.ones((4, 3))
    features = layer.feature_map(x)
    new_layer = layer.update_precision(features)
    # The container should be a fresh instance with an updated precision.
    assert new_layer is not layer
    assert new_layer.covariance.precision.shape == (8, 8)


def test_sngp_module_no_numpyro():
    import pathlib

    import geonnax.sngp as sngp_mod

    src = pathlib.Path(sngp_mod.__file__).read_text()
    assert "numpyro" not in src


def test_jit_round_trip_siren():
    """Sanity-check a JIT'd SIREN forward — exercise the static fields."""
    net = geonnax.SIREN.init(2, 8, 1, depth=3, key=jr.PRNGKey(0))
    f = jax.jit(lambda m, x: m(x))
    out = f(net, jnp.zeros((4, 2)))
    assert out.shape == (4, 1)


# ---- Tier D: deterministic linear core (variational dense family) ----------


def test_linear_core_init_and_forward():
    layer = geonnax.LinearCore.init(4, 3, key=jr.PRNGKey(0))
    assert layer.W.shape == (4, 3)
    assert layer.b is not None
    assert layer.b.shape == (3,)
    out = layer(jnp.ones((5, 4)))
    assert out.shape == (5, 3)


def test_linear_core_no_bias():
    layer = geonnax.LinearCore.init(4, 3, key=jr.PRNGKey(0), bias=False)
    assert layer.b is None
    out = layer(jnp.ones((2, 4)))
    assert out.shape == (2, 3)


def test_linear_core_matches_manual_matmul():
    layer = geonnax.LinearCore.init(4, 3, key=jr.PRNGKey(0))
    x = jnp.arange(8.0).reshape(2, 4)
    expected = x @ layer.W + layer.b
    assert jnp.allclose(layer(x), expected, atol=1e-6)


def test_dense_module_no_numpyro():
    import pathlib

    import geonnax.dense as dense_mod

    src = pathlib.Path(dense_mod.__file__).read_text()
    assert "numpyro" not in src


# ---- Tier D: DeepVSSGP deterministic core ----------------------------------


def test_deep_vssgp_core_init_and_forward():
    core = geonnax.DeepVSSGPCore.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=3,
        key=jr.PRNGKey(0),
        n_features=8,
    )
    assert len(core.W_freqs) == 3
    assert len(core.W_projs) == 3
    assert core.lengthscales.shape == (3,)
    assert core.W_freqs[0].shape == (2, 8)
    assert core.W_freqs[1].shape == (4, 8)
    assert core.W_projs[0].shape == (16, 4)
    assert core.W_projs[-1].shape == (16, 1)
    out = core(jnp.zeros((6, 2)))
    assert out.shape == (6, 1)


def test_deep_vssgp_core_depth_one():
    core = geonnax.DeepVSSGPCore.init(
        in_features=3,
        hidden_features=5,
        out_features=2,
        depth=1,
        key=jr.PRNGKey(1),
        n_features=4,
    )
    out = core(jnp.ones((7, 3)))
    assert out.shape == (7, 2)


def test_vssgp_module_no_numpyro():
    import pathlib

    import geonnax.vssgp as vssgp_mod

    src = pathlib.Path(vssgp_mod.__file__).read_text()
    assert "numpyro" not in src


# ---- Tier C/E: ensemble (BatchEnsemble / Rank-1) ----------------------------


def test_dense_rank1_init_and_call():
    layer = geonnax.DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
    )
    out = layer(jnp.ones((5, 4)))
    assert out.shape == (3, 5, 2)


def test_dense_rank1_no_bias_omits_bias():
    layer = geonnax.DenseRank1.init(
        jr.PRNGKey(1),
        in_features=3,
        out_features=2,
        ensemble_size=2,
        bias=False,
    )
    # With bias=False and x=0, output is purely the bias-free linear map → 0.
    out = layer(jnp.zeros((4, 3)))
    assert out.shape == (2, 4, 2)
    assert jnp.allclose(out, 0.0)


def test_layer_norm_ensemble_shape_preserved():
    ln = geonnax.LayerNormEnsemble.init(ensemble_size=3, feature_dim=4)
    x = jnp.ones((3, 5, 4))
    out = ln(x)
    assert out.shape == (3, 5, 4)


def test_layer_norm_ensemble_normalizes_to_zero_mean():
    ln = geonnax.LayerNormEnsemble.init(ensemble_size=2, feature_dim=4)
    key = jr.PRNGKey(0)
    x = jr.normal(key, (2, 6, 4))
    out = ln(x)
    # Default scales=1, biases=0 → per-slice mean ≈ 0, var ≈ 1.
    assert jnp.allclose(jnp.mean(out, axis=-1), 0.0, atol=1e-5)


def test_multi_head_attention_be_self_attention():
    mha = geonnax.MultiHeadAttentionBE.init(
        jr.PRNGKey(0),
        embed_dim=8,
        num_heads=2,
        ensemble_size=3,
    )
    x = jnp.ones((5, 8))
    out = mha(x, x, x)
    assert out.shape == (3, 5, 8)


def test_rank1_proj_helpers():
    proj = geonnax.init_rank1_proj(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        init_scale=0.5,
    )
    assert isinstance(proj, geonnax.Rank1ProjInit)
    x = jnp.ones((5, 4))
    out = geonnax.apply_rank1_proj(
        x, proj, ensemble_size=3, bias=True, has_ensemble=False
    )
    assert out.shape == (3, 5, 2)
    # Round-trip with has_ensemble=True.
    proj2 = geonnax.init_rank1_proj(
        jr.PRNGKey(1),
        in_features=2,
        out_features=2,
        ensemble_size=3,
        init_scale=0.5,
    )
    out2 = geonnax.apply_rank1_proj(
        out, proj2, ensemble_size=3, bias=True, has_ensemble=True
    )
    assert out2.shape == (3, 5, 2)


def test_jit_round_trip_dense_rank1():
    """JIT'd DenseRank1 forward — exercise the static fields under jax.jit."""
    layer = geonnax.DenseRank1.init(jr.PRNGKey(0), 4, 2, 3)
    f = jax.jit(lambda m, x: m(x))
    out = f(layer, jnp.ones((5, 4)))
    assert out.shape == (3, 5, 2)


def test_ensemble_module_no_numpyro():
    import geonnax.ensemble as ensemble_mod

    with open(ensemble_mod.__file__) as f:
        src = f.read()
    assert "numpyro" not in src


# ---- Tier B: conditioning ---------------------------------------------------


def test_concat_conditioner_shape():
    cond = geonnax.ConcatConditioner.init(num_features=8, cond_dim=4, key=jr.PRNGKey(0))
    h = jnp.ones((5, 8))
    z = jnp.ones((5, 4))
    out = cond(h, z)
    assert out.shape == (5, 8)


def test_concat_conditioner_broadcast_z():
    cond = geonnax.ConcatConditioner.init(num_features=8, cond_dim=4, key=jr.PRNGKey(0))
    out = cond(jnp.ones((5, 8)), jnp.ones((4,)))
    assert out.shape == (5, 8)


def test_affine_modulation_identity_at_init():
    """AffineModulation is identity at init (bias=0, one_plus_tanh)."""
    cond = geonnax.AffineModulation.init(num_features=6, cond_dim=3, key=jr.PRNGKey(0))
    h = jnp.arange(12.0).reshape(2, 6)
    # zero context => raw_gamma=0 => gamma = 1+tanh(0)=1, beta=0 => out == h.
    out = cond(h, jnp.zeros((2, 3)))
    assert jnp.allclose(out, h, atol=1e-6)


def test_film_alias_is_affine_modulation():
    assert geonnax.FiLM is geonnax.AffineModulation


def test_affine_modulation_log_det_exp_only():
    cond = geonnax.AffineModulation.init(
        num_features=4, cond_dim=2, key=jr.PRNGKey(0), gamma_activation="exp"
    )
    ldj = cond.log_det(jnp.zeros((3, 2)))
    assert ldj.shape == (3,)


def test_hyper_linear_shared_path():
    hyper = geonnax.HyperLinear.init(
        target_in=4, target_out=8, cond_dim=3, key=jr.PRNGKey(0)
    )
    out = hyper(jnp.ones((6, 4)), jnp.ones((3,)))
    assert out.shape == (6, 8)


def test_hyper_linear_per_sample_path():
    hyper = geonnax.HyperLinear.init(
        target_in=4, target_out=8, cond_dim=3, key=jr.PRNGKey(0)
    )
    out = hyper(jnp.ones((6, 4)), jnp.ones((6, 3)))
    assert out.shape == (6, 8)


def test_conditioned_inr_feature_mode_with_siren():
    key = jr.PRNGKey(0)
    inner = geonnax.SIREN.init(2, 16, 1, depth=4, key=key)
    wrapped = geonnax.ConditionedINR.init(
        inner,
        conditioner_cls=geonnax.AffineModulation,
        cond_dim=4,
        key=key,
    )
    out = wrapped(jnp.zeros((10, 2)), jnp.zeros((10, 4)))
    assert out.shape == (10, 1)


def test_conditioned_inr_input_mode_with_siren():
    key = jr.PRNGKey(0)
    inner = geonnax.SIREN.init(2, 16, 1, depth=3, key=key)
    wrapped = geonnax.ConditionedINR.init(
        inner,
        conditioner_cls=geonnax.ConcatConditioner,
        cond_dim=4,
        key=key,
        mode="input",
    )
    out = wrapped(jnp.zeros((10, 2)), jnp.zeros((10, 4)))
    assert out.shape == (10, 1)


def test_hyper_siren_forward():
    import equinox as eqx

    key = jr.PRNGKey(0)
    pnet_key, build_key = jr.split(key)
    pnet = eqx.nn.MLP(in_size=3, out_size=4, width_size=8, depth=2, key=pnet_key)
    net = geonnax.HyperSIREN(
        in_features=2,
        hidden_features=16,
        out_features=1,
        depth=3,
        cond_dim=4,
        parameter_net=pnet,
        key=build_key,
    )
    assert isinstance(net, geonnax.GeneratedSiren)
    out = net(jnp.zeros((5, 2)), jnp.zeros((3,)))
    assert out.shape == (5, 1)


def test_hyper_siren_single_point_squeeze():
    import equinox as eqx

    key = jr.PRNGKey(0)
    pnet_key, build_key = jr.split(key)
    pnet = eqx.nn.MLP(in_size=2, out_size=4, width_size=8, depth=2, key=pnet_key)
    net = geonnax.HyperSIREN(
        in_features=3,
        hidden_features=8,
        out_features=2,
        depth=2,
        cond_dim=4,
        parameter_net=pnet,
        key=build_key,
    )
    out = net(jnp.zeros(3), jnp.zeros(2))
    assert out.shape == (2,)


def test_conditioning_module_no_numpyro():
    import pathlib

    import geonnax.conditioning as cond_mod

    src = pathlib.Path(cond_mod.__file__).read_text()
    assert "numpyro" not in src
