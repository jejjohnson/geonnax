"""Tests for geonnax.neural_field — the deterministic BayesNF core."""

from __future__ import annotations

import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import geonnax.neural_field as nf_mod
from geonnax import NeuralFieldMLP
from geonnax.basis import fourier_features, interaction_features, seasonal_features


KEY = jr.PRNGKey(0)

FULL = dict(
    fourier_degrees=(0, 3, 2),
    interactions=((1, 2), (0, 1)),
    seasonality_periods=(365.25, 7.0),
    num_seasonal_harmonics=(2, 1),
    time_col=0,
)


def _reference_batch(m: NeuralFieldMLP, x: jax.Array) -> jax.Array:
    """BayesNF forward on a batch, written as pyrox-nn's BayesianNeuralField is.

    Mirrors ``pyrox_nn._bnf.BayesianNeuralField.__call__`` step by step, with
    the sampled sites replaced by the module's fields.
    """
    scales = jnp.asarray(m.input_scales, dtype=jnp.float32)  # as pyrox-nn builds them
    scaled = x / (scales * jnp.exp(m.log_scale_adjustment))
    blocks = [scaled]
    for c, d in enumerate(m.fourier_degrees):
        if d > 0:
            blocks.append(fourier_features(scaled[:, c], d, rescale=True))
    if m.seasonality_periods and any(m.num_seasonal_harmonics):
        blocks.append(
            seasonal_features(
                x[:, m.time_col],
                m.seasonality_periods,
                m.num_seasonal_harmonics,
                rescale=True,
            )
        )
    if m.interactions:
        blocks.append(interaction_features(scaled, jnp.asarray(m.interactions)))
    h = jnp.concatenate(
        [b * jax.nn.softplus(g) for g, b in zip(m.feature_gain, blocks, strict=True)],
        axis=-1,
    )
    alpha = jax.nn.sigmoid(m.logit_activation_weight)
    for W, b, c in zip(m.layers_W, m.layers_b, m.layers_gain, strict=True):
        h = h / jnp.sqrt(h.shape[-1])
        z = jax.nn.softplus(c) * (h @ W + b)
        h = alpha * jax.nn.elu(z) + (1 - alpha) * jnp.tanh(z)
    h = h / jnp.sqrt(h.shape[-1])
    return jax.nn.softplus(m.output_gain) * (h @ m.output_W + m.output_b)


def _randomize(m: NeuralFieldMLP, key: jax.Array) -> NeuralFieldMLP:
    """Replace every array leaf with a random draw (a stand-in posterior sample)."""
    leaves, treedef = jax.tree_util.tree_flatten(m)
    keys = jr.split(key, len(leaves))
    new = [0.5 * jr.normal(k, leaf.shape) for k, leaf in zip(keys, leaves, strict=True)]
    return jax.tree_util.tree_unflatten(treedef, new)


@pytest.mark.parametrize("out_features", [1, 3])
@pytest.mark.parametrize("config", [{}, FULL])
def test_shapes(config, out_features):
    m = NeuralFieldMLP.init(
        (30.0, 2.0, 1.5),
        width=16,
        depth=2,
        out_features=out_features,
        key=KEY,
        **config,
    )
    assert m(jnp.array([1.0, 0.2, -0.3])).shape == (out_features,)
    assert jax.vmap(m)(jnp.ones((5, 3))).shape == (5, out_features)


def test_matches_bayesnf_reference_forward():
    m = _randomize(
        NeuralFieldMLP.init((30.0, 2.0, 1.5), width=16, depth=3, key=KEY, **FULL),
        jr.PRNGKey(1),
    )
    x = jr.normal(jr.PRNGKey(2), (32, 3)) * jnp.array([100.0, 1.0, 1.0])
    # float32: the same operations reordered (single example vs batch).
    np.testing.assert_allclose(
        jax.vmap(m)(x), _reference_batch(m, x), rtol=1e-5, atol=1e-5
    )


def test_feature_width_and_block_count():
    m = NeuralFieldMLP.init(
        (30.0, 2.0, 1.5), width=8, depth=1, extra_features=4, key=KEY, **FULL
    )
    # blocks: inputs (3), fourier col1 (6), fourier col2 (4), seasonal (2*(2+1)=6),
    # interactions (2), extra (4)
    assert m.feature_gain.shape == (6,)
    assert m.layers_W[0].shape == (3 + 6 + 4 + 6 + 2 + 4, 8)
    assert m.features(jnp.ones(3), jnp.ones(4)).shape == (25,)


def test_extra_block_enters_gated():
    m = NeuralFieldMLP.init((1.0, 1.0), width=4, depth=1, extra_features=3, key=KEY)
    extra = jnp.array([1.0, -2.0, 0.5])
    h = m.features(jnp.array([0.1, 0.2]), extra)
    np.testing.assert_allclose(h[-3:], extra, rtol=1e-6)  # unit gain at init
    m2 = eqx.tree_at(lambda m: m.feature_gain, m, m.feature_gain.at[-1].set(-50.0))
    np.testing.assert_allclose(
        m2.features(jnp.array([0.1, 0.2]), extra)[-3:], 0.0, atol=1e-6
    )


def test_init_is_unit_gain_and_zero_readout_gives_bias():
    m = NeuralFieldMLP.init((1.0, 1.0), width=4, depth=2, out_features=2, key=KEY)
    for g in [m.feature_gain, m.layers_gain, m.output_gain]:
        np.testing.assert_allclose(jax.nn.softplus(g), 1.0, rtol=1e-6)
    m = eqx.tree_at(
        lambda m: (m.output_W, m.output_b),
        m,
        (jnp.zeros((4, 2)), jnp.array([0.3, -1.0])),
    )
    np.testing.assert_allclose(
        m(jnp.array([5.0, -2.0])), jnp.array([0.3, -1.0]), rtol=1e-6
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(input_scales=()), "input_scales"),
        (dict(input_scales=(1.0, 0.0)), "input_scales"),
        (dict(input_scales=(1.0, float("nan"))), "input_scales"),
        (dict(width=0), "width"),
        (dict(fourier_degrees=(1,)), "one entry per input"),
        (dict(fourier_degrees=(1, -1)), ">= 0"),
        (dict(interactions=((0, 2),)), "interactions"),
        (dict(seasonality_periods=(1.0,)), "same length"),
        (dict(seasonality_periods=(0.0,), num_seasonal_harmonics=(1,)), "periods"),
        (dict(time_col=2), "time_col"),
    ],
)
def test_init_validation(kwargs, match):
    base = dict(input_scales=(1.0, 1.0), width=4, depth=1, key=KEY)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        NeuralFieldMLP.init(base.pop("input_scales"), **base)


def test_call_validation():
    m = NeuralFieldMLP.init((1.0, 1.0), width=4, depth=1, key=KEY)
    with pytest.raises(ValueError, match=r"jax\.vmap"):
        m(jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="extra must be given"):
        m(jnp.ones(2), jnp.ones(2))
    m_extra = NeuralFieldMLP.init(
        (1.0, 1.0), width=4, depth=1, extra_features=2, key=KEY
    )
    with pytest.raises(ValueError, match="extra must be given"):
        m_extra(jnp.ones(2))
    with pytest.raises(ValueError, match=r"extra must have shape \(2,\)"):
        m_extra(jnp.ones(2), jnp.ones(3))


def test_depth_zero_is_linear_readout_of_features():
    m = NeuralFieldMLP.init((2.0, 4.0), width=4, depth=0, key=KEY)
    x = jnp.array([1.0, 2.0])
    h = m.features(x)
    expected = (h / jnp.sqrt(h.shape[0])) @ m.output_W + m.output_b
    np.testing.assert_allclose(m(x), expected, rtol=1e-6)


@pytest.mark.integration
def test_jit_vmap_grad():
    m = NeuralFieldMLP.init((30.0, 2.0, 1.5), width=16, depth=2, key=KEY, **FULL)
    x = jr.normal(jr.PRNGKey(3), (64, 3))
    y = jnp.sin(x[:, 1])[:, None]

    @eqx.filter_jit
    def loss(m):
        return jnp.mean((jax.vmap(m)(x) - y) ** 2)

    grads = eqx.filter_grad(loss)(m)
    assert jnp.isfinite(loss(m))
    for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)):
        assert bool(jnp.all(jnp.isfinite(g)))


def test_neural_field_module_no_numpyro():
    src = pathlib.Path(nf_mod.__file__).read_text()
    assert "numpyro" not in src


def test_integer_inputs_are_promoted_before_scaling():
    # Integer day indices must not truncate non-integer scales (365.25 -> 365).
    m = NeuralFieldMLP.init((365.25, 1.5), width=8, depth=1, key=KEY)
    x = jnp.array([730, 3], dtype=jnp.int32)
    np.testing.assert_allclose(
        m.features(x)[:2], jnp.array([730 / 365.25, 2.0]), rtol=1e-6
    )
    np.testing.assert_allclose(m(x), m(x.astype(jnp.float32)), rtol=1e-6)


def test_hidden_parameters_are_immutable_tuples():
    m = NeuralFieldMLP.init((1.0, 1.0), width=4, depth=2, key=KEY)
    assert isinstance(m.layers_W, tuple) and isinstance(m.layers_b, tuple)
    swapped = eqx.tree_at(
        lambda m: m.layers_W, m, tuple(jnp.zeros_like(w) for w in m.layers_W)
    )
    assert all(bool(jnp.all(w == 0)) for w in swapped.layers_W)
