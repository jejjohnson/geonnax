"""Tests for the Edward2-style deterministic output heads (issue #12).

`geonnax.MixtureOfGaussiansDenseHead` and `geonnax.LinearChainCRF`. The CRF
recursions are checked against brute-force enumeration over all label
sequences for small ``T`` and ``K``.
"""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import geonnax


# ---- MixtureOfGaussiansDenseHead -------------------------------------------


def test_mog_head_parameter_shapes():
    head = geonnax.MixtureOfGaussiansDenseHead.init(64, 5, 3, key=jr.PRNGKey(0))
    pi, mu, log_sigma = head(jnp.ones(64))
    assert pi.shape == (5,)
    assert mu.shape == (5, 3)
    assert log_sigma.shape == (5, 3)


def test_mog_head_pi_is_a_simplex():
    head = geonnax.MixtureOfGaussiansDenseHead.init(8, 7, 2, key=jr.PRNGKey(1))
    pi, _, _ = head(jr.normal(jr.PRNGKey(2), (8,)))
    assert jnp.allclose(pi.sum(), 1.0, atol=1e-6)
    assert jnp.all(pi >= 0.0)


def test_mog_head_components_are_independent():
    """Each component has its own mean / log-scale projection."""
    head = geonnax.MixtureOfGaussiansDenseHead.init(8, 4, 2, key=jr.PRNGKey(3))
    assert len(head.mu_projs) == 4
    assert len(head.log_sigma_projs) == 4
    _, mu, _ = head(jr.normal(jr.PRNGKey(4), (8,)))
    # Distinct projections ⇒ distinct means (up to a measure-zero coincidence).
    assert not jnp.allclose(mu[0], mu[1])


def test_mog_head_vmaps_over_a_batch():
    head = geonnax.MixtureOfGaussiansDenseHead.init(4, 3, 2, key=jr.PRNGKey(5))
    pi, mu, log_sigma = jax.vmap(head)(jr.normal(jr.PRNGKey(6), (11, 4)))
    assert pi.shape == (11, 3)
    assert mu.shape == (11, 3, 2)
    assert log_sigma.shape == (11, 3, 2)
    assert jnp.allclose(pi.sum(axis=-1), 1.0, atol=1e-6)


def test_mog_head_jits():
    head = geonnax.MixtureOfGaussiansDenseHead.init(4, 3, 2, key=jr.PRNGKey(7))
    pi, mu, log_sigma = jax.jit(lambda m, x: m(x))(head, jnp.ones(4))
    assert all(jnp.all(jnp.isfinite(a)) for a in (pi, mu, log_sigma))


def test_mog_head_gradient_flows_to_every_component():
    head = geonnax.MixtureOfGaussiansDenseHead.init(4, 3, 2, key=jr.PRNGKey(8))

    @eqx.filter_grad
    def grad_fn(module, x):
        pi, mu, log_sigma = module(x)
        # `pi` is a simplex, so `sum(pi)` is constant — square it to get a
        # non-zero gradient through the gate projection.
        return jnp.sum(pi**2) + jnp.sum(mu**2) + jnp.sum(log_sigma**2)

    grads = grad_fn(head, jnp.ones(4))
    assert jnp.any(grads.pi_proj.weight != 0.0)
    for mu_g, sigma_g in zip(grads.mu_projs, grads.log_sigma_projs, strict=True):
        assert jnp.all(jnp.isfinite(mu_g.weight)) and jnp.any(mu_g.weight != 0.0)
        assert jnp.all(jnp.isfinite(sigma_g.weight)) and jnp.any(sigma_g.weight != 0.0)


def test_mog_head_composes_with_an_upstream_backbone():
    k_feat, k_head = jr.split(jr.PRNGKey(9))
    backbone = eqx.nn.MLP(8, 64, 64, depth=2, key=k_feat)
    head = geonnax.MixtureOfGaussiansDenseHead.init(64, 5, 1, key=k_head)
    pi, mu, log_sigma = head(backbone(jnp.ones(8)))
    assert (pi.shape, mu.shape, log_sigma.shape) == ((5,), (5, 1), (5, 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_features": 0, "num_components": 3, "num_outputs": 1},
        {"in_features": 4, "num_components": 0, "num_outputs": 1},
        {"in_features": 4, "num_components": 3, "num_outputs": 0},
    ],
)
def test_mog_head_rejects_non_positive_dimensions(kwargs):
    with pytest.raises(ValueError, match="must be > 0"):
        geonnax.MixtureOfGaussiansDenseHead.init(**kwargs, key=jr.PRNGKey(0))


# ---- LinearChainCRF --------------------------------------------------------


def _all_sequences(n_steps, n_labels):
    """Every label sequence of length ``n_steps`` as an ``(N, n_steps)`` array."""
    return jnp.array(list(itertools.product(range(n_labels), repeat=n_steps)))


def _brute_force_scores(crf, unary):
    """Unnormalised score of every label sequence, by explicit enumeration.

    Computed in NumPy so it is genuinely independent of the `jax.lax.scan`
    recursions under test.
    """
    n_steps, n_labels = unary.shape
    unary_np = np.asarray(unary)
    pairwise_np = np.asarray(crf.pairwise)
    sequences = np.asarray(_all_sequences(n_steps, n_labels))
    unary_part = unary_np[np.arange(n_steps), sequences].sum(axis=-1)
    pair_part = pairwise_np[sequences[:, :-1], sequences[:, 1:]].sum(axis=-1)
    return sequences, unary_part + pair_part


@pytest.mark.parametrize(("n_steps", "n_labels"), [(1, 3), (2, 4), (3, 3), (5, 2)])
def test_crf_log_partition_matches_brute_force(n_steps, n_labels):
    crf = geonnax.LinearChainCRF.init(n_labels, key=jr.PRNGKey(0), scale=0.7)
    unary = jr.normal(jr.PRNGKey(1), (n_steps, n_labels))
    _, scores = _brute_force_scores(crf, unary)
    expected = np.log(np.exp(scores).sum())
    assert jnp.allclose(crf.log_partition(unary), expected, atol=1e-4)


@pytest.mark.parametrize(("n_steps", "n_labels"), [(1, 3), (2, 4), (4, 3), (5, 4)])
def test_crf_log_prob_matches_brute_force(n_steps, n_labels):
    crf = geonnax.LinearChainCRF.init(n_labels, key=jr.PRNGKey(2), scale=0.7)
    unary = jr.normal(jr.PRNGKey(3), (n_steps, n_labels))
    sequences, scores = _brute_force_scores(crf, unary)
    log_z = np.log(np.exp(scores).sum())
    got = jax.vmap(lambda y: crf.log_prob(unary, y))(jnp.asarray(sequences))
    assert jnp.allclose(got, scores - log_z, atol=1e-4)


@pytest.mark.parametrize(("n_steps", "n_labels"), [(1, 4), (2, 3), (4, 4), (5, 3)])
def test_crf_log_prob_normalizes_over_all_sequences(n_steps, n_labels):
    crf = geonnax.LinearChainCRF.init(n_labels, key=jr.PRNGKey(4), scale=0.7)
    unary = jr.normal(jr.PRNGKey(5), (n_steps, n_labels))
    sequences = _all_sequences(n_steps, n_labels)
    log_probs = jax.vmap(lambda y: crf.log_prob(unary, y))(sequences)
    assert jnp.abs(jnp.exp(log_probs).sum() - 1.0) < 1e-4


@pytest.mark.parametrize(("n_steps", "n_labels"), [(1, 3), (2, 4), (4, 3), (5, 4)])
def test_crf_viterbi_matches_brute_force_argmax(n_steps, n_labels):
    crf = geonnax.LinearChainCRF.init(n_labels, key=jr.PRNGKey(6), scale=0.9)
    unary = jr.normal(jr.PRNGKey(7), (n_steps, n_labels))
    sequences, scores = _brute_force_scores(crf, unary)
    expected = sequences[int(np.argmax(scores))]
    assert crf.viterbi(unary).tolist() == expected.tolist()


def test_crf_viterbi_ignores_transitions_when_pairwise_is_zero():
    crf = geonnax.LinearChainCRF.init(4, key=jr.PRNGKey(8), scale=0.0)
    unary = jr.normal(jr.PRNGKey(9), (7, 4))
    assert crf.viterbi(unary).tolist() == jnp.argmax(unary, axis=-1).tolist()


def test_crf_log_partition_of_a_single_step_is_the_unary_logsumexp():
    crf = geonnax.LinearChainCRF.init(5, key=jr.PRNGKey(10), scale=0.9)
    unary = jr.normal(jr.PRNGKey(11), (1, 5))
    assert jnp.allclose(crf.log_partition(unary), jax.nn.logsumexp(unary[0]), atol=1e-6)


def test_crf_strong_transition_dominates_weak_unary():
    """A large pairwise bonus pulls the MAP path off the per-step argmax."""
    pairwise = jnp.full((3, 3), -10.0).at[0, 0].set(10.0)
    crf = eqx.tree_at(
        lambda m: m.pairwise,
        geonnax.LinearChainCRF.init(3, key=jr.PRNGKey(12)),
        pairwise,
    )
    unary = jnp.tile(jnp.array([0.0, 1.0, 0.5]), (4, 1))
    assert crf.viterbi(unary).tolist() == [0, 0, 0, 0]


def test_crf_jits_and_vmaps():
    crf = geonnax.LinearChainCRF.init(4, key=jr.PRNGKey(13), scale=0.5)
    unary = jr.normal(jr.PRNGKey(14), (6, 6, 4))  # (batch, T, K)
    log_z = jax.jit(jax.vmap(lambda u: crf.log_partition(u)))(unary)
    paths = jax.jit(jax.vmap(lambda u: crf.viterbi(u)))(unary)
    assert log_z.shape == (6,)
    assert paths.shape == (6, 6)
    assert jnp.all(jnp.isfinite(log_z))


def test_crf_gradient_flows_to_pairwise():
    crf = geonnax.LinearChainCRF.init(4, key=jr.PRNGKey(15), scale=0.5)
    unary = jr.normal(jr.PRNGKey(16), (6, 4))
    y = jr.randint(jr.PRNGKey(17), (6,), 0, 4)
    grads = eqx.filter_grad(lambda m: -m.log_prob(unary, y))(crf)
    assert jnp.all(jnp.isfinite(grads.pairwise))
    assert jnp.any(grads.pairwise != 0.0)


def test_crf_gradient_flows_to_upstream_unary_head():
    k_head, k_crf, k_feat = jr.split(jr.PRNGKey(18), 3)
    unary_head = eqx.nn.Linear(16, 5, key=k_head)
    crf = geonnax.LinearChainCRF.init(5, key=k_crf)
    features = jr.normal(k_feat, (8, 16))
    y = jr.randint(jr.PRNGKey(19), (8,), 0, 5)

    @eqx.filter_grad
    def loss(head):
        return -crf.log_prob(jax.vmap(head)(features), y)

    grads = loss(unary_head)
    assert jnp.all(jnp.isfinite(grads.weight))
    assert jnp.any(grads.weight != 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_labels": 0}, "num_labels must be > 0"),
        ({"num_labels": 3, "scale": -1.0}, "scale must be >= 0"),
    ],
)
def test_crf_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        geonnax.LinearChainCRF.init(key=jr.PRNGKey(0), **kwargs)


def test_head_modules_do_not_import_numpyro():
    """The heads stay pure-Equinox; numpyro may only appear as docstring prose."""
    import pathlib
    import re

    from geonnax import crf as crf_mod, mixture as mixture_mod

    numpyro_import = re.compile(r"^\s*(?:import numpyro|from numpyro)", re.MULTILINE)
    for module in (crf_mod, mixture_mod):
        source = pathlib.Path(module.__file__).read_text()
        assert not numpyro_import.search(source)
