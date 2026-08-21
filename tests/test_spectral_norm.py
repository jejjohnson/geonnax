"""Tests for `geonnax.spectral_norm.SpectralNormalization`."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax


def _linear_from(weight, key):
    """An `eqx.nn.Linear` with no bias whose weight is exactly ``weight``."""
    m, n = weight.shape
    return eqx.tree_at(
        lambda layer: layer.weight,
        eqx.nn.Linear(n, m, use_bias=False, key=key),
        weight,
    )


def _advance(sn, n_calls, x):
    for _ in range(n_calls):
        sn, _ = sn(x)
    return sn


def test_spectral_norm_init_state_shapes():
    k1, k2 = jr.split(jr.PRNGKey(0))
    sn = geonnax.SpectralNormalization.init(eqx.nn.Linear(16, 8, key=k1), key=k2)
    assert sn.u.shape == (8,)
    assert sn.v.shape == (16,)
    assert jnp.allclose(jnp.linalg.norm(sn.u), 1.0, atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(sn.v), 1.0, atol=1e-6)


# Power iteration converges at rate (sigma_2 / sigma_1) ** (2 k), so the
# iteration count needed for a given tolerance is set by the spectral gap of
# the specific draw. 100 is comfortably enough for every seed below.
_N_ITERS_TO_CONVERGE = 100


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spectral_norm_estimated_sigma_matches_svd(seed):
    """Power iteration converges to the top singular value (issue #11 task iv-i)."""
    k_w, k_init = jr.split(jr.PRNGKey(seed))
    weight = jr.normal(k_w, (32, 16))
    layer = _linear_from(weight, k_w)
    sn = geonnax.SpectralNormalization.init(layer, key=k_init)
    sn = _advance(sn, _N_ITERS_TO_CONVERGE, jnp.ones(16))
    true_sigma = jnp.linalg.svd(weight, compute_uv=False)[0]
    assert jnp.abs(sn.estimated_sigma() - true_sigma) < 1e-3


def test_spectral_norm_normalized_weight_has_spectral_norm_coeff():
    k_w, k_init = jr.split(jr.PRNGKey(1))
    layer = _linear_from(jr.normal(k_w, (24, 12)), k_w)
    sn = geonnax.SpectralNormalization.init(layer, coeff=0.95, key=k_init)
    sn = _advance(sn, _N_ITERS_TO_CONVERGE, jnp.ones(12))
    sigma = jnp.linalg.svd(sn.normalized_layer().weight, compute_uv=False)[0]
    assert jnp.abs(sigma - 0.95) < 1e-3


def test_spectral_norm_bounds_output_norm():
    """``||SN(L)(x)|| <= coeff * ||x||`` for unit-norm ``x`` (task iv-ii)."""
    k_w, k_init, k_x = jr.split(jr.PRNGKey(2), 3)
    layer = _linear_from(jr.normal(k_w, (16, 16)), k_w)
    sn = geonnax.SpectralNormalization.init(layer, coeff=0.95, key=k_init)
    sn = _advance(sn, _N_ITERS_TO_CONVERGE, jnp.ones(16))
    xs = jr.normal(k_x, (64, 16))
    xs = xs / jnp.linalg.norm(xs, axis=-1, keepdims=True)
    outs = jax.vmap(lambda x: sn(x)[1])(xs)
    # Slack covers the residual power-iteration error: sigma is estimated from
    # below, so the realised norm sits just above `coeff`.
    assert jnp.all(jnp.linalg.norm(outs, axis=-1) <= 0.95 * (1 + 1e-3))


def test_spectral_norm_chain_lipschitz_bound():
    """A 3-layer stack is ``coeff ** 3``-Lipschitz (task iv-iv)."""
    coeff = 0.9
    keys = jr.split(jr.PRNGKey(3), 8)
    layers = [
        geonnax.SpectralNormalization.init(
            _linear_from(jr.normal(kw, (16, 16)), kw), coeff=coeff, key=ki
        )
        for kw, ki in zip(keys[:3], keys[3:6], strict=True)
    ]
    layers = [_advance(sn, _N_ITERS_TO_CONVERGE, jnp.ones(16)) for sn in layers]

    def chain(x):
        for sn in layers:
            _, x = sn(x)
        return x

    a, b = jr.normal(keys[6], (16,)), jr.normal(keys[7], (16,))
    lipschitz = jnp.linalg.norm(chain(a) - chain(b)) / jnp.linalg.norm(a - b)
    assert lipschitz <= coeff**3 * (1 + 1e-3)


def test_spectral_norm_lipschitz_bound_holds_with_bias():
    """Bias shifts the output but not the Lipschitz constant."""
    k_w, k_init, k_a, k_b = jr.split(jr.PRNGKey(4), 4)
    layer = eqx.tree_at(
        lambda m: m.weight, eqx.nn.Linear(16, 16, key=k_w), jr.normal(k_w, (16, 16))
    )
    sn = geonnax.SpectralNormalization.init(layer, coeff=0.95, key=k_init)
    sn = _advance(sn, _N_ITERS_TO_CONVERGE, jnp.ones(16))
    a, b = jr.normal(k_a, (16,)), jr.normal(k_b, (16,))
    ratio = jnp.linalg.norm(sn(a)[1] - sn(b)[1]) / jnp.linalg.norm(a - b)
    assert ratio <= 0.95 * (1 + 1e-3)


def test_spectral_norm_call_returns_updated_state_and_is_pure():
    k1, k2 = jr.split(jr.PRNGKey(5))
    sn = geonnax.SpectralNormalization.init(eqx.nn.Linear(8, 4, key=k1), key=k2)
    u_before, v_before = sn.u, sn.v
    new_sn, out = sn(jnp.ones(8))
    assert out.shape == (4,)
    assert jnp.any(new_sn.u != u_before) or jnp.any(new_sn.v != v_before)
    # `self` is untouched — the module is a pure pytree.
    assert jnp.array_equal(sn.u, u_before)
    assert jnp.array_equal(sn.v, v_before)


def test_spectral_norm_n_power_iterations_speeds_convergence():
    k_w, k_init = jr.split(jr.PRNGKey(6))
    weight = jr.normal(k_w, (16, 16))
    layer = _linear_from(weight, k_w)
    true_sigma = jnp.linalg.svd(weight, compute_uv=False)[0]
    slow = _advance(
        geonnax.SpectralNormalization.init(layer, n_power_iterations=1, key=k_init),
        1,
        jnp.ones(16),
    )
    fast = _advance(
        geonnax.SpectralNormalization.init(layer, n_power_iterations=10, key=k_init),
        1,
        jnp.ones(16),
    )
    assert jnp.abs(fast.estimated_sigma() - true_sigma) < jnp.abs(
        slow.estimated_sigma() - true_sigma
    )


def test_spectral_norm_jits():
    k1, k2 = jr.split(jr.PRNGKey(7))
    sn = geonnax.SpectralNormalization.init(
        eqx.nn.Linear(8, 4, use_bias=False, key=k1), key=k2
    )
    f = jax.jit(lambda m, x: m(x))
    new_sn, out = f(sn, jnp.ones(8))
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(jnp.isfinite(new_sn.u))


def test_spectral_norm_vmaps_over_a_batch_without_recompiling():
    k1, k2 = jr.split(jr.PRNGKey(8))
    sn = geonnax.SpectralNormalization.init(
        eqx.nn.Linear(8, 4, use_bias=False, key=k1), key=k2
    )
    forward = jax.jit(jax.vmap(lambda x: sn(x)[1]))
    assert forward(jnp.ones((16, 8))).shape == (16, 4)
    # (u, v) do not depend on x, so a second batch of the same shape reuses
    # the same trace — the state is not silently batched.
    assert forward(jnp.zeros((16, 8))).shape == (16, 4)


def test_spectral_norm_gradient_flows_to_wrapped_weight():
    k_w, k_init = jr.split(jr.PRNGKey(9))
    layer = _linear_from(jr.normal(k_w, (4, 4)), k_w)
    sn = _advance(
        geonnax.SpectralNormalization.init(layer, key=k_init),
        _N_ITERS_TO_CONVERGE,
        jnp.ones(4),
    )

    @eqx.filter_grad
    def grad_fn(module, x):
        return jnp.sum(module(x)[1] ** 2)

    grads = grad_fn(sn, jnp.ones(4))
    assert jnp.all(jnp.isfinite(grads.layer.weight))
    assert jnp.any(grads.layer.weight != 0.0)
    # (u, v) are detached from autodiff.
    assert jnp.allclose(grads.u, 0.0)
    assert jnp.allclose(grads.v, 0.0)


def test_spectral_norm_scales_up_a_small_weight():
    """Rescaling is unconditional: sigma is driven to ``coeff``, not clipped to it."""
    k_w, k_init = jr.split(jr.PRNGKey(10))
    layer = _linear_from(1e-3 * jr.normal(k_w, (8, 8)), k_w)
    sn = _advance(
        geonnax.SpectralNormalization.init(layer, coeff=1.0, key=k_init),
        _N_ITERS_TO_CONVERGE,
        jnp.ones(8),
    )
    sigma = jnp.linalg.svd(sn.normalized_layer().weight, compute_uv=False)[0]
    assert jnp.abs(sigma - 1.0) < 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spectral_norm_estimated_sigma_is_a_lower_bound(seed):
    """Power iteration approaches sigma from below — never overshoots it.

    This is what makes the Lipschitz contract approximate rather than
    enforced: dividing by an underestimate leaves the realised norm at
    ``coeff * sigma / sigma_hat >= coeff``.
    """
    k_w, k_init = jr.split(jr.PRNGKey(seed))
    weight = jr.normal(k_w, (16, 12))
    layer = _linear_from(weight, k_w)
    true_sigma = jnp.linalg.svd(weight, compute_uv=False)[0]
    sn = geonnax.SpectralNormalization.init(layer, key=k_init)
    for _ in range(_N_ITERS_TO_CONVERGE):
        assert sn.estimated_sigma() <= true_sigma * (1 + 1e-5)
        sn, _ = sn(jnp.ones(12))


def test_spectral_norm_estimated_sigma_is_non_negative_before_convergence():
    """A fresh random (u, v) must not yield a negative sigma — that would
    flip the sign of the normalised weight."""
    for seed in range(8):
        k_w, k_init = jr.split(jr.PRNGKey(seed))
        layer = _linear_from(jr.normal(k_w, (8, 8)), k_w)
        sn = geonnax.SpectralNormalization.init(layer, key=k_init)
        assert sn.estimated_sigma() >= 0.0


def test_spectral_norm_realised_norm_exceeds_coeff_before_convergence():
    """The documented failure mode, pinned: one iteration from a random start
    leaves the layer above its nominal bound."""
    k_w, k_init = jr.split(jr.PRNGKey(11))
    weight = jr.normal(k_w, (16, 16))
    sn, _ = geonnax.SpectralNormalization.init(
        _linear_from(weight, k_w), coeff=0.95, key=k_init
    )(jnp.ones(16))
    realised = jnp.linalg.svd(sn.normalized_layer().weight, compute_uv=False)[0]
    assert realised > 0.95
    # ... and it is exactly coeff * sigma / sigma_hat.
    true_sigma = jnp.linalg.svd(weight, compute_uv=False)[0]
    assert jnp.allclose(realised, 0.95 * true_sigma / sn.estimated_sigma(), rtol=1e-4)


def test_spectral_norm_zero_weight_stays_zero_instead_of_nan():
    """A zero-initialised layer already satisfies every Lipschitz bound."""
    k_w, k_init = jr.split(jr.PRNGKey(12))
    layer = _linear_from(jnp.zeros((8, 8)), k_w)
    sn = geonnax.SpectralNormalization.init(layer, key=k_init)
    for _ in range(3):
        sn, out = sn(jnp.ones(8))
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(out == 0.0)
    assert jnp.all(jnp.isfinite(sn.normalized_layer().weight))
    assert jnp.all(sn.normalized_layer().weight == 0.0)
    assert sn.estimated_sigma() == 0.0


def test_spectral_norm_zero_weight_gradient_is_finite():
    """The NaN from ``0 * (coeff / 0)`` would otherwise poison the backward pass."""
    k_w, k_init = jr.split(jr.PRNGKey(13))
    sn = geonnax.SpectralNormalization.init(
        _linear_from(jnp.zeros((4, 4)), k_w), key=k_init
    )
    sn, _ = sn(jnp.ones(4))
    grads = eqx.filter_grad(lambda m, x: jnp.sum(m(x)[1] ** 2))(sn, jnp.ones(4))
    assert jnp.all(jnp.isfinite(grads.layer.weight))


def test_spectral_norm_rejects_layer_without_weight():
    class NoWeight(eqx.Module):
        pass

    with pytest.raises(ValueError, match=r"must expose a `\.weight`"):
        geonnax.SpectralNormalization.init(NoWeight(), key=jr.PRNGKey(0))


def test_spectral_norm_rejects_non_2d_weight():
    k1, k2 = jr.split(jr.PRNGKey(0))
    conv = eqx.nn.Conv1d(3, 4, kernel_size=3, key=k1)
    with pytest.raises(ValueError, match="must be 2D"):
        geonnax.SpectralNormalization.init(conv, key=k2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coeff": 0.0}, "coeff must be > 0"),
        ({"n_power_iterations": 0}, "n_power_iterations must be >= 1"),
    ],
)
def test_spectral_norm_rejects_invalid_hyperparameters(kwargs, match):
    k1, k2 = jr.split(jr.PRNGKey(0))
    with pytest.raises(ValueError, match=match):
        geonnax.SpectralNormalization.init(
            eqx.nn.Linear(4, 4, key=k1), key=k2, **kwargs
        )


def test_spectral_norm_module_no_numpyro():
    import pathlib

    from geonnax import spectral_norm as spectral_norm_mod

    assert "numpyro" not in pathlib.Path(spectral_norm_mod.__file__).read_text()
