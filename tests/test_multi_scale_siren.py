"""Tests for `geonnax.multi_scale_siren.MultiScaleSIREN` (issue #10)."""

import itertools
import statistics

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax


def test_multi_scale_siren_single_example_shape():
    net = geonnax.MultiScaleSIREN.init(
        2, 32, 1, depth=4, num_scales=5, key=jr.PRNGKey(0)
    )
    assert net(jnp.ones(2)).shape == (1,)


def test_multi_scale_siren_vmap_shape():
    net = geonnax.MultiScaleSIREN.init(
        2, 32, 1, depth=4, num_scales=5, key=jr.PRNGKey(0)
    )
    assert jax.vmap(net)(jnp.ones((10, 2))).shape == (10, 1)


@pytest.mark.parametrize("num_scales", [1, 2, 5, 8])
def test_multi_scale_siren_sub_network_count(num_scales):
    net = geonnax.MultiScaleSIREN.init(
        2, 16, 1, depth=3, num_scales=num_scales, key=jr.PRNGKey(0)
    )
    assert len(net.sub_networks) == num_scales
    assert all(isinstance(sub, geonnax.SIREN) for sub in net.sub_networks)
    first_omegas = [sub.first_omega for sub in net.sub_networks]
    assert len(set(first_omegas)) == num_scales
    assert first_omegas == list(net.omega_0_values)


def test_multi_scale_siren_log_spaced_omegas_default():
    net = geonnax.MultiScaleSIREN.init(
        1,
        8,
        1,
        depth=3,
        num_scales=5,
        omega_min=1.0,
        omega_max=100.0,
        key=jr.PRNGKey(0),
    )
    assert jnp.allclose(jnp.array(net.omega_0_values), jnp.logspace(0, 2, 5), atol=1e-6)


def test_multi_scale_siren_explicit_omegas():
    omegas = (2.0, 10.0, 50.0)
    net = geonnax.MultiScaleSIREN.init(
        1, 8, 1, depth=3, num_scales=3, omega_0_values=omegas, key=jr.PRNGKey(0)
    )
    assert net.omega_0_values == omegas
    assert tuple(sub.first_omega for sub in net.sub_networks) == omegas
    assert tuple(sub.layers[0].omega for sub in net.sub_networks) == omegas


def test_multi_scale_siren_reduces_to_single_siren_when_K_is_1():
    """``num_scales=1`` is behaviourally identical to the SIREN it wraps."""
    net = geonnax.MultiScaleSIREN.init(
        2, 16, 1, depth=4, num_scales=1, omega_0_values=(30.0,), key=jr.PRNGKey(0)
    )
    (sub,) = net.sub_networks
    assert sub.first_omega == 30.0
    xs = jr.normal(jr.PRNGKey(1), (32, 2))
    assert jnp.allclose(jax.vmap(net)(xs), jax.vmap(sub)(xs), atol=1e-6)


def test_multi_scale_siren_sum_not_mean():
    net = geonnax.MultiScaleSIREN.init(
        2, 16, 3, depth=3, num_scales=3, key=jr.PRNGKey(0)
    )
    x = jr.normal(jr.PRNGKey(1), (2,))
    parts = jnp.stack([sub(x) for sub in net.sub_networks])
    assert jnp.allclose(net(x), parts.sum(axis=0), atol=1e-6)
    assert not jnp.allclose(net(x), parts.mean(axis=0), atol=1e-6)


def test_multi_scale_siren_jits():
    net = geonnax.MultiScaleSIREN.init(
        2, 16, 1, depth=3, num_scales=4, key=jr.PRNGKey(0)
    )
    out = jax.jit(lambda m, x: m(x))(net, jnp.ones(2))
    assert out.shape == (1,)
    assert jnp.all(jnp.isfinite(out))


def test_multi_scale_siren_gradient_flows_to_all_scales():
    net = geonnax.MultiScaleSIREN.init(
        2, 16, 1, depth=3, num_scales=4, key=jr.PRNGKey(0)
    )
    grads = eqx.filter_grad(lambda m, x: jnp.sum(m(x) ** 2))(net, jnp.ones(2))
    for sub_grad in grads.sub_networks:
        for layer in sub_grad.layers:
            assert jnp.all(jnp.isfinite(layer.W))
            assert jnp.any(layer.W != 0.0)


def test_multi_scale_siren_rejects_invalid_num_scales():
    with pytest.raises(ValueError, match="num_scales must be >= 1"):
        geonnax.MultiScaleSIREN.init(2, 16, 1, depth=3, num_scales=0, key=jr.PRNGKey(0))


def test_multi_scale_siren_rejects_mismatched_omega_list_length():
    with pytest.raises(ValueError, match="must have length num_scales=5"):
        geonnax.MultiScaleSIREN.init(
            2,
            16,
            1,
            depth=3,
            num_scales=5,
            omega_0_values=(1.0, 2.0),
            key=jr.PRNGKey(0),
        )


def test_multi_scale_siren_propagates_siren_validation():
    with pytest.raises(ValueError, match="depth must be >= 2"):
        geonnax.MultiScaleSIREN.init(2, 16, 1, depth=1, num_scales=3, key=jr.PRNGKey(0))


def test_multi_scale_siren_activation_variance_preserved_per_band():
    """Every band keeps hidden-layer variance in the SIREN-stable range."""
    net = geonnax.MultiScaleSIREN.init(
        2, 64, 1, depth=5, num_scales=5, key=jr.PRNGKey(0)
    )
    xs = jr.normal(jr.PRNGKey(1), (1024, 2))
    for sub in net.sub_networks:
        z = xs
        for layer in sub.layers[:-1]:
            z = jax.vmap(layer)(z)
            if layer.layer_type == "hidden":
                assert 0.3 <= float(jnp.var(z)) <= 3.0


def test_log_spaced_omegas_endpoints_and_ordering():
    omegas = geonnax.log_spaced_omegas(2.0, 64.0, 6)
    assert len(omegas) == 6
    assert omegas[0] == pytest.approx(2.0)
    assert omegas[-1] == pytest.approx(64.0)
    assert list(omegas) == sorted(omegas)
    # Log-spacing ⇒ a constant ratio between consecutive bands.
    ratios = [b / a for a, b in itertools.pairwise(omegas)]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


def test_log_spaced_omegas_single_scale():
    assert geonnax.log_spaced_omegas(30.0, 60.0, 1) == (30.0,)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((1.0, 10.0, 0), "num_scales must be >= 1"),
        ((0.0, 10.0, 3), "omega_min must be > 0"),
        ((10.0, 1.0, 3), "omega_max must be >= omega_min"),
    ],
)
def test_log_spaced_omegas_rejects_invalid_arguments(args, match):
    with pytest.raises(ValueError, match=match):
        geonnax.log_spaced_omegas(*args)


@eqx.filter_jit
def _fit_mse(net, X, Y, steps=1000, lr=1e-3):
    """Adam-fit ``net`` to ``(X, Y)`` and return the final MSE.

    A hand-rolled Adam keeps the test dependency-free — geonnax does not
    depend on optax. The whole loop is a single `jax.lax.scan` under
    `equinox.filter_jit`, so ``steps`` costs one compilation, not ``steps``.
    """
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def loss(p):
        return jnp.mean((jax.vmap(eqx.combine(p, static))(X) - Y) ** 2)

    b1, b2, eps = 0.9, 0.999, 1e-8

    def step(carry, t):
        p, m, v = carry
        grads = jax.grad(loss)(p)
        m = jax.tree.map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
        v = jax.tree.map(lambda a, g: b2 * a + (1 - b2) * g**2, v, grads)
        p = jax.tree.map(
            lambda p_, m_, v_: (
                p_ - lr * (m_ / (1 - b1**t)) / (jnp.sqrt(v_ / (1 - b2**t)) + eps)
            ),
            p,
            m,
            v,
        )
        return (p, m, v), None

    zeros = jax.tree.map(jnp.zeros_like, params)
    (fitted, _, _), _ = jax.lax.scan(
        step, (params, zeros, zeros), jnp.arange(1, steps + 1)
    )
    return loss(fitted)


@pytest.mark.slow
def test_multi_scale_siren_fits_mixed_frequency_signal_better_than_single_siren():
    r"""The reason the layer exists: mixed-band signals fit far better.

    Target $y(x) = \sin(2\pi x) + 0.5 \sin(16\pi x) + 0.3 \sin(80\pi x)$ has
    content in three well-separated bands. The baseline is a *parameter-matched*
    single SIREN at the usual ``first_omega=30`` — width 143 against five bands of
    width 64 — so the comparison isolates the banded architecture rather than
    capacity.

    Compared on the median over seeds, not per-seed: summing five sub-networks
    makes the initial output scale larger, and at this learning rate the
    occasional seed fails to settle. The median margin is ~4 orders of
    magnitude, far beyond the 2x the comparison asks for.
    """
    X = jnp.linspace(-1.0, 1.0, 256)[:, None]
    Y = (
        jnp.sin(2 * jnp.pi * X)
        + 0.5 * jnp.sin(2 * jnp.pi * 8 * X)
        + 0.3 * jnp.sin(2 * jnp.pi * 40 * X)
    )
    multi_mse, single_mse = [], []
    for seed in range(5):
        key = jr.PRNGKey(seed)
        multi = geonnax.MultiScaleSIREN.init(1, 64, 1, depth=4, num_scales=5, key=key)
        single = geonnax.SIREN.init(1, 143, 1, depth=4, first_omega=30.0, key=key)
        multi_mse.append(float(_fit_mse(multi, X, Y)))
        single_mse.append(float(_fit_mse(single, X, Y)))
    assert statistics.median(single_mse) >= 2.0 * statistics.median(multi_mse)


def test_multi_scale_siren_module_no_numpyro():
    import pathlib

    from geonnax import multi_scale_siren as mss_mod

    assert "numpyro" not in pathlib.Path(mss_mod.__file__).read_text()
