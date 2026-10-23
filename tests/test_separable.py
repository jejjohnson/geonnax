"""Tests for geonnax.separable.SeparableINR."""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from geonnax import SIREN, SeparableINR


KEY = jr.PRNGKey(0)


def _grid_points(axes):
    return jnp.stack([g.ravel() for g in jnp.meshgrid(*axes, indexing="ij")], -1)


@pytest.mark.parametrize("num_axes", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
@pytest.mark.parametrize("rank", [1, 8])
@pytest.mark.parametrize("net", ["siren", "mlp"])
def test_on_grid_matches_pointwise(num_axes, rank, net):
    f = SeparableINR.init(
        num_axes, rank, hidden_features=8, depth=2, out_features=2, net=net, key=KEY
    )
    sizes = (5, 4, 3)[:num_axes]
    axes = tuple(jnp.linspace(-1.0, 1.0, n) for n in sizes)
    grid = f.on_grid(axes)
    assert grid.shape == (2, *sizes)
    pointwise = jax.vmap(f)(_grid_points(axes))  # (prod N, K)
    expected = jnp.moveaxis(pointwise.reshape(*sizes, 2), -1, 0)
    # Same arithmetic in a different contraction order: float32 round-off.
    np.testing.assert_allclose(grid, expected, atol=1e-5, rtol=1e-5)


def test_on_grid_under_vmap():
    f = SeparableINR.init(2, 2, hidden_features=4, depth=2, key=KEY)
    xs = jnp.linspace(0.0, 1.0, 3)
    ys = jnp.stack([jnp.linspace(0.0, 1.0, 4) + s for s in (0.0, 0.5)])
    out = jax.vmap(lambda y: f.on_grid((xs, y)))(ys)
    assert out.shape == (2, 1, 3, 4)
    np.testing.assert_allclose(out[1], f.on_grid((xs, ys[1])), atol=1e-6)


def test_call_is_the_rank_sum_of_axis_products():
    f = SeparableINR.init(2, 3, hidden_features=8, depth=2, out_features=2, key=KEY)
    x = jnp.array([0.3, -0.4])
    phi = [
        np.asarray(net(x[a : a + 1])).reshape(3, 2) for a, net in enumerate(f.axis_nets)
    ]
    np.testing.assert_allclose(f(x), (phi[0] * phi[1]).sum(0), atol=1e-6)


def test_axis_nets_are_sirens_by_default():
    f = SeparableINR.init(3, 4, hidden_features=8, depth=3, key=KEY)
    assert f.num_axes == 3 and len(f.axis_nets) == 3
    assert all(isinstance(n, SIREN) for n in f.axis_nets)
    # Distinct keys per axis.
    assert not np.allclose(f.axis_nets[0].layers[0].W, f.axis_nets[1].layers[0].W)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_axes": 0}, "num_axes"),
        ({"num_axes": 11}, "num_axes"),
        ({"rank": 0}, "rank"),
        ({"hidden_features": 0}, "rank"),
        ({"out_features": 0}, "rank"),
        ({"net": "rbf"}, "net"),
        ({"depth": 1}, "depth"),
        ({"depth": 0, "net": "mlp"}, "depth"),
    ],
)
def test_init_validation(kwargs, match):
    args = {"num_axes": 2, "rank": 2, "hidden_features": 4, "depth": 2} | kwargs
    with pytest.raises(ValueError, match=match):
        SeparableINR.init(**args, key=KEY)


def test_shape_validation():
    f = SeparableINR.init(2, 2, hidden_features=4, depth=2, key=KEY)
    with pytest.raises(ValueError, match="shape"):
        f(jnp.zeros(3))
    with pytest.raises(ValueError, match="one-dimensional"):
        f.on_grid((jnp.zeros(3),))
    with pytest.raises(ValueError, match="one-dimensional"):
        f.on_grid((jnp.zeros(3), jnp.zeros((3, 1))))


def test_jit_and_grad():
    f = SeparableINR.init(2, 4, hidden_features=8, depth=2, key=KEY)
    axes = (jnp.linspace(0, 1, 6), jnp.linspace(0, 1, 7))

    @eqx.filter_jit
    @eqx.filter_grad
    def loss_grad(model):
        return jnp.mean(model.on_grid(axes) ** 2)

    g = loss_grad(f)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    # Axis derivative of the field is finite (forward mode through one axis net).
    dfdx = jax.jacfwd(lambda x: f(x)[0])(jnp.array([0.2, 0.7]))
    assert dfdx.shape == (2,) and bool(jnp.all(jnp.isfinite(dfdx)))


@pytest.mark.integration
def test_fits_a_rank_one_target():
    # sin(πx)·cos(πy) is exactly rank 1. Over keys 0-4 the final MSE is
    # 1.5e-5 to 3.1e-5; the bound (0.4% of the target variance 0.25) leaves a
    # 30x margin.
    axes = (jnp.linspace(-1, 1, 24), jnp.linspace(-1, 1, 24))
    target = jnp.outer(jnp.sin(jnp.pi * axes[0]), jnp.cos(jnp.pi * axes[1]))[None]
    f = SeparableINR.init(2, 2, hidden_features=16, depth=3, first_omega=3.0, key=KEY)
    params, static = eqx.partition(f, eqx.is_array)
    zeros = jax.tree.map(jnp.zeros_like, params)
    lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8  # Adam (Kingma & Ba, 2015)

    @jax.jit
    def step(params, m, v, t):
        loss, g = jax.value_and_grad(
            lambda p: jnp.mean((eqx.combine(p, static).on_grid(axes) - target) ** 2)
        )(params)
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, g)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g**2, v, g)
        params = jax.tree.map(
            lambda p, m, v: (
                p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)
            ),
            params,
            m,
            v,
        )
        return params, m, v, loss

    m, v = zeros, zeros
    for t in range(1, 601):
        params, m, v, loss = step(params, m, v, t)
    assert float(loss) < 1e-3, float(loss)


def test_grid_points_helper_order():
    axes = (jnp.arange(2.0), jnp.arange(3.0))
    pts = _grid_points(axes)
    assert [tuple(map(float, p)) for p in pts] == list(
        itertools.product([0.0, 1.0], [0.0, 1.0, 2.0])
    )
