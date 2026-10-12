"""Tests for `geonnax.geofno.LearnedDeformation`, the learned Geo-FNO chart."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import geonnax
from geonnax.geofno import LearnedDeformation


KEY = jr.PRNGKey(0)
KINDS = ["identity", "mlp", "fourier"]
# A lon/lat-like box far from the origin: the reference's multiplicative
# residual would not start at the identity here.
BOXES = {
    1: ([-3.0], [7.0]),
    2: ([-6.0, 35.0], [10.0, 45.0]),
    3: ([0.0, -1.0, 2.0], [1.0, 1.0, 5.0]),
}


def _make(kind: str, d: int = 2, **kwargs) -> LearnedDeformation:
    lower, upper = BOXES[d]
    small = {"width": 16, "depth": 2, "n_frequencies": 4} if kind == "mlp" else {}
    if kind == "fourier":
        small = {"n_basis": 4}
    return LearnedDeformation.init(lower, upper, key=KEY, kind=kind, **(small | kwargs))


def _points(d: int, n: int = 64, key: jax.Array | None = None) -> jax.Array:
    key = jr.PRNGKey(1) if key is None else key
    lower, upper = (jnp.asarray(v) for v in BOXES[d])
    u = jr.uniform(key, (n, d))
    return einx.add("d, n d", lower, einx.multiply("n d, d", u, upper - lower))


def _box(x: jax.Array, d: int, pad: float = 0.1) -> jax.Array:
    lower, upper = (jnp.asarray(v) for v in BOXES[d])
    b = 1.0 / (1.0 + pad)
    return einx.multiply(
        "n d, d", einx.subtract("n d, d", x, lower), b / (upper - lower)
    )


# --- identity at init ------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_identity_at_init(kind, d):
    phi = _make(kind, d)
    x = _points(d)
    xi = phi(x)
    assert xi.shape == x.shape
    assert jnp.allclose(xi, _box(x, d), atol=1e-6)
    assert jnp.allclose(xi, phi.box(x), atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("pad", [0.0, 0.1, 0.5])
def test_box_maps_into_latent_extent(kind, pad):
    phi = _make(kind, 2, pad=pad)
    b = 1.0 / (1.0 + pad)
    assert phi.extent == pytest.approx(b)
    x = _points(2, n=256)
    xi = phi(x)
    assert bool(jnp.all((xi >= -1e-6) & (xi <= b + 1e-6)))
    # The corners land on the corners of [0, b]^d.
    corners = jnp.asarray(BOXES[2])  # rows: lower, upper
    assert jnp.allclose(phi(corners), jnp.asarray([[0.0, 0.0], [b, b]]), atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_jacobian_det_at_init_is_prod_scale(kind):
    phi = _make(kind, 2)
    det = phi.jacobian_det(_points(2, n=16))
    assert det.shape == (16,)
    assert jnp.allclose(det, jnp.prod(phi.scale), rtol=1e-5)


def test_fields_per_kind():
    ident = _make("identity")
    assert ident.mlp is None and ident.frequencies is None and ident.coeffs is None
    assert ident.n_basis == ()

    mlp = _make("mlp")
    assert mlp.coeffs is None and mlp.mlp is not None
    assert mlp.frequencies is not None
    assert jnp.allclose(mlp.frequencies, jnp.pi * 2.0 ** jnp.arange(4))
    assert jnp.all(mlp.mlp.layers[-1].weight == 0)
    assert jnp.all(mlp.mlp.layers[-1].bias == 0)

    four = LearnedDeformation.init(
        [0.0, 0.0], [1.0, 2.0], kind="fourier", n_basis=(3, 5)
    )
    assert four.mlp is None and four.frequencies is None
    assert four.n_basis == (3, 5)
    assert four.coeffs is not None and four.coeffs.shape == (15, 2)
    assert jnp.all(four.coeffs == 0)


def test_exported_from_top_level():
    assert geonnax.LearnedDeformation is LearnedDeformation
    assert "LearnedDeformation" in geonnax.__all__
    assert "geofno" in geonnax.__all__


# --- the residuals once trained ---------------------------------------------


def _with_random_coeffs(phi: LearnedDeformation) -> LearnedDeformation:
    assert phi.coeffs is not None
    coeffs = 0.05 * jr.normal(jr.PRNGKey(3), phi.coeffs.shape)
    return eqx.tree_at(lambda m: m.coeffs, phi, coeffs)


def test_fourier_residual_vanishes_on_box_boundary():
    phi = _with_random_coeffs(_make("fourier", 2))
    lower, upper = BOXES[2]
    t = jnp.linspace(0.0, 1.0, 17)
    lo0, lo1 = lower
    hi0, hi1 = upper
    pair = "n, n -> n (1 + 1)"
    along0, along1 = lo0 + t * (hi0 - lo0), lo1 + t * (hi1 - lo1)
    x = einx.id(
        "n d, n d, n d, n d -> (n + n + n + n) d",
        einx.id(pair, jnp.full_like(t, lo0), along1),
        einx.id(pair, jnp.full_like(t, hi0), along1),
        einx.id(pair, along0, jnp.full_like(t, lo1)),
        einx.id(pair, along0, jnp.full_like(t, hi1)),
    )
    assert jnp.allclose(phi(x), phi.box(x), atol=1e-5)
    # ...and is non-trivial inside.
    inner = _points(2, n=32)
    assert float(jnp.max(jnp.abs(phi(inner) - phi.box(inner)))) > 1e-3


def test_mlp_residual_moves_points_once_readout_is_nonzero():
    phi = _make("mlp", 2)
    assert phi.mlp is not None
    w = phi.mlp.layers[-1].weight
    phi = eqx.tree_at(
        lambda m: m.mlp.layers[-1].weight, phi, 0.1 * jr.normal(jr.PRNGKey(4), w.shape)
    )
    x = _points(2)
    assert float(jnp.max(jnp.abs(phi(x) - phi.box(x)))) > 1e-3
    det = phi.jacobian_det(x)
    assert det.shape == (x.shape[0],)
    assert not jnp.allclose(det, jnp.prod(phi.scale), rtol=1e-3)


def test_jacobian_det_matches_finite_differences():
    phi = _with_random_coeffs(_make("fourier", 2))
    x = _points(2, n=4)
    h = 1e-2
    eye = jnp.eye(2) * h
    cols = [(phi(x + eye[a]) - phi(x - eye[a])) / (2 * h) for a in range(2)]
    jac = einx.id("n i, n i -> n i (1 + 1)", *cols)  # (N, d_out, d_in)
    # Central differences in float32 with h = 1e-2 on O(10) coordinates: the
    # truncation and round-off errors are ~1e-4 relative, so rtol=1e-2.
    assert jnp.allclose(phi.jacobian_det(x), jnp.linalg.det(jac), rtol=1e-2)


# --- code conditioning -------------------------------------------------------


def test_code_conditioning():
    phi = _make("mlp", 2, code_dim=3)
    assert phi.mlp is not None
    assert phi.mlp.in_size == 2 * (1 + 2 * 4) + 3
    x = _points(2)
    c1, c2 = jnp.ones(3), -jnp.ones(3)
    # Identity at init whatever the code.
    assert jnp.allclose(phi(x, c1), phi.box(x), atol=1e-6)
    assert jnp.allclose(phi.jacobian_det(x[:4], c1), jnp.prod(phi.scale), rtol=1e-5)
    # Once trained, different codes give different charts.
    w = phi.mlp.layers[-1].weight
    phi = eqx.tree_at(
        lambda m: m.mlp.layers[-1].weight, phi, 0.1 * jr.normal(jr.PRNGKey(5), w.shape)
    )
    assert not jnp.allclose(phi(x, c1), phi(x, c2), atol=1e-4)


def test_code_must_match_code_dim():
    x = _points(2)
    with pytest.raises(ValueError, match="code must be given"):
        _make("mlp", 2, code_dim=3)(x)
    with pytest.raises(ValueError, match="code must be given"):
        _make("mlp", 2)(x, jnp.ones(3))
    with pytest.raises(ValueError, match=r"code must have shape \(3,\)"):
        _make("mlp", 2, code_dim=3)(x, jnp.ones(4))
    with pytest.raises(ValueError, match=r"x must have shape \(N, 2\)"):
        _make("identity", 2)(jnp.ones((5, 3)))


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lower": [0.0, 0.0], "upper": [1.0]}, "equal length"),
        ({"lower": [], "upper": []}, "non-empty"),
        ({"lower": [0.0, 1.0], "upper": [1.0, 1.0]}, "upper must exceed lower"),
        ({"lower": [2.0], "upper": [1.0]}, "upper must exceed lower"),
        ({"pad": -0.1}, "pad must be >= 0"),
        ({"pad": float("nan")}, "pad must be >= 0"),
        ({"kind": "spline"}, "kind must be one of"),
        ({"kind": "mlp", "key": None}, "needs a PRNG key"),
        ({"kind": "mlp", "width": 0}, "width must be >= 1"),
        ({"kind": "mlp", "depth": -1}, "depth must be >= 0"),
        ({"kind": "mlp", "n_frequencies": 0}, "n_frequencies must be >= 1"),
        ({"kind": "fourier", "n_basis": 0}, "n_basis must be"),
        ({"kind": "fourier", "n_basis": (3,)}, "n_basis must be"),
        ({"code_dim": -1}, "code_dim must be >= 0"),
        ({"kind": "fourier", "code_dim": 2}, "needs kind='mlp'"),
        ({"kind": "identity", "code_dim": 2}, "needs kind='mlp'"),
    ],
)
def test_init_validation(kwargs, match):
    args = {"lower": [0.0, 0.0], "upper": [1.0, 1.0], "key": KEY} | kwargs
    with pytest.raises(ValueError, match=match):
        LearnedDeformation.init(**args)


@pytest.mark.parametrize("kind", ["identity", "fourier"])
def test_keyless_kinds(kind):
    phi = LearnedDeformation.init([0.0], [1.0], kind=kind)
    assert phi(jnp.asarray([[0.5]])).shape == (1, 1)


# --- jit / vmap / grad -------------------------------------------------------


@pytest.mark.parametrize("kind", ["mlp", "fourier"])
def test_grads_reach_residual_and_not_frozen_leaves(kind):
    phi = _make(kind, 2)
    x = _points(2, n=32)
    target = phi.box(x) + 0.01

    @eqx.filter_grad
    def grad_fn(m):
        return jnp.mean((m(x) - target) ** 2)

    g = grad_fn(phi)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    # Frozen leaves: stop_gradient at use.
    assert jnp.all(g.lower == 0) and jnp.all(g.scale == 0)
    if kind == "fourier":
        assert float(jnp.max(jnp.abs(g.coeffs))) > 0
    else:
        assert jnp.all(g.frequencies == 0)
        assert float(jnp.max(jnp.abs(g.mlp.layers[-1].weight))) > 0
        assert float(jnp.max(jnp.abs(g.mlp.layers[-1].bias))) > 0


@pytest.mark.parametrize("kind", KINDS)
def test_grad_wrt_x_is_scale_at_init(kind):
    phi = _make(kind, 2)
    x = _points(2, n=8)
    g = jax.grad(lambda z: jnp.sum(phi(z)))(x)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.allclose(
        g, einx.multiply("n d, d", jnp.ones_like(x), phi.scale), rtol=1e-5
    )


@pytest.mark.integration
@pytest.mark.parametrize("kind", KINDS)
def test_jit_vmap_grad(kind):
    phi = _make(kind, 2)
    batch = jnp.stack([_points(2, key=jr.PRNGKey(k)) for k in range(3)])  # (B, N, d)

    out = eqx.filter_jit(lambda m, xs: jax.vmap(m)(xs))(phi, batch)
    assert out.shape == batch.shape
    assert jnp.allclose(out[1], phi(batch[1]), atol=1e-6)

    dets = eqx.filter_jit(lambda m, xs: jax.vmap(m.jacobian_det)(xs))(phi, batch)
    assert dets.shape == batch.shape[:2]
    assert jnp.allclose(dets, math.prod(phi.scale.tolist()), rtol=1e-5)

    @eqx.filter_jit
    @eqx.filter_grad
    def grad_fn(m):
        xi = jax.vmap(m)(batch)
        fold = jnp.mean(jax.nn.relu(-jax.vmap(m.jacobian_det)(batch)))
        return jnp.mean(jnp.sin(10.0 * xi)) + fold

    g = grad_fn(phi)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    if kind == "fourier":
        assert float(jnp.max(jnp.abs(g.coeffs))) > 0
    elif kind == "mlp":
        assert float(jnp.max(jnp.abs(g.mlp.layers[-1].weight))) > 0


@pytest.mark.integration
def test_vmap_over_codes():
    phi = _make("mlp", 2, code_dim=3)
    x = _points(2)
    codes = jr.normal(jr.PRNGKey(6), (4, 3))
    out = jax.vmap(phi, in_axes=(None, 0))(x, codes)
    assert out.shape == (4, *x.shape)
