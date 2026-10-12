"""Tests for `geonnax.geofno`: the learned chart `LearnedDeformation` and `GeoFNO`."""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from _point_invariance import assert_point_refinement_converges, grid_nodes

import geonnax
from geonnax.geofno import GeoFNO, LearnedDeformation


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


# === GeoFNO ==================================================================


def _geofno(d: int = 2, **kwargs) -> GeoFNO:
    lower, upper = BOXES[d]
    args = {
        "key": KEY,
        "lower": lower,
        "upper": upper,
        "hidden_channels": 8,
        "n_layers": 3,
        "deformation_kwargs": {"width": 16, "depth": 1, "n_frequencies": 3},
    } | kwargs
    return GeoFNO.init(2, 3, (3,) * d, (8,) * d, **args)


def _move_chart(model: GeoFNO) -> GeoFNO:
    """Give the MLP chart a non-zero readout, so φ is no longer the box map."""
    readout = model.deformation.mlp.layers[-1]
    weight = 0.1 * jr.normal(jr.PRNGKey(4), readout.weight.shape)
    return eqx.tree_at(lambda m: m.deformation.mlp.layers[-1].weight, model, weight)


def _cloud(d: int, n: int, key: jax.Array) -> tuple[jax.Array, ...]:
    """``(u (2, n), x (n, d), w (n,))`` with ``w = 1/n`` inside the box of ``d``."""
    k_x, k_u = jr.split(key)
    return jr.normal(k_u, (2, n)), _points(d, n, key=k_x), jnp.full((n,), 1.0 / n)


def _pad(u, x, w, n_max, key):
    """Zero-pad a cloud to ``n_max`` points at in-box coordinates, u = w = 0."""
    n_pad = n_max - x.shape[0]
    if n_pad == 0:
        return u, x, w
    x_pad = _points(x.shape[1], n_pad, key=key)
    return (
        einx.id("c n, c p -> c (n + p)", u, jnp.zeros((u.shape[0], n_pad))),
        einx.id("n d, p d -> (n + p) d", x, x_pad),
        einx.id("n, p -> (n + p)", w, jnp.zeros((n_pad,))),
    )


@pytest.mark.parametrize("d", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_geofno_shapes(d):
    model = _geofno(d)
    u, x, w = _cloud(d, 50, jr.PRNGKey(1))
    assert model(u, x, w).shape == (3, 50)
    x_out = _points(d, 17, key=jr.PRNGKey(2))
    assert model(u, x, w, x_out=x_out).shape == (3, 17)


def test_geofno_fields():
    model = _geofno(2, n_layers=4, projection_channels=5)
    assert len(model.blocks) == 2
    assert all(not blk.use_activation for blk in model.blocks)
    assert model.pos_bias is not None and len(model.pos_bias) == 3
    assert model.pos_bias_out is not None
    assert model.proj1.out_channels == 5
    assert model.latent_shape == (8, 8)
    assert model.deformation.kind == "mlp"
    assert not model.jacobian_weights
    plain = _geofno(2, n_layers=2, positional_bias=False)
    assert plain.blocks == [] and plain.pos_bias is None and plain.pos_bias_out is None


def test_geofno_exported_from_top_level():
    assert geonnax.GeoFNO is GeoFNO
    assert "GeoFNO" in geonnax.__all__
    assert "GeoFNO" in geonnax.geofno.__all__


def test_geofno_permutation_invariant():
    model = _move_chart(_geofno(2))
    u, x, w = _cloud(2, 80, jr.PRNGKey(1))
    perm = jr.permutation(jr.PRNGKey(2), 80)
    y = model(u, x, w)
    y_perm = model(u[:, perm], x[perm], w[perm])
    # Only the summation order of the encoder changes: float32 rounding.
    assert jnp.allclose(y_perm, y[:, perm], atol=1e-5)


def test_geofno_zero_weight_padding_is_bit_identical():
    # Padded points contribute exact zeros to the encoder (#80): for a fixed
    # padded size, the output at the real points is bit-identical whatever the
    # padding's coordinates and values are.
    model = _move_chart(_geofno(2))
    u, x, w = _cloud(2, 60, jr.PRNGKey(1))
    u_a, x_a, w_a = _pad(u, x, w, 90, jr.PRNGKey(3))
    _, x_b, _ = _pad(u, x, w, 90, jr.PRNGKey(4))
    u_b = u_a.at[:, 60:].set(jr.normal(jr.PRNGKey(6), (2, 30)))
    q = _points(2, 25, key=jr.PRNGKey(5))
    assert jnp.array_equal(model(u_a, x_a, w_a)[:, :60], model(u_b, x_b, w_a)[:, :60])
    assert jnp.array_equal(model(u_a, x_a, w_a, x_out=q), model(u_b, x_b, w_a, x_out=q))
    # Against the unpadded cloud only float32 rounding differs: XLA picks
    # different matmul / conv kernels for N = 60 and N = 90 in the lifting
    # and the chart MLP.
    assert jnp.allclose(model(u_a, x_a, w_a)[:, :60], model(u, x, w), atol=1e-6)


def test_geofno_ragged_vmap_matches_unbatched():
    model = _move_chart(_geofno(2))
    clouds = [_cloud(2, n, jr.PRNGKey(n)) for n in (70, 45)]
    padded = [_pad(*c, 70, jr.PRNGKey(9)) for c in clouds]
    ub, xb, wb = (jnp.stack(arrs) for arrs in zip(*padded, strict=True))
    yb = jax.vmap(model)(ub, xb, wb)
    assert yb.shape == (2, 3, 70)
    for b, (u, x, w) in enumerate(clouds):
        n = x.shape[0]
        assert jnp.allclose(yb[b, :, :n], model(u, x, w), atol=1e-5)


def test_geofno_identity_two_layers_is_hand_composed():
    # deformation="identity", n_layers=2, no positional bias: the model is
    # lifting -> from_points -> σ -> to_points -> projection.
    model = _geofno(2, n_layers=2, deformation="identity", positional_bias=False)
    u, x, w = _cloud(2, 60, jr.PRNGKey(1))
    x_out = _points(2, 20, key=jr.PRNGKey(2))
    act = model.activation
    xi, xi_out = model.deformation.box(x), model.deformation.box(x_out)
    h = act(model.encoder.from_points(model.lifting(u), xi, w, (8, 8)))
    ref = model.proj2(act(model.proj1(model.decoder.to_points(h, xi_out))))
    assert jnp.allclose(model(u, x, w, x_out=x_out), ref, atol=1e-6)


def test_geofno_jacobian_weights_at_init_are_normalised_weights():
    # At init det∇φ is the constant prod(scale), so the Jacobian weights are
    # w / Σw: with Σw = 1 the output is unchanged (up to rounding).
    model = _geofno(2)
    jac = _geofno(2, jacobian_weights=True)
    u, x, w = _cloud(2, 60, jr.PRNGKey(1))
    assert jnp.allclose(jac(u, x, w), model(u, x, w), atol=1e-5)
    assert jnp.allclose(jac(u, x, 3.0 * w), model(u, x, w), atol=1e-5)
    u_pad, x_pad, w_pad = _pad(u, x, w, 90, jr.PRNGKey(3))
    assert jnp.allclose(jac(u_pad, x_pad, w_pad)[:, :60], jac(u, x, w), atol=1e-6)


def test_geofno_code_conditioning():
    model = _geofno(2, code_dim=4)
    u, x, w = _cloud(2, 40, jr.PRNGKey(1))
    codes = jr.normal(jr.PRNGKey(2), (3, 4))
    y = jax.vmap(lambda c: model(u, x, w, code=c))(codes)
    assert y.shape == (3, 3, 40)
    with pytest.raises(ValueError, match="code must be given"):
        model(u, x, w)


# --- point-refinement convergence (the #82 harness) -------------------------


def _band_limited(xi):
    """A smooth periodic field inside a (3, 3)-mode band, (2, N) from xi (N, 2)."""
    a, b = xi[:, 0], xi[:, 1]
    two_pi = 2 * jnp.pi
    f = jnp.sin(two_pi * a) + jnp.cos(two_pi * (a + 2 * b))
    g = 0.5 * jnp.sin(two_pi * 3 * b) - jnp.cos(two_pi * (2 * a - b))
    return jnp.stack([f, g])


def _refinement_setup():
    # Identity chart on the unit box with pad = 0, so ξ = x and the stratified
    # clouds of the harness tile the whole latent torus. The lifted field is
    # band-limited (|k| <= 3), so on the 32 x 32 grid its (6, 6)-mode
    # coefficients are exact and the grid output is the continuum reference.
    model = GeoFNO.init(
        2,
        1,
        (6, 6),
        (16, 16),
        key=KEY,
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        hidden_channels=8,
        n_layers=3,
        deformation="identity",
        deformation_kwargs={"pad": 0.0},
    )
    q = jr.uniform(jr.PRNGKey(7), (200, 2))
    nodes = grid_nodes((32, 32))
    reference = model(_band_limited(nodes), nodes, jnp.full((1024,), 1 / 1024), x_out=q)
    return model, q, reference


def test_geofno_point_refinement_converges():
    # Jittered stratified clouds at N = 2025 / 8100, query output at 200 fixed
    # points against the dense-grid reference. Stratified quadrature converges
    # ~N^-1 (the error falls ~4x for 4x the points); eight harness keys
    # measured 0.012-0.016 / 0.0030-0.0039 (KEY: 0.015 / 0.0033), so the
    # bounds leave a factor ~2.5 over the worst key.
    model, q, reference = _refinement_setup()
    assert_point_refinement_converges(
        lambda u, xi, w: model(u, xi, w, x_out=q),
        _band_limited,
        reference,
        (45, 90),
        d=2,
        key=KEY,
        max_rel_err=(0.04, 0.01),
    )


def test_geofno_point_refinement_unweighted_fails():
    # Negative control: without the weights (w = 1, the reference Geo-FNO's
    # unweighted sum) the encoder output grows ~N (rel. error ~240 / ~910).
    model, q, reference = _refinement_setup()
    with pytest.raises(AssertionError, match="does not converge"):
        assert_point_refinement_converges(
            lambda u, xi, w: model(u, xi, jnp.ones_like(w), x_out=q),
            _band_limited,
            reference,
            (45, 90),
            d=2,
            key=KEY,
            max_rel_err=(0.04, 0.01),
        )


# --- jit + vmap over ragged clouds + grad -------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["mlp", "fourier"])
def test_geofno_vmap_jit_grad(kind):
    model = _geofno(2, deformation=kind, deformation_kwargs=None)
    # Ragged clouds of 64, 40 and 25 real points, zero-weight padded to 64.
    clouds = [_cloud(2, n, jr.PRNGKey(n)) for n in (64, 40, 25)]
    padded = [_pad(*c, 64, jr.PRNGKey(11)) for c in clouds]
    ub, xb, wb = (jnp.stack(arrs) for arrs in zip(*padded, strict=True))
    target = jr.normal(jr.PRNGKey(12), (3, 3, 64))

    y = eqx.filter_jit(lambda m, u, x, w: jax.vmap(m)(u, x, w))(model, ub, xb, wb)
    assert y.shape == (3, 3, 64)
    assert jnp.allclose(y[2, :, :25], model(*clouds[2]), atol=1e-5)

    @eqx.filter_jit
    @eqx.filter_grad
    def grad_fn(m):
        pred = jax.vmap(m)(ub, xb, wb)
        mask = einx.greater("b n, -> b n", wb, 0.0)  # loss on the real points only
        sq = einx.where("b n, b c n, -> b c n", mask, (pred - target) ** 2, 0.0)
        return jnp.sum(sq) / (3 * jnp.sum(mask))

    g = grad_fn(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    if kind == "mlp":
        dphi = g.deformation.mlp.layers[-1].weight
    else:
        dphi = g.deformation.coeffs
    assert float(jnp.max(jnp.abs(dphi))) > 0
    # The frozen box map gets no gradient.
    assert float(jnp.max(jnp.abs(g.deformation.scale))) == 0


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_layers": 1}, "n_layers must be >= 2"),
        ({"n_modes": (3,)}, "equal length"),
        ({"latent_shape": (8, 8, 8)}, "equal length"),
        ({"lower": [0.0], "upper": [1.0]}, "equal length"),
        (
            {"n_modes": (), "latent_shape": (), "lower": [], "upper": []},
            "non-empty",
        ),
        ({"latent_shape": (6, 8)}, r"must exceed 2 \* n_modes"),
        ({"deformation": "spline"}, "kind must be one of"),
        ({"deformation": "identity", "code_dim": 2}, "needs kind='mlp'"),
        ({"deformation_kwargs": {"pad": -1.0}}, "pad must be >= 0"),
    ],
)
def test_geofno_init_validation(kwargs, match):
    args = {
        "in_channels": 2,
        "out_channels": 3,
        "n_modes": (3, 3),
        "latent_shape": (8, 8),
        "key": KEY,
        "lower": [0.0, 0.0],
        "upper": [1.0, 1.0],
    } | kwargs
    with pytest.raises(ValueError, match=match):
        GeoFNO.init(**args)


def test_geofno_call_validation():
    model = _geofno(2)
    u, x, w = _cloud(2, 30, jr.PRNGKey(1))
    with pytest.raises(ValueError, match=r"x must have shape \(N, 2\)"):
        model(u, jnp.ones((30, 3)), w)
    with pytest.raises(ValueError, match=r"u must have shape"):
        model(jnp.ones((1, 30)), x, w)
    with pytest.raises(ValueError, match=r"u must have shape"):
        model(jnp.ones((2, 29)), x, w)
    with pytest.raises(ValueError, match=r"weights must have shape \(30,\)"):
        model(u, x, w[:-1])
    with pytest.raises(ValueError, match=r"x_out must have shape \(M, 2\)"):
        model(u, x, w, x_out=jnp.ones((5, 1)))
