"""Tests for `BasisTransform` and `BasisSpectralConv`."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from geonnax.basis import (
    cotangent_laplacian_eigpairs,
    gauss_legendre_grid,
    icosphere,
    slepian_region_basis,
)
from geonnax.layers import (
    BasisSpectralConv,
    BasisTransform,
    SphericalHarmonicTransform,
    SphericalSpectralConv,
)
from geonnax.layers._spherical_transform import _gauss_legendre_grid


KEY = jr.PRNGKey(0)


def _orthonormal_transform(n_points: int = 40, n_modes: int = 7) -> BasisTransform:
    phi, _ = jnp.linalg.qr(jr.normal(KEY, (n_points, n_modes)))
    return BasisTransform.from_matrix(phi, jnp.ones(n_points))


def _identity_conv(channels: int, n_modes: int) -> BasisSpectralConv:
    conv = BasisSpectralConv.init(channels, channels, n_modes, key=KEY, bias=False)
    eye = jnp.broadcast_to(jnp.eye(channels), (n_modes, channels, channels))
    return eqx.tree_at(lambda c: c.weight, conv, eye)


def _cap_setup():
    points, weights = gauss_legendre_grid(12)
    cap = points[:, 2] > 0.5
    basis = slepian_region_basis(5, points[cap], weights[cap])
    return points, weights, cap, basis


# --- BasisTransform ---------------------------------------------------------


def test_from_matrix_shapes_and_tables():
    phi = jr.normal(KEY, (30, 5))
    w = jnp.linspace(0.5, 1.5, 30)
    phi_out = jr.normal(jr.PRNGKey(1), (12, 5))
    t = BasisTransform.from_matrix(phi, w, phi_out)
    assert t.n_modes == 5
    assert t.analysis.shape == (5, 30)
    assert t.synthesis.shape == (12, 5)
    assert jnp.allclose(t.analysis, einx.multiply("n m, n -> m n", phi, w))
    u = jr.normal(KEY, (3, 30))
    assert t.forward(u).shape == (3, 5)
    assert t.inverse(t.forward(u)).shape == (3, 12)


def test_orthonormal_round_trip():
    t = _orthonormal_transform()
    coeffs = jr.normal(jr.PRNGKey(2), (3, 7))
    assert jnp.allclose(t.forward(t.inverse(coeffs)), coeffs, atol=1e-5)
    u = t.inverse(coeffs)  # a field in span(Φ)
    assert jnp.allclose(t.inverse(t.forward(u)), u, atol=1e-5)


@pytest.mark.parametrize(
    ("phi_shape", "w_shape", "out_shape", "match"),
    [
        ((10,), (10,), None, "phi must be rank 2"),
        ((10, 3), (9,), None, "weights must be"),
        ((10, 3), (10,), (4, 2), "phi_out must have 3 modes"),
        ((10, 3), (10,), (4,), "phi_out must be rank 2"),
    ],
)
def test_from_matrix_validation(phi_shape, w_shape, out_shape, match):
    phi_out = None if out_shape is None else jnp.ones(out_shape)
    with pytest.raises(ValueError, match=match):
        BasisTransform.from_matrix(jnp.ones(phi_shape), jnp.ones(w_shape), phi_out)


# --- from_slepian -----------------------------------------------------------


def test_from_slepian_round_trip_in_cap():
    points, weights, cap, basis = _cap_setup()
    assert basis.num_modes == 9  # Shannon number of the 60° cap at l_max = 5
    t = BasisTransform.from_slepian(basis, points[cap], weights[cap])
    assert t.n_modes == 9
    assert t.analysis.shape == (9, int(cap.sum()))
    # A band-limited field concentrated in the cap: a mix of its Slepians.
    coeffs = jr.normal(jr.PRNGKey(3), (2, basis.num_modes))
    u = einx.dot("n k, c k -> c n", basis.evaluate(points[cap]), coeffs)
    recon = t.inverse(t.forward(u))
    rel = jnp.linalg.norm(recon - u) / jnp.linalg.norm(u)
    assert rel < 0.05
    assert jnp.allclose(t.forward(u), coeffs, atol=1e-4)


def test_from_slepian_output_anywhere():
    points, weights, cap, basis = _cap_setup()
    t = BasisTransform.from_slepian(
        basis, points[cap], weights[cap], points_out_xyz=points
    )
    assert t.synthesis.shape == (points.shape[0], 9)
    coeffs = jr.normal(jr.PRNGKey(4), (2, 9))
    u_cap = einx.dot("n k, c k -> c n", basis.evaluate(points[cap]), coeffs)
    u_all = einx.dot("n k, c k -> c n", basis.evaluate(points), coeffs)
    assert jnp.allclose(t.inverse(t.forward(u_cap)), u_all, atol=1e-4)


def test_from_slepian_validation():
    points, weights, cap, basis = _cap_setup()
    with pytest.raises(ValueError, match="points_xyz must be"):
        BasisTransform.from_slepian(basis, points[cap][:, :2], weights[cap])
    with pytest.raises(ValueError, match="weights must be"):
        BasisTransform.from_slepian(basis, points[cap], weights)
    with pytest.raises(ValueError, match="points_out_xyz must be"):
        BasisTransform.from_slepian(
            basis, points[cap], weights[cap], points_out_xyz=points[:, :2]
        )


# --- from_mesh --------------------------------------------------------------


def test_from_mesh_tables():
    vertices, triangles = icosphere(2)
    t = BasisTransform.from_mesh(vertices, triangles, num_basis=9)
    _, phi, areas = cotangent_laplacian_eigpairs(vertices, triangles, 9)
    assert t.n_modes == 9
    assert t.analysis.shape == (9, 162)
    assert t.synthesis.shape == (162, 9)
    assert jnp.allclose(t.synthesis, phi)
    assert jnp.allclose(t.analysis, einx.multiply("n m, n -> m n", phi, areas))
    # M-orthonormal: A Φ = Φᵀ diag(a) Φ = I.
    assert jnp.allclose(t.analysis @ t.synthesis, jnp.eye(9), atol=1e-5)


def test_from_mesh_identity_weights_reproduce_span():
    vertices, triangles = icosphere(2)
    t = BasisTransform.from_mesh(vertices, triangles, num_basis=9)
    conv = _identity_conv(2, 9)
    u = t.inverse(jr.normal(jr.PRNGKey(8), (2, 9)))  # a vertex field in span(Φ)
    assert jnp.allclose(conv(u, t), u, atol=1e-5)


def test_from_mesh_validation():
    vertices, triangles = icosphere(1)
    with pytest.raises(ValueError, match="num_basis"):
        BasisTransform.from_mesh(vertices, triangles, num_basis=0)


# --- BasisSpectralConv ------------------------------------------------------


def test_conv_shapes_and_init():
    t = _orthonormal_transform()
    conv = BasisSpectralConv.init(3, 5, 7, key=KEY)
    assert conv.weight.shape == (7, 5, 3)
    assert conv.bias is not None
    assert conv.bias.shape == (5, 1)
    assert conv(jnp.ones((3, 40)), t).shape == (5, 40)
    assert BasisSpectralConv.init(3, 5, 7, key=KEY, bias=False).bias is None


def test_conv_identity_weights_is_projector():
    t = _orthonormal_transform()
    conv = _identity_conv(2, 7)
    u = jr.normal(jr.PRNGKey(5), (2, 40))
    once = conv(u, t)
    assert jnp.allclose(conv(once, t), once, atol=1e-5)  # idempotent
    in_span = t.inverse(jr.normal(jr.PRNGKey(6), (2, 7)))
    assert jnp.allclose(conv(in_span, t), in_span, atol=1e-5)


def test_conv_matches_sfno_with_degree_tied_weights():
    n_lat, n_lon, l_max = 10, 20, 6
    sht = SphericalHarmonicTransform.init(n_lat, n_lon, l_max)
    _, w = _gauss_legendre_grid(n_lat, n_lon)
    t = BasisTransform.from_matrix(sht.synth, jnp.asarray(w))
    assert jnp.allclose(t.analysis, sht.analy, atol=1e-6)

    sconv = SphericalSpectralConv.init(3, 4, l_max, key=KEY)
    sconv = eqx.tree_at(lambda c: c.bias, sconv, jr.normal(KEY, (4, 1, 1)))
    conv = BasisSpectralConv(
        weight=sconv.weight[sconv.degree_index],
        bias=sconv.bias[:, :, 0],
        n_modes=t.n_modes,
        in_channels=3,
        out_channels=4,
    )
    x = jr.normal(jr.PRNGKey(7), (3, n_lat, n_lon))
    expected = sconv(x, sht)
    got = conv(einx.id("c h w -> c (h w)", x), t)
    assert jnp.allclose(got, einx.id("c h w -> c (h w)", expected), atol=1e-5)


def test_conv_slepian_shapes():
    points, weights, cap, basis = _cap_setup()
    t = BasisTransform.from_slepian(
        basis, points[cap], weights[cap], points_out_xyz=points
    )
    conv = BasisSpectralConv.init(2, 3, t.n_modes, key=KEY)
    assert conv(jnp.ones((2, int(cap.sum()))), t).shape == (3, points.shape[0])


def test_conv_validation():
    for kwargs, name in (
        ({"in_channels": 0, "out_channels": 2, "n_modes": 3}, "in_channels"),
        ({"in_channels": 2, "out_channels": 0, "n_modes": 3}, "out_channels"),
        ({"in_channels": 2, "out_channels": 2, "n_modes": 0}, "n_modes"),
    ):
        with pytest.raises(ValueError, match=name):
            BasisSpectralConv.init(**kwargs, key=KEY)
    conv = BasisSpectralConv.init(2, 2, 3, key=KEY)
    with pytest.raises(ValueError, match="transform has 7 modes"):
        conv(jnp.ones((2, 40)), _orthonormal_transform())


def test_frozen_tables_get_zero_gradient():
    t = _orthonormal_transform()
    conv = BasisSpectralConv.init(2, 3, 7, key=KEY)
    u = jr.normal(KEY, (2, 40))

    def loss(conv, t):
        return jnp.sum(conv(u, t) ** 2)

    g_conv, g_t = eqx.filter_grad(loss)(conv, t), jax.grad(loss, argnums=1)(conv, t)
    assert jnp.all(jnp.isfinite(g_conv.weight))
    assert jnp.any(g_conv.weight != 0)
    assert jnp.all(g_t.analysis == 0)
    assert jnp.all(g_t.synthesis == 0)


@pytest.mark.integration
def test_conv_jit_vmap_grad():
    points, weights, cap, basis = _cap_setup()
    t = BasisTransform.from_slepian(basis, points[cap], weights[cap])
    conv = BasisSpectralConv.init(2, 3, t.n_modes, key=KEY)
    batch = jr.normal(KEY, (4, 2, int(cap.sum())))

    @eqx.filter_jit
    def apply(conv, t, x):
        return jax.vmap(lambda xi: conv(xi, t))(x)

    out = apply(conv, t, batch)
    assert out.shape == (4, 3, int(cap.sum()))
    assert jnp.allclose(out[1], conv(batch[1], t), atol=1e-5)

    grads = eqx.filter_grad(lambda c: jnp.mean(apply(c, t, batch) ** 2))(conv)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
