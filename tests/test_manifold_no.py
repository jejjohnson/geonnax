"""Tests for `ManifoldNO` and `ManifoldNOBlock`."""

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
from geonnax.layers import BasisTransform
from geonnax.manifold_no import ManifoldNO, ManifoldNOBlock


KEY = jr.PRNGKey(0)
# The first 9 cotangent Laplace–Beltrami modes of an icosphere are the
# degrees l = 0, 1, 1, 1, 2, 2, 2, 2, 2 (eigenvalues ≈ l(l + 1)); this maps
# each mode to the first mode of its degree cluster.
_DEGREE_HEAD = jnp.array([0, 1, 1, 1, 4, 4, 4, 4, 4])


def _mesh_transform(level: int = 2) -> BasisTransform:
    return BasisTransform.from_mesh(*icosphere(level), num_basis=9)


def _slepian_transform() -> BasisTransform:
    points, weights = gauss_legendre_grid(12)
    cap = points[:, 2] > 0.5
    basis = slepian_region_basis(5, points[cap], weights[cap])
    return BasisTransform.from_slepian(basis, points[cap], weights[cap])


def _smooth_field(vertices):
    """A two-channel field of spherical-harmonic degree <= 2, (2, V)."""
    x, y, z = einx.id("v d -> d v", vertices)
    return jnp.stack([x + 0.5 * z, x * y - 0.3 * z * z])


def _tie_by_degree(op: ManifoldNO) -> ManifoldNO:
    """Give every mode of a degree cluster the same mixing matrix."""
    return eqx.tree_at(
        lambda o: [b.spectral.weight for b in o.blocks],
        op,
        [b.spectral.weight[_DEGREE_HEAD] for b in op.blocks],
    )


# --- shapes -----------------------------------------------------------------


def test_block_shape():
    t = _mesh_transform()
    blk = ManifoldNOBlock.init(5, t.n_modes, key=KEY)
    assert blk(jnp.ones((5, 162)), t).shape == (5, 162)
    assert blk.use_activation


def test_mesh_shapes_and_structure():
    t = _mesh_transform()
    op = ManifoldNO.init(
        3, 2, t, key=KEY, hidden_channels=8, n_layers=3, projection_channels=12
    )
    assert op(jnp.ones((3, 162))).shape == (2, 162)
    assert len(op.blocks) == 3
    # The last block is linear before projection, as in FNO.
    assert [b.use_activation for b in op.blocks] == [True, True, False]
    assert op.proj1.out_channels == 12
    assert (op.in_channels, op.out_channels, op.hidden_channels) == (3, 2, 8)
    assert op.transform is t


def test_slepian_shapes():
    t = _slepian_transform()
    op = ManifoldNO.init(2, 1, t, key=KEY, hidden_channels=8, n_layers=2)
    u = jr.normal(KEY, (2, t.analysis.shape[1]))
    y = op(u)
    assert y.shape == (1, 96)
    assert bool(jnp.all(jnp.isfinite(y)))


# --- validation -------------------------------------------------------------


def test_rejects_output_points_differing_from_input_points():
    points, weights = gauss_legendre_grid(12)
    cap = points[:, 2] > 0.5
    basis = slepian_region_basis(5, points[cap], weights[cap])
    t = BasisTransform.from_slepian(
        basis, points[cap], weights[cap], points_out_xyz=points
    )
    with pytest.raises(ValueError, match="N_out == N"):
        ManifoldNO.init(1, 1, t, key=KEY)


def test_rejects_zero_layers():
    with pytest.raises(ValueError, match="n_layers"):
        ManifoldNO.init(1, 1, _mesh_transform(), key=KEY, n_layers=0)


def test_rejects_bad_channels():
    with pytest.raises(ValueError, match="in_channels"):
        ManifoldNOBlock.init(0, 9, key=KEY)


def test_rejects_transform_with_other_mode_count():
    op = ManifoldNO.init(1, 1, _mesh_transform(), key=KEY, n_layers=1)
    other = BasisTransform.from_mesh(*icosphere(2), num_basis=4)
    op = eqx.tree_at(lambda o: o.transform, op, other)
    with pytest.raises(ValueError, match="modes"):
        op(jnp.ones((1, 162)))


# --- mesh refinement --------------------------------------------------------


def test_mesh_refinement_agrees_at_shared_vertices():
    """The same weights on icosphere(2) and icosphere(3) agree at shared vertices.

    Inline point-refinement check; switch to the #82 harness
    (``tests/_point_invariance.py``) once it lands.

    The weights are per mode, so the modes must match in order. Signs cancel
    (each mode enters once in the analysis and once in the synthesis), but
    ``eigh`` picks an arbitrary rotation inside each degenerate degree
    cluster, which differs between meshes. Tying the weights by degree (as
    `SphericalSpectralConv` does) makes the operator independent of that
    choice; ``num_basis = 9`` is the complete set of degrees l <= 2.
    """
    v2, t2 = icosphere(2)
    v3, t3 = icosphere(3)
    # icosphere(3) keeps icosphere(2)'s vertices first.
    assert jnp.array_equal(v3[: v2.shape[0]], v2)
    coarse = BasisTransform.from_mesh(v2, t2, num_basis=9)
    fine = BasisTransform.from_mesh(v3, t3, num_basis=9)
    op = ManifoldNO.init(2, 1, coarse, key=KEY, hidden_channels=8, n_layers=3)
    op = _tie_by_degree(op)
    op_fine = eqx.tree_at(lambda o: o.transform, op, fine)

    y_coarse = op(_smooth_field(v2))
    y_fine = op_fine(_smooth_field(v3))[:, : v2.shape[0]]
    rel = jnp.linalg.norm(y_coarse - y_fine) / jnp.linalg.norm(y_coarse)
    # Measured 4e-4: the cotangent eigenvectors and vertex areas of the two
    # meshes are O(h²) discretisations of the same l <= 2 harmonics and
    # surface measure. 1e-2 leaves room for the eigensolver; untied random
    # weights give ~0.2 (the cluster rotation), so the bound is meaningful.
    assert float(rel) < 1e-2


def test_mesh_refinement_needs_the_quadrature_weights():
    """Negative control: dropping the vertex areas (w = 1) breaks refinement.

    Without the mass the analysis scales with the vertex count, so the same
    weights see coefficients about 4× larger on icosphere(3) than on
    icosphere(2). This is why `BasisTransform` carries the weights.
    """
    outs = []
    for level in (2, 3):
        vertices, triangles = icosphere(level)
        _, phi, _ = cotangent_laplacian_eigpairs(vertices, triangles, 9)
        t = BasisTransform.from_matrix(phi, jnp.ones(vertices.shape[0]))
        op = ManifoldNO.init(2, 1, t, key=KEY, hidden_channels=8, n_layers=3)
        outs.append(_tie_by_degree(op)(_smooth_field(vertices))[:, :162])
    rel = jnp.linalg.norm(outs[0] - outs[1]) / jnp.linalg.norm(outs[0])
    assert float(rel) > 0.1


# --- jit / vmap / grad ------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("make_transform", [_mesh_transform, _slepian_transform])
def test_vmap_jit_grad_and_frozen_transform(make_transform):
    t = make_transform()
    n = t.analysis.shape[1]
    op = ManifoldNO.init(2, 1, t, key=KEY, hidden_channels=8, n_layers=2)
    xs = jr.normal(KEY, (4, 2, n))

    ys = jax.vmap(op)(xs)
    assert ys.shape == (4, 1, n)
    y_jit = eqx.filter_jit(lambda m, x: jax.vmap(m)(x))(op, xs)
    assert jnp.allclose(y_jit, ys, atol=1e-5)

    def loss(m):
        return jnp.mean(jax.vmap(m)(xs) ** 2)

    grads = eqx.filter_jit(eqx.filter_grad(loss))(op)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert float(jnp.abs(grads.blocks[0].spectral.weight).max()) > 0.0
    assert float(jnp.abs(grads.lifting.weight).max()) > 0.0
    # The basis tables sit behind stop_gradient: they never train.
    assert not bool(jnp.any(grads.transform.analysis))
    assert not bool(jnp.any(grads.transform.synthesis))
