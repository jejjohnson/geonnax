"""Tests for the cotangent Laplace–Beltrami eigenbasis of a triangle mesh (#86)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import cotangent_laplacian_eigpairs, icosphere, mesh_vertex_areas


def _square_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit square ``[0, 1]²`` at ``z = 0``: an ``n × n`` grid, 2n² triangles."""
    s = np.linspace(0.0, 1.0, n + 1)
    x, y = np.meshgrid(s, s, indexing="ij")
    vertices = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return vertices, triangles


def _reference_laplacian(vertices, triangles) -> np.ndarray:
    """The issue's loop prototype: an independent assembly of the cotangent L."""
    V, T = np.asarray(vertices, dtype=float), np.asarray(triangles)
    L = np.zeros((len(V), len(V)))
    for tri in T:
        for i in range(3):
            p, q, r = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
            e1, e2 = V[q] - V[p], V[r] - V[p]
            cot = e1 @ e2 / np.linalg.norm(np.cross(e1, e2))
            L[q, r] -= 0.5 * cot
            L[r, q] -= 0.5 * cot
    np.fill_diagonal(L, -L.sum(1))
    return L


@pytest.fixture(scope="module")
def sphere3():
    vertices, triangles = icosphere(3)
    eigvals, eigvecs, areas = cotangent_laplacian_eigpairs(vertices, triangles, 16)
    return vertices, triangles, np.asarray(eigvals), np.asarray(eigvecs), areas


def test_shapes_and_types(sphere3):
    vertices, _, eigvals, eigvecs, areas = sphere3
    assert eigvals.shape == (16,)
    assert eigvecs.shape == (vertices.shape[0], 16)
    assert areas.shape == (vertices.shape[0],)
    assert isinstance(areas, jnp.ndarray)


def test_icosphere_spectrum_approximates_l_l_plus_1(sphere3):
    eigvals = sphere3[2]
    expected = np.repeat([0.0, 2.0, 6.0, 12.0], [1, 3, 5, 7])
    assert eigvals[0] == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(eigvals[1:], expected[1:], rtol=0.02)
    assert np.all(np.diff(eigvals) >= -1e-5)  # ascending


def test_icosphere_multiplicities(sphere3):
    eigvals = sphere3[2]
    # Clusters: consecutive eigenvalues within 2 % belong to one degree ℓ.
    gaps = np.flatnonzero(np.diff(eigvals) > 0.1 * np.maximum(eigvals[1:], 1.0))
    sizes = np.diff(np.concatenate([[0], gaps + 1, [eigvals.size]]))
    assert sizes.tolist() == [1, 3, 5, 7]


def test_eigvecs_mass_orthonormal(sphere3):
    _, _, _, eigvecs, areas = sphere3
    gram = eigvecs.T @ (np.asarray(areas)[:, None] * eigvecs)
    np.testing.assert_allclose(gram, np.eye(16), atol=1e-5)


def test_constant_in_null_space(sphere3):
    _, _, _, eigvecs, areas = sphere3
    phi0 = eigvecs[:, 0]
    total = float(np.asarray(areas).sum())
    np.testing.assert_allclose(np.abs(phi0), 1.0 / math.sqrt(total), rtol=1e-4)


def test_generalised_eigen_equation_against_reference_assembly():
    vertices, triangles = icosphere(2)
    eigvals, eigvecs, areas = map(
        np.asarray, cotangent_laplacian_eigpairs(vertices, triangles, 10)
    )
    L = _reference_laplacian(vertices, triangles)
    residual = L @ eigvecs - areas[:, None] * eigvecs * eigvals[None, :]
    assert np.abs(residual).max() < 1e-5
    assert np.abs(L @ np.ones(L.shape[0])).max() < 1e-10  # constants in null space


@pytest.mark.parametrize("radius", [1.0, 2.0])
def test_vertex_areas_sum_to_surface_area(radius):
    vertices, triangles = icosphere(3, radius=radius)
    areas = mesh_vertex_areas(vertices, triangles)
    assert float(areas.sum()) == pytest.approx(4 * math.pi * radius**2, rel=0.01)
    assert np.all(np.asarray(areas) > 0)


def test_vertex_areas_match_eigpairs_areas(sphere3):
    vertices, triangles, *_, areas = sphere3
    np.testing.assert_allclose(mesh_vertex_areas(vertices, triangles), areas)


def test_vertex_areas_orientation_independent():
    vertices, triangles = icosphere(1)
    flipped = np.asarray(triangles)[:, ::-1]
    np.testing.assert_allclose(
        mesh_vertex_areas(vertices, triangles), mesh_vertex_areas(vertices, flipped)
    )


def test_flat_square_matches_neumann_spectrum():
    vertices, triangles = _square_mesh(20)
    eigvals, _, areas = cotangent_laplacian_eigpairs(vertices, triangles, 6)
    expected = np.pi**2 * np.array([0, 1, 1, 2, 4, 4], dtype=float)
    assert float(areas.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(eigvals[0]) == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(np.asarray(eigvals)[1:], expected[1:], rtol=0.05)


def test_single_triangle_areas():
    vertices = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    triangles = jnp.array([[0, 1, 2]])
    np.testing.assert_allclose(mesh_vertex_areas(vertices, triangles), [1.0] * 3)


@pytest.mark.slow
def test_icosphere_convergence_under_refinement():
    exact = np.repeat([0.0, 2.0, 6.0, 12.0], [1, 3, 5, 7])[1:]
    errors = []
    for level in (2, 3, 4):
        vertices, triangles = icosphere(level)
        eigvals, _, _ = cotangent_laplacian_eigpairs(vertices, triangles, 16)
        errors.append(np.abs(np.asarray(eigvals)[1:] / exact - 1).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.005


# --- validation ----------------------------------------------------------------


def _tet():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, triangles


def test_tetrahedron_runs():
    vertices, triangles = _tet()
    eigvals, eigvecs, _ = cotangent_laplacian_eigpairs(vertices, triangles, 4)
    assert eigvals.shape == (4,) and eigvecs.shape == (4, 4)
    assert float(eigvals[0]) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    ("vertices", "triangles", "num_basis", "match"),
    [
        (np.zeros((4, 2)), _tet()[1], 2, "vertices must have shape"),
        (_tet()[0], _tet()[1][:, :2], 2, "triangles must have shape"),
        (_tet()[0], np.zeros((0, 3), dtype=int), 2, "triangles must have shape"),
        (_tet()[0], _tet()[1].astype(float), 2, "integer"),
        (_tet()[0], _tet()[1] + 1, 2, "indices must lie"),
        (_tet()[0], _tet()[1] - 1, 2, "indices must lie"),
        (np.where(np.eye(4, 3) > 0, np.nan, 0.0), _tet()[1], 2, "finite"),
        (_tet()[0], np.array([[0, 1, 1], [0, 1, 2]]), 2, "degenerate"),
        (
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]]),
            np.array([[0, 1, 2]]),
            1,
            "degenerate",
        ),
        (_tet()[0], _tet()[1], 0, "num_basis"),
        (_tet()[0], _tet()[1], 5, "num_basis"),
        (np.vstack([_tet()[0], [[5.0, 5.0, 5.0]]]), _tet()[1], 2, "no triangle"),
    ],
)
def test_validation(vertices, triangles, num_basis, match):
    with pytest.raises(ValueError, match=match):
        cotangent_laplacian_eigpairs(vertices, triangles, num_basis)


def test_vertex_areas_validation():
    vertices, triangles = _tet()
    with pytest.raises(ValueError, match="indices must lie"):
        mesh_vertex_areas(vertices, triangles + 1)
    with pytest.raises(ValueError, match="degenerate"):
        mesh_vertex_areas(vertices, np.array([[0, 0, 1]]))
