"""Tests for the icosphere mesh (GN3)."""

from __future__ import annotations

import math
from collections import Counter
from itertools import pairwise

import jax
import numpy as np
import pytest

from geonnax.basis import icosphere


def _edges(triangles: np.ndarray) -> Counter:
    return Counter(
        tuple(sorted((int(a), int(b))))
        for tri in triangles
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    )


@pytest.mark.parametrize("level", range(6))
def test_counts_and_closed_manifold(level):
    vertices, triangles = map(np.asarray, icosphere(level))
    n_vertices, n_triangles = 10 * 4**level + 2, 20 * 4**level
    assert vertices.shape == (n_vertices, 3)
    assert triangles.shape == (n_triangles, 3)
    edges = _edges(triangles)
    assert set(edges.values()) == {2}  # every edge borders exactly two triangles
    assert len(edges) == 30 * 4**level
    assert n_vertices - len(edges) + n_triangles == 2  # Euler characteristic


def test_radius_and_outward_orientation():
    with jax.enable_x64(True):
        vertices, triangles = map(np.asarray, icosphere(3, radius=2.5))
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 2.5, atol=1e-12)
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    assert np.all(np.sum(normals * (a + b + c), axis=1) > 0)


def test_area_converges_to_sphere():
    areas = []
    with jax.enable_x64(True):
        for level in range(6):
            vertices, triangles = map(np.asarray, icosphere(level))
            a, b, c = (vertices[triangles[:, k]] for k in range(3))
            areas.append(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    assert all(x < y for x, y in pairwise(areas))
    assert areas[-1] < 4.0 * math.pi
    assert 4.0 * math.pi - areas[-1] < 1e-2


def test_negative_level_raises():
    with pytest.raises(ValueError, match="level"):
        icosphere(-1)
