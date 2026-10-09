r"""Triangle meshes of the sphere.

- `icosphere` — the subdivided icosahedron: a quasi-uniform triangulation of
  the sphere at any resolution, for mesh Laplacians and finite-element bases.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=float,
    )  # fmt: skip
    triangles = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True), triangles


def _subdivide(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four via its (deduplicated) edge midpoints."""
    verts = list(vertices)
    midpoint: dict[tuple[int, int], int] = {}

    def mid(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key not in midpoint:
            point = verts[i] + verts[j]
            verts.append(point / np.linalg.norm(point))
            midpoint[key] = len(verts) - 1
        return midpoint[key]

    out = []
    for a, b, c in triangles:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.asarray(verts), np.asarray(out, dtype=np.int64)


def icosphere(
    level: int, *, radius: float = 1.0
) -> tuple[Float[Array, "V 3"], Int[Array, "T 3"]]:
    r"""Subdivided icosahedron: a quasi-uniform triangle mesh of the sphere.

    Starting from the 12-vertex icosahedron, each level splits every triangle
    into four through its edge midpoints (shared between neighbouring
    triangles) and projects the new vertices onto the sphere. Level $\ell$ has
    $V = 10\cdot4^{\ell} + 2$ vertices, $E = 30\cdot4^{\ell}$ edges and
    $T = 20\cdot4^{\ell}$ triangles. Triangles are ordered counter-clockwise
    seen from outside, so their normals point outward.

    Built on the host with NumPy (the connectivity is static); returned as JAX
    arrays.

    Args:
        level: Number of subdivisions, ``>= 0``.
        radius: Sphere radius.

    Returns:
        ``(vertices, triangles)`` of shapes ``(V, 3)`` and ``(T, 3)``.

    Examples:
        >>> from geonnax.basis import icosphere
        >>> vertices, triangles = icosphere(2)
        >>> vertices.shape, triangles.shape  # V = 10·4² + 2, T = 20·4²
        ((162, 3), (320, 3))
    """
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}.")
    vertices, triangles = _icosahedron()
    for _ in range(level):
        vertices, triangles = _subdivide(vertices, triangles)
    return jnp.asarray(radius * vertices), jnp.asarray(triangles)
