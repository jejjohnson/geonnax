r"""Triangle meshes and their cotangent Laplace–Beltrami eigenbasis.

- `icosphere` — the subdivided icosahedron: a quasi-uniform triangulation of
  the sphere at any resolution, for mesh Laplacians and finite-element bases.
- `mesh_vertex_areas` — lumped (barycentric) vertex areas, the mass matrix and
  quadrature weights of a triangle mesh.
- `cotangent_laplacian_eigpairs` — the smallest eigenpairs of the cotangent
  Laplace–Beltrami operator, $L\varphi = \lambda M \varphi$, on any triangle
  mesh (closed, or with a boundary under natural Neumann conditions).

Unlike the combinatorial `graph_laplacian_eigpairs`, the cotangent Laplacian
with a mass matrix is a consistent discretisation of the Laplace–Beltrami
operator, so its eigenpairs converge to the surface's eigenfunctions under mesh
refinement (on the unit sphere $\lambda \to \ell(\ell + 1)$ with multiplicity
$2\ell + 1$).

References:
    Pinkall & Polthier (1993), *Computing discrete minimal surfaces and their
    conjugates*, Experimental Mathematics 2(1), 15–36.
    Meyer, Desbrun, Schröder & Barr (2003), *Discrete differential-geometry
    operators for triangulated 2-manifolds*, Visualization and Mathematics
    III, 35–57.
    Chen, Liu, Meng, Chen, Liu & Li (2024), *Learning neural operators on
    Riemannian manifolds* (NORM), National Science Open; arXiv:2302.08166.
"""

from __future__ import annotations

import math

import einx
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


def _validate_mesh(vertices, triangles) -> tuple[np.ndarray, np.ndarray]:
    """Host copies of a mesh, checked for shape, index range and finiteness."""
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3); got {verts.shape}.")
    if not np.all(np.isfinite(verts)):
        raise ValueError("vertices must be finite.")
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.shape[0] == 0:
        raise ValueError(f"triangles must have shape (T, 3), T >= 1; got {tris.shape}.")
    if not np.issubdtype(tris.dtype, np.integer):
        raise ValueError(f"triangles must hold integer indices; got {tris.dtype}.")
    n_vertices = verts.shape[0]
    if tris.min() < 0 or tris.max() >= n_vertices:
        raise ValueError(
            f"triangle indices must lie in [0, {n_vertices - 1}]; "
            f"got [{tris.min()}, {tris.max()}]."
        )
    return verts, tris.astype(np.int64)


def _corner_geometry(
    verts: np.ndarray, tris: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle doubled areas ``(T,)`` and corner cotangents ``(T, 3)``.

    ``cot[:, k]`` is the cotangent of the angle at corner ``k``, the angle
    opposite the edge ``(k + 1, k + 2) mod 3``. Raises on zero-area triangles.
    """
    p = verts[tris]  # (T, 3 corners, 3 coords)
    e1 = np.roll(p, -1, axis=1) - p  # corner k -> k + 1
    e2 = np.roll(p, -2, axis=1) - p  # corner k -> k + 2
    double_area = np.linalg.norm(np.cross(e1, e2), axis=-1)  # (T, 3), equal per row
    scale = einx.multiply(
        "t k, t k -> t k",
        np.linalg.norm(e1, axis=-1),
        np.linalg.norm(e2, axis=-1),
    )
    degenerate = np.any(double_area <= 1e-12 * scale, axis=1)
    if np.any(degenerate):
        bad = np.flatnonzero(degenerate)
        raise ValueError(
            f"{bad.size} degenerate (zero-area) triangle(s), e.g. index {bad[0]}."
        )
    cot = einx.divide(
        "t k, t k -> t k", einx.dot("t k d, t k d -> t k", e1, e2), double_area
    )
    return double_area[:, 0], cot


def _lumped_areas(n_vertices: int, tris: np.ndarray, double_area: np.ndarray):
    areas = np.zeros(n_vertices)
    third = einx.id("t -> t k", double_area / 6.0, k=3)
    np.add.at(areas, tris.ravel(), third.ravel())
    return areas


def mesh_vertex_areas(
    vertices: Float[Array, "V 3"], triangles: Int[Array, "T 3"]
) -> Float[Array, " V"]:
    r"""Lumped (barycentric) vertex areas of a triangle mesh.

    Each triangle gives a third of its area to each of its three corners:

    $$
    a_v = \sum_{t \ni v} \frac{|t|}{3}, \qquad
    |t| = \tfrac12 \lVert (x_q - x_p) \times (x_r - x_p) \rVert ,
    $$

    so $\sum_v a_v$ is the total surface area and $M = \operatorname{diag}(a)$
    is the lumped (diagonal) finite-element mass matrix. The areas are also
    quadrature weights: $\int_S g \, dA \approx \sum_v a_v\, g(x_v)$. This is
    the barycentric lumping; Meyer et al. (2003) also describe a mixed Voronoi
    variant, which differs at obtuse triangles but has the same convergence.

    Computed eagerly on the host with NumPy in float64 (the mesh is static
    geometry), returned as a JAX array; not traceable under ``jit``.

    Pseudocode:

        a = zeros(V)
        for (p, q, r) in triangles:
            a[[p, q, r]] += |(x_q - x_p) × (x_r - x_p)| / 6

    Args:
        vertices: Vertex positions of shape ``(V, 3)``.
        triangles: Integer vertex indices of shape ``(T, 3)``; orientation does
            not matter.

    Returns:
        Vertex areas of shape ``(V,)``. Vertices that no triangle references
        get area zero.

    Raises:
        ValueError: If the shapes are wrong, an index is out of range, a
            vertex is not finite, or a triangle has zero area.

    References:
        Meyer, Desbrun, Schröder & Barr (2003), *Discrete
        differential-geometry operators for triangulated 2-manifolds*.

    Examples:
        >>> import math
        >>> from geonnax.basis import icosphere, mesh_vertex_areas
        >>> vertices, triangles = icosphere(3)
        >>> areas = mesh_vertex_areas(vertices, triangles)
        >>> areas.shape
        (642,)
        >>> round(float(areas.sum()) / (4 * math.pi), 2)  # ≈ the sphere's 4π
        1.0
    """
    verts, tris = _validate_mesh(vertices, triangles)
    double_area, _ = _corner_geometry(verts, tris)
    return jnp.asarray(_lumped_areas(verts.shape[0], tris, double_area))


def cotangent_laplacian_eigpairs(
    vertices: Float[Array, "V 3"],
    triangles: Int[Array, "T 3"],
    num_basis: int,
) -> tuple[Float[Array, " M"], Float[Array, "V M"], Float[Array, " V"]]:
    r"""Smallest eigenpairs of the cotangent Laplace–Beltrami operator of a mesh.

    The cotangent Laplacian (Pinkall & Polthier, 1993) weights each edge
    $(i, j)$ by the cotangents of the two angles $\alpha_{ij}, \beta_{ij}$
    opposite it (one per adjacent triangle; a boundary edge has one):

    $$
    L_{ij} = -\tfrac12 (\cot\alpha_{ij} + \cot\beta_{ij}) \quad (i \ne j
    \text{ an edge}), \qquad L_{ii} = -\sum_{j \ne i} L_{ij},
    $$

    a symmetric positive semidefinite stiffness matrix with the constants in
    its null space. With the lumped mass $M = \operatorname{diag}(a)$ from
    `mesh_vertex_areas`, the discrete Laplace–Beltrami eigenproblem is the
    generalised symmetric problem

    $$
    L \varphi = \lambda M \varphi
    \iff (M^{-1/2} L M^{-1/2})\, \psi = \lambda \psi, \quad
    \varphi = M^{-1/2} \psi ,
    $$

    so the eigenvectors are $M$-orthonormal, $\Phi^\top M \Phi = I$: they are
    $L^2(S)$-orthonormal under the vertex-area quadrature. On a closed mesh
    $\lambda_0 = 0$ with $\varphi_0 = 1 / \sqrt{\sum_v a_v}$; on the unit
    sphere $\lambda \to \ell(\ell + 1)$ with multiplicity $2\ell + 1$ under
    refinement. On a mesh with a boundary the cotangent Laplacian imposes
    natural (Neumann, free-boundary) conditions. Obtuse triangles give negative
    cotangents, which is still the correct finite-element stiffness. This is
    the eigenbasis behind NORM (Chen et al., 2024).

    The construction is eager: the matrix is assembled and diagonalised on the
    host with NumPy in float64, and the results are returned as JAX arrays
    (not traceable under ``jit``). It is a dense eigendecomposition,
    $O(V^3)$ time and $O(V^2)$ memory, practical up to a few thousand
    vertices; the sparse / Lanczos path is ``kernellib.laplacian_eigpairs``.

    Pseudocode:

        W = zeros(V, V); a = mesh_vertex_areas(vertices, triangles)
        for (p, q, r) in triangles, for each corner p (edge qr opposite):
            c = (x_q - x_p)·(x_r - x_p) / |(x_q - x_p) × (x_r - x_p)|
            W[q, r] += c / 2; W[r, q] += c / 2
        L = diag(W.sum(1)) - W
        lam, U = eigh(diag(a^-1/2) L diag(a^-1/2))
        return max(lam[:M], 0), diag(a^-1/2) U[:, :M], a

    Args:
        vertices: Vertex positions of shape ``(V, 3)``.
        triangles: Integer vertex indices of shape ``(T, 3)``; orientation does
            not matter. Every vertex must belong to a triangle.
        num_basis: Number $M$ of low-frequency eigenpairs, ``1 <= M <= V``.

    Returns:
        ``(eigvals, eigvecs, areas)``: eigenvalues of shape ``(M,)`` sorted
        ascending (tiny negatives from roundoff clipped to 0), $M$-orthonormal
        eigenvectors of shape ``(V, M)`` (column ``m`` is $\varphi_m$ at the
        vertices), and the vertex areas of shape ``(V,)`` — the mass matrix
        diagonal and quadrature weights. The analysis of a vertex field $u$
        is ``eigvecs.T @ (areas * u)``; the synthesis is ``eigvecs @ coeffs``.

    Raises:
        ValueError: If the shapes are wrong, an index is out of range, a
            vertex is not finite, a triangle has zero area, a vertex is in no
            triangle, or ``num_basis`` is outside ``[1, V]``.

    References:
        Pinkall & Polthier (1993), *Computing discrete minimal surfaces and
        their conjugates*, Experimental Mathematics 2(1), 15–36.
        Meyer, Desbrun, Schröder & Barr (2003), *Discrete
        differential-geometry operators for triangulated 2-manifolds*.
        Chen et al. (2024), *Learning neural operators on Riemannian
        manifolds*, arXiv:2302.08166.

    Examples:
        >>> from geonnax.basis import cotangent_laplacian_eigpairs, icosphere
        >>> vertices, triangles = icosphere(3)
        >>> eigvals, eigvecs, areas = cotangent_laplacian_eigpairs(
        ...     vertices, triangles, num_basis=9
        ... )
        >>> eigvals.shape, eigvecs.shape, areas.shape  # (M,), (V, M), (V,)
        ((9,), (642, 9), (642,))
        >>> [round(float(v), 1) for v in eigvals]  # ≈ ℓ(ℓ+1): 0, 2×3, 6×5
        [0.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0]
    """
    verts, tris = _validate_mesh(vertices, triangles)
    n_vertices = verts.shape[0]
    if num_basis < 1 or num_basis > n_vertices:
        raise ValueError(f"num_basis must be in [1, {n_vertices}]; got {num_basis}.")
    double_area, cot = _corner_geometry(verts, tris)
    areas = _lumped_areas(n_vertices, tris, double_area)
    if np.any(areas <= 0):
        raise ValueError(
            f"{int(np.sum(areas <= 0))} vertex(es) belong to no triangle; "
            "remove unreferenced vertices first."
        )

    # Corner k's cotangent weights the opposite edge (k + 1, k + 2).
    q = np.roll(tris, -1, axis=1).ravel()
    r = np.roll(tris, -2, axis=1).ravel()
    half_cot = 0.5 * cot.ravel()
    weights = np.zeros((n_vertices, n_vertices))
    np.add.at(weights, (q, r), half_cot)
    np.add.at(weights, (r, q), half_cot)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    laplacian = 0.5 * (laplacian + einx.id("i j -> j i", laplacian))

    m_inv_sqrt = 1.0 / np.sqrt(areas)
    sym = einx.multiply("i, i j, j -> i j", m_inv_sqrt, laplacian, m_inv_sqrt)
    sym = 0.5 * (sym + einx.id("i j -> j i", sym))
    eigvals, psi = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals[:num_basis], 0.0)
    eigvecs = einx.multiply("v, v m -> v m", m_inv_sqrt, psi[:, :num_basis])
    return jnp.asarray(eigvals), jnp.asarray(eigvecs), jnp.asarray(areas)
