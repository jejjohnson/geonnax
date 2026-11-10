"""Feature engineering helpers, and the old home of the bases now in functax.

geonnax's bases moved to functax (jejjohnson/geonnax#153): the eigenfunction,
localized and frame bases to the family submodules of `functax.basis`, the
sphere grids and meshes to `functax.geometry`, and ``gauss_legendre`` to
`functax.quad`. Their names still resolve here for one release, with a
`DeprecationWarning` naming the functax path; see the ``_MOVED`` table.

The feature-engineering helpers stay in geonnax:

* `interaction_features` — element-wise products on selected pairs of input
  columns.
* `standardize` / `unstandardize` — affine transform with a precomputed mean
  and std.
"""

from __future__ import annotations

import einx
import functax.basis.eigen as _eigen
import functax.basis.features as _features
import functax.basis.fourier as _fourier
import functax.basis.local as _local
import functax.basis.polynomial as _polynomial
import functax.basis.slepian as _slepian
import functax.basis.sphere as _sphere
import functax.basis.vector as _vector
import functax.geometry as _geometry
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from geonnax import _compat
from geonnax._deprecation import Moved, deprecated_dir, deprecated_getattr


def interaction_features(
    x: Float[Array, "N D"],
    pairs: Int[Array, "K 2"],
) -> Float[Array, "N K"]:
    r"""Element-wise products on selected pairs of input columns.

    For each pair $(i, j)$ and each row $n$, computes
    $x_{n, i} \cdot x_{n, j}$.

    Args:
        x: Input matrix, shape ``(N, D)``.
        pairs: Index pairs, shape ``(K, 2)``. Empty pairs yield an
            ``(N, 0)`` output.

    Returns:
        Array of shape ``(N, K)`` of pairwise products.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import interaction_features
        >>> x = jnp.arange(6.0).reshape(2, 3)  # (N=2, D=3)
        >>> pairs = jnp.array([[0, 1], [0, 2]])  # (K=2, 2)
        >>> interaction_features(x, pairs).shape
        (2, 2)
    """
    if pairs.shape[0] == 0:
        return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    # x[:, pairs] has shape (N, K, 2); reduce the paired axis with prod.
    selected = x[:, pairs]
    return einx.prod("n k [two]", selected)


def standardize(
    x: Float[Array, "*shape"],
    mu: Float[Array, "*shape"],
    std: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    """Affine standardize: ``(x - mu) / std``.

    Broadcasts ``mu`` and ``std`` against ``x`` per the JAX broadcasting
    rules. ``std`` is *not* clamped; pass a positive value or guard
    upstream.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import standardize
        >>> x = jnp.array([1.0, 3.0, 5.0])
        >>> bool(jnp.allclose(standardize(x, 3.0, 2.0), jnp.array([-1, 0, 1])))
        True
    """
    return (x - mu) / std


def unstandardize(
    z: Float[Array, "*shape"],
    mu: Float[Array, "*shape"],
    std: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    """Inverse of `standardize`: ``z * std + mu``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.basis import standardize, unstandardize
        >>> x = jnp.array([1.0, 3.0, 5.0])
        >>> # unstandardize undoes standardize for the same (mu, std).
        >>> z = standardize(x, 3.0, 2.0)
        >>> bool(jnp.allclose(unstandardize(z, 3.0, 2.0), x))
        True
    """
    return z * std + mu


#: names that moved to functax, served with a DeprecationWarning until removal
_MOVED: dict[str, Moved] = {
    "SlepianBasis": Moved("functax.basis.slepian.SlepianBasis", _slepian.SlepianBasis),
    "SlepianCapBasis": Moved(
        "functax.basis.slepian.SlepianCapBasis", _slepian.SlepianCapBasis
    ),
    "associated_legendre": Moved(
        "functax.basis.polynomial.associated_legendre", _polynomial.associated_legendre
    ),
    "associated_legendre_indices": Moved(
        "functax.basis.polynomial.associated_legendre_indices",
        _polynomial.associated_legendre_indices,
    ),
    "cotangent_laplacian_eigpairs": Moved(
        "functax.basis.eigen.cotangent_laplacian_eigpairs",
        _eigen.cotangent_laplacian_eigpairs,
    ),
    "curlfree_basis": Moved(
        "functax.basis.vector.curlfree_basis", _vector.curlfree_basis
    ),
    "divfree_basis": Moved("functax.basis.vector.divfree_basis", _vector.divfree_basis),
    "eof_basis": Moved("functax.basis.eigen.eof_basis", _eigen.eof_basis),
    "fibonacci_sphere": Moved(
        "functax.geometry.fibonacci_sphere",
        _compat.fibonacci_sphere,
        "Note that it takes unit= in place of output_unit=, with no default.",
    ),
    "fourier_basis": Moved(
        "functax.basis.fourier.fourier_basis", _fourier.fourier_basis
    ),
    "fourier_basis_1d": Moved(
        "functax.basis.fourier.fourier_basis_1d", _fourier.fourier_basis_1d
    ),
    "fourier_eigenvalues": Moved(
        "functax.basis.fourier.fourier_eigenvalues", _fourier.fourier_eigenvalues
    ),
    "fourier_eigenvalues_1d": Moved(
        "functax.basis.fourier.fourier_eigenvalues_1d", _fourier.fourier_eigenvalues_1d
    ),
    "fourier_features": Moved(
        "functax.basis.features.fourier_features", _features.fourier_features
    ),
    "gabor_frame": Moved("functax.basis.local.gabor_frame", _local.gabor_frame),
    "gabor_frame_grid": Moved(
        "functax.basis.local.gabor_frame_grid", _local.gabor_frame_grid
    ),
    "gauss_legendre": Moved(
        "functax.quad.gauss_legendre",
        _compat.gauss_legendre,
        "Note that it returns a Rule, which unpacks to (nodes, weights).",
    ),
    "gauss_legendre_grid": Moved(
        "functax.geometry.gauss_legendre_grid",
        _compat.gauss_legendre_grid,
        "Note that it takes unit= in place of output_unit=, with no default.",
    ),
    "gaussian_window_features": Moved(
        "functax.basis.features.gaussian_window_features",
        _features.gaussian_window_features,
    ),
    "graph_laplacian_eigpairs": Moved(
        "functax.basis.eigen.graph_laplacian_eigpairs", _eigen.graph_laplacian_eigpairs
    ),
    "harmonic_degrees": Moved(
        "functax.basis.sphere.harmonic_degrees", _sphere.harmonic_degrees
    ),
    "healpix_grid": Moved(
        "functax.geometry.healpix_grid",
        _compat.healpix_grid,
        "Note that it takes unit= in place of output_unit=, with no default.",
    ),
    "icosphere": Moved("functax.geometry.icosphere", _geometry.icosphere),
    "latlon_grid": Moved(
        "functax.geometry.latlon_grid",
        _compat.latlon_grid,
        "Note that it takes unit= in place of output_unit=, with no default.",
    ),
    "legendre_polynomials": Moved(
        "functax.basis.polynomial.legendre_polynomials",
        _polynomial.legendre_polynomials,
    ),
    "mesh_vertex_areas": Moved(
        "functax.geometry.mesh_vertex_areas", _geometry.mesh_vertex_areas
    ),
    "needlet_basis": Moved("functax.basis.sphere.needlet_basis", _sphere.needlet_basis),
    "needlet_window": Moved(
        "functax.basis.sphere.needlet_window", _sphere.needlet_window
    ),
    "rbf_basis": Moved("functax.basis.local.rbf_basis", _local.rbf_basis),
    "real_spherical_harmonics": Moved(
        "functax.basis.sphere.real_spherical_harmonics",
        _sphere.real_spherical_harmonics,
    ),
    "seasonal_features": Moved(
        "functax.basis.features.seasonal_features", _features.seasonal_features
    ),
    "seasonal_frequencies": Moved(
        "functax.basis.features.seasonal_frequencies", _features.seasonal_frequencies
    ),
    "shannon_number": Moved(
        "functax.basis.slepian.shannon_number", _slepian.shannon_number
    ),
    "shell_basis": Moved("functax.basis.sphere.shell_basis", _sphere.shell_basis),
    "slepian_cap_basis": Moved(
        "functax.basis.slepian.slepian_cap_basis", _slepian.slepian_cap_basis
    ),
    "slepian_cap_eigh_per_m": Moved(
        "functax.basis.slepian.slepian_cap_eigh_per_m", _slepian.slepian_cap_eigh_per_m
    ),
    "slepian_concentration_matrix": Moved(
        "functax.basis.slepian.slepian_concentration_matrix",
        _slepian.slepian_concentration_matrix,
    ),
    "slepian_polar_gap_basis": Moved(
        "functax.basis.slepian.slepian_polar_gap_basis",
        _slepian.slepian_polar_gap_basis,
    ),
    "slepian_region_basis": Moved(
        "functax.basis.slepian.slepian_region_basis", _slepian.slepian_region_basis
    ),
    "spherical_polygon_mask": Moved(
        "functax.basis.slepian.spherical_polygon_mask",
        _compat.spherical_polygon_mask,
        "Note that it takes unit= in place of input_unit=, with no default.",
    ),
    "spherical_rbf_basis": Moved(
        "functax.basis.local.spherical_rbf_basis", _local.spherical_rbf_basis
    ),
    "vector_spherical_harmonics": Moved(
        "functax.basis.sphere.vector_spherical_harmonics",
        _sphere.vector_spherical_harmonics,
    ),
    "wavelet_basis_1d": Moved(
        "functax.basis.local.wavelet_basis_1d", _local.wavelet_basis_1d
    ),
    "wavelet_basis_2d": Moved(
        "functax.basis.local.wavelet_basis_2d", _local.wavelet_basis_2d
    ),
    "wendland_c2": Moved("functax.basis.local.wendland_c2", _local.wendland_c2),
    "wendland_c4": Moved("functax.basis.local.wendland_c4", _local.wendland_c4),
}

__all__ = ["interaction_features", "standardize", "unstandardize"]

__getattr__ = deprecated_getattr(__name__, _MOVED)
__dir__ = deprecated_dir(__all__, _MOVED)
