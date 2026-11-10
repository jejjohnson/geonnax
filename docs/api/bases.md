# Bases

The bases geonnax used to hold now live in [functax](https://github.com/jejjohnson/functax), which geonnax depends on (jejjohnson/geonnax#153). geonnax keeps the networks and imports functax for the maths, so there is one copy of each basis in the stack.

## Where each basis went

The names did not change, only their module:

- `functax.basis.fourier`: the box eigenpairs `fourier_basis`, `fourier_basis_1d`, `fourier_eigenvalues` and `fourier_eigenvalues_1d`, with Dirichlet, Neumann or periodic boundaries.
- `functax.basis.vector`: the divergence-free and curl-free box bases `divfree_basis` and `curlfree_basis`.
- `functax.basis.sphere`: `real_spherical_harmonics`, `harmonic_degrees`, `vector_spherical_harmonics`, the needlets (`needlet_basis`, `needlet_window`) and `shell_basis`.
- `functax.basis.slepian`: the cap, polar-gap and region Slepian bases, `SlepianBasis`, `SlepianCapBasis`, `shannon_number` and `spherical_polygon_mask`.
- `functax.basis.local`: `rbf_basis`, `spherical_rbf_basis`, the Wendland kernels, the Gabor frames and the DWT matrices `wavelet_basis_1d` and `wavelet_basis_2d`.
- `functax.basis.eigen`: `eof_basis`, `graph_laplacian_eigpairs` and `cotangent_laplacian_eigpairs`.
- `functax.basis.polynomial`: `legendre_polynomials`, `associated_legendre` and `associated_legendre_indices`.
- `functax.basis.features`: `fourier_features`, `seasonal_features`, `seasonal_frequencies`, `gaussian_window_features`, and the random Fourier feature maps `rff_forward` and `rff_cosine_forward`.
- `functax.geometry`: the sphere grids (`fibonacci_sphere`, `gauss_legendre_grid`, `latlon_grid`, `healpix_grid`), `icosphere` and `mesh_vertex_areas`.
- `functax.quad`: `gauss_legendre`, which returns a `Rule` that unpacks to `(nodes, weights)`.

## Deprecated paths

Every moved name still resolves from `geonnax.basis` until geonnax 0.0.13, and emits a `DeprecationWarning` naming its functax path. The sphere grids and `spherical_polygon_mask` keep geonnax's defaults there (radians for grid output, degrees for polygon vertices). functax takes an explicit `unit=` with no default, so a caller who relied on a default gets an error instead of wrong numbers.

## What stays in geonnax

The feature-engineering helpers stay here: `interaction_features` (products of selected column pairs) and `standardize` / `unstandardize` (an affine map with a precomputed mean and standard deviation).

::: geonnax.basis
