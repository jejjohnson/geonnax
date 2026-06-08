"""Pure-JAX eigenfunction bases shared by NN spectral layers and GP inducing features.

Both NN-side spectral layers (HSGP-style features) and GP-side inter-domain
inducing features evaluate the same Laplacian eigenfunctions on a bounded box.
Owning that math here removes the duplication and keeps the basis pandas-free.

Public surface (kernel-free pieces only — kernel-dependent helpers such as
``spectral_density`` and the RFF cosine path draws live with the consuming
library, since they ``isinstance``-dispatch on kernel classes):

- `fourier_basis_1d` / `fourier_eigenvalues_1d` — 1D Dirichlet
  eigenpairs of $-d^2/dx^2$ on $[-L, L]$.
- `fourier_basis` — tensor-product extension to $[-L, L]^D$.
- `divfree_basis` — divergence-free vector atoms on $[-L, L]^2$
  (skew gradients of the box-Dirichlet stream functions).
- `real_spherical_harmonics` — real SHs on the unit 2-sphere.
- `graph_laplacian_eigpairs` — smallest eigenpairs of a graph Laplacian.
- `SlepianCapBasis` / `slepian_cap_basis` — Slepian eigenfunctions
  on a spherical cap.
- `wavelet_basis_1d` / `wavelet_basis_2d` — orthonormal DWT basis
  matrices (Haar / Daubechies, periodic, power-of-two).

Localized / overcomplete bases carry no eigenvalues; they expose the per-atom
geometry instead (the other half of the basis contract):

- `rbf_basis` / `spherical_rbf_basis` / `wendland_c2` / `wendland_c4` —
  placeable radial bumps with Gaussian or compact (Wendland) support, in
  Euclidean or geodesic (on-sphere) distance.
- `gabor_frame` / `gabor_frame_grid` — fixed overcomplete
  multiscale Gabor frame.

A data-driven basis stands apart, returning its own spectrum:

- `eof_basis` — empirical orthogonal functions (PCA) of a data matrix.
"""

from geonnax._basis._divfree import divfree_basis
from geonnax._basis._eof import eof_basis
from geonnax._basis._fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)
from geonnax._basis._gabor import gabor_frame, gabor_frame_grid
from geonnax._basis._laplacian import graph_laplacian_eigpairs
from geonnax._basis._rbf import (
    rbf_basis,
    spherical_rbf_basis,
    wendland_c2,
    wendland_c4,
)
from geonnax._basis._slepian import (
    SlepianCapBasis,
    shannon_number,
    slepian_cap_basis,
    slepian_cap_eigh_per_m,
    slepian_concentration_matrix,
)
from geonnax._basis._spherical import harmonic_degrees, real_spherical_harmonics
from geonnax._basis._wavelet import wavelet_basis_1d, wavelet_basis_2d


__all__ = [
    "SlepianCapBasis",
    "divfree_basis",
    "eof_basis",
    "fourier_basis",
    "fourier_basis_1d",
    "fourier_eigenvalues",
    "fourier_eigenvalues_1d",
    "gabor_frame",
    "gabor_frame_grid",
    "graph_laplacian_eigpairs",
    "harmonic_degrees",
    "rbf_basis",
    "real_spherical_harmonics",
    "shannon_number",
    "slepian_cap_basis",
    "slepian_cap_eigh_per_m",
    "slepian_concentration_matrix",
    "spherical_rbf_basis",
    "wavelet_basis_1d",
    "wavelet_basis_2d",
    "wendland_c2",
    "wendland_c4",
]
