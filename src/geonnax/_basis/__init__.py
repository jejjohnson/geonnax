"""Pure-JAX eigenfunction bases shared by NN spectral layers and GP inducing features.

Both NN-side spectral layers (HSGP-style features) and GP-side inter-domain
inducing features evaluate the same Laplacian eigenfunctions on a bounded box.
Owning that math here removes the duplication and keeps the basis pandas-free.

Public surface (kernel-free pieces only — kernel-dependent helpers such as
``spectral_density`` and the RFF cosine path draws live with the consuming
library, since they ``isinstance``-dispatch on kernel classes):

- :func:`fourier_basis_1d` / :func:`fourier_eigenvalues_1d` — 1D Dirichlet
  eigenpairs of :math:`-d^2/dx^2` on :math:`[-L, L]`.
- :func:`fourier_basis` — tensor-product extension to :math:`[-L, L]^D`.
- :func:`real_spherical_harmonics` — real SHs on the unit 2-sphere.
- :func:`graph_laplacian_eigpairs` — smallest eigenpairs of a graph Laplacian.
- :class:`SlepianCapBasis` / :func:`slepian_cap_basis` — Slepian eigenfunctions
  on a spherical cap.
"""

from geonnax._basis._fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)
from geonnax._basis._laplacian import graph_laplacian_eigpairs
from geonnax._basis._slepian import (
    SlepianCapBasis,
    shannon_number,
    slepian_cap_basis,
    slepian_cap_eigh_per_m,
    slepian_concentration_matrix,
)
from geonnax._basis._spherical import harmonic_degrees, real_spherical_harmonics


__all__ = [
    "SlepianCapBasis",
    "fourier_basis",
    "fourier_basis_1d",
    "fourier_eigenvalues",
    "fourier_eigenvalues_1d",
    "graph_laplacian_eigpairs",
    "harmonic_degrees",
    "real_spherical_harmonics",
    "shannon_number",
    "slepian_cap_basis",
    "slepian_cap_eigh_per_m",
    "slepian_concentration_matrix",
]
