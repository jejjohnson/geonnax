"""Pure-Equinox neural network model zoo.

Geographic / spherical helpers
    :mod:`geonnax.geo` — :func:`deg2rad`, :func:`lonlat_scale`,
    :func:`lonlat_to_cartesian3d`, :func:`cyclic_encode`,
    :func:`spherical_harmonic_encode`.

Basis / temporal transforms
    :mod:`geonnax.basis` — :func:`fourier_features`,
    :func:`seasonal_features`, :func:`seasonal_frequencies`,
    :func:`interaction_features`, :func:`standardize`,
    :func:`unstandardize`.

Eigenfunction bases (kernel-free)
    :mod:`geonnax._basis` — Dirichlet / spherical-harmonic / Slepian /
    graph-Laplacian eigenpairs shared by NN spectral layers and
    GP-side inducing features.
"""

from geonnax import basis, geo


__version__ = "0.1.0"

__all__ = [
    "__version__",
    "basis",
    "geo",
]
