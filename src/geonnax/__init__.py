"""Pure-Equinox neural network model zoo.

geonnax holds deterministic neural-network cores — SIREN, MFN, random
features, SNGP / VSSGP basis maps, conditioning / hypernetworks, geo
encoders, Slepian eigenfunctions — as plain ``equinox.Module`` classes
with no NumPyro dependency. Probabilistic libraries layer on top by
wrapping a geonnax core and swapping its named parameters for
sample / param sites.

Submodules:

- :mod:`geonnax.geo` — lon/lat helpers (``deg2rad``, ``lonlat_scale``, …).
- :mod:`geonnax.basis` — Fourier / seasonal / interaction transforms.
- :mod:`geonnax._basis` — Dirichlet / spherical-harmonic / Slepian /
  graph-Laplacian eigenpairs.
- :mod:`geonnax.encoders` — coordinate encoders (``Deg2Rad``, …,
  ``SphericalHarmonicEncoder``).
- :mod:`geonnax.dropout` — ``MCDropout``.
- :mod:`geonnax.ncp` — ``NCPContinuousPerturb`` input perturbation.
- :mod:`geonnax.siren` — ``SirenDense``, ``SIREN`` (Sitzmann et al.,
  2020).
- :mod:`geonnax.slepian` — ``SlepianEncoder``,
  ``HybridSphericalSlepianEncoder``.
- :mod:`geonnax.randfeat` — ``OrthogonalRandomFeatures`` +
  ``rff_forward`` / ``rff_cosine_forward`` helpers.
"""

from geonnax import basis, dropout, encoders, geo, ncp, randfeat, siren, slepian
from geonnax.dropout import MCDropout
from geonnax.encoders import (
    Cartesian3DEncoder,
    CyclicEncoder,
    Deg2Rad,
    LonLatScale,
    SphericalHarmonicEncoder,
)
from geonnax.ncp import NCPContinuousPerturb
from geonnax.randfeat import (
    OrthogonalRandomFeatures,
    orthogonal_blocks,
    rff_cosine_forward,
    rff_forward,
)
from geonnax.siren import (
    SIREN,
    SirenDense,
    SirenLayerSpec,
    SirenLayerType,
    build_siren_specs,
    siren_W_limit,
)
from geonnax.slepian import HybridSphericalSlepianEncoder, SlepianEncoder


__version__ = "0.1.0"

__all__ = [
    "SIREN",
    "Cartesian3DEncoder",
    "CyclicEncoder",
    "Deg2Rad",
    "HybridSphericalSlepianEncoder",
    "LonLatScale",
    "MCDropout",
    "NCPContinuousPerturb",
    "OrthogonalRandomFeatures",
    "SirenDense",
    "SirenLayerSpec",
    "SirenLayerType",
    "SlepianEncoder",
    "SphericalHarmonicEncoder",
    "__version__",
    "basis",
    "build_siren_specs",
    "dropout",
    "encoders",
    "geo",
    "ncp",
    "orthogonal_blocks",
    "randfeat",
    "rff_cosine_forward",
    "rff_forward",
    "siren",
    "siren_W_limit",
    "slepian",
]
