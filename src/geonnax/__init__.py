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

from geonnax import (
    basis,
    conditioning,
    dense,
    dropout,
    encoders,
    ensemble,
    geo,
    heteroscedastic,
    mfn,
    ncp,
    randfeat,
    siren,
    slepian,
    sngp,
    vssgp,
)
from geonnax.conditioning import (
    AbstractConditioner,
    AffineModulation,
    ConcatConditioner,
    ConditionedINR,
    FiLM,
    GeneratedSiren,
    HyperLinear,
    HyperSIREN,
)
from geonnax.dense import LinearCore
from geonnax.dropout import MCDropout
from geonnax.encoders import (
    Cartesian3DEncoder,
    CyclicEncoder,
    Deg2Rad,
    LonLatScale,
    SphericalHarmonicEncoder,
)
from geonnax.ensemble import (
    DenseRank1,
    LayerNormEnsemble,
    MultiHeadAttentionBE,
    Rank1ProjInit,
    apply_rank1_proj,
    init_rank1_proj,
)
from geonnax.heteroscedastic import (
    HeteroscedasticHead,
    MCSigmoidDenseFA,
    MCSoftmaxDenseFA,
    hetero_noisy_logits,
)
from geonnax.mfn import FourierFilter, FourierNet, GaborFilter, GaborNet, mfn_forward
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
from geonnax.sngp import LaplaceRandomFeatureCovariance, RandomFeatureGaussianProcess
from geonnax.vssgp import DeepVSSGPCore


__version__ = "0.1.0"

__all__ = [
    "SIREN",
    "AbstractConditioner",
    "AffineModulation",
    "Cartesian3DEncoder",
    "ConcatConditioner",
    "ConditionedINR",
    "CyclicEncoder",
    "DeepVSSGPCore",
    "Deg2Rad",
    "DenseRank1",
    "FiLM",
    "FourierFilter",
    "FourierNet",
    "GaborFilter",
    "GaborNet",
    "GeneratedSiren",
    "HeteroscedasticHead",
    "HybridSphericalSlepianEncoder",
    "HyperLinear",
    "HyperSIREN",
    "LaplaceRandomFeatureCovariance",
    "LayerNormEnsemble",
    "LinearCore",
    "LonLatScale",
    "MCDropout",
    "MCSigmoidDenseFA",
    "MCSoftmaxDenseFA",
    "MultiHeadAttentionBE",
    "NCPContinuousPerturb",
    "OrthogonalRandomFeatures",
    "RandomFeatureGaussianProcess",
    "Rank1ProjInit",
    "SirenDense",
    "SirenLayerSpec",
    "SirenLayerType",
    "SlepianEncoder",
    "SphericalHarmonicEncoder",
    "__version__",
    "apply_rank1_proj",
    "basis",
    "build_siren_specs",
    "conditioning",
    "dense",
    "dropout",
    "encoders",
    "ensemble",
    "geo",
    "hetero_noisy_logits",
    "heteroscedastic",
    "init_rank1_proj",
    "mfn",
    "mfn_forward",
    "ncp",
    "orthogonal_blocks",
    "randfeat",
    "rff_cosine_forward",
    "rff_forward",
    "siren",
    "siren_W_limit",
    "slepian",
    "sngp",
    "vssgp",
]
