"""Pure-Equinox neural network model zoo.

geonnax holds deterministic neural-network cores — SIREN, MFN, random
features, SNGP / VSSGP basis maps, conditioning / hypernetworks, geo
encoders, Slepian eigenfunctions — as plain ``equinox.Module`` classes
with no NumPyro dependency. Probabilistic libraries layer on top by
wrapping a geonnax core and swapping its named parameters for
sample / param sites.

Submodules:

- `geonnax.geo` — lon/lat helpers (``deg2rad``, ``lonlat_scale``, …).
- `geonnax.basis` — the public basis surface: Fourier / seasonal /
  interaction feature transforms, plus the re-exported eigenfunction
  (Dirichlet, spherical-harmonic, Slepian, graph-Laplacian), localized
  (RBF), and overcomplete (Gabor frame) spatial bases.
- `geonnax._basis` — implementation home for the spatial bases above
  (re-exported through `geonnax.basis`).
- `geonnax.encoders` — coordinate encoders (``Deg2Rad``, …,
  ``SphericalHarmonicEncoder``).
- `geonnax.ncp` — ``NCPContinuousPerturb`` input perturbation.
- `geonnax.siren` — ``SirenDense``, ``SIREN`` (Sitzmann et al.,
  2020).
- `geonnax.slepian` — ``SlepianEncoder``,
  ``HybridSphericalSlepianEncoder``.
- `geonnax.randfeat` — ``OrthogonalRandomFeatures`` +
  ``rff_forward`` / ``rff_cosine_forward`` helpers.
"""

from geonnax import (
    basis,
    conditioning,
    encoders,
    ensemble,
    fno,
    geo,
    heteroscedastic,
    layers,
    mfn,
    mswt,
    ncp,
    randfeat,
    sfno,
    siren,
    slepian,
    sngp,
    unet,
    vssgp,
    wno,
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
from geonnax.fno import FNO, DomainPadding, FNOBlock, FourierNeuralOperator
from geonnax.heteroscedastic import (
    HeteroscedasticHead,
    MCSigmoidDenseFA,
    MCSoftmaxDenseFA,
    hetero_noisy_logits,
)
from geonnax.layers import (
    Attention,
    Block,
    ConvNeXtBlock,
    CPTensor,
    DenseTensor,
    Downsample,
    GlobalResponseNorm,
    LinearAttention,
    ResnetBlock,
    SpectralConv,
    SphericalHarmonicTransform,
    SphericalSpectralConv,
    SphericalWaveletConv,
    SphericalWaveletTransform,
    SqueezeExcitation,
    StandardizedConv,
    TTTensor,
    TuckerTensor,
    Upsample,
    WaveletAttention,
    WaveletConv,
    WindowedAttention,
    init_factorized_tensor,
)
from geonnax.mfn import FourierFilter, FourierNet, GaborFilter, GaborNet, mfn_forward
from geonnax.mswt import MSWT, MSWTBlock, MultiScaleWaveletTransformer
from geonnax.ncp import NCPContinuousPerturb
from geonnax.randfeat import (
    OrthogonalRandomFeatures,
    orthogonal_blocks,
    rff_cosine_forward,
    rff_forward,
)
from geonnax.sfno import SFNO, SphericalFNOBlock
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
from geonnax.unet import NestedResidualUNet, Stage, UNet, XUNet
from geonnax.vssgp import DeepVSSGPCore
from geonnax.wno import WNO, WaveletNeuralOperator, WNOBlock


__version__ = "0.0.4"  # x-release-please-version

__all__ = [
    "FNO",
    "MSWT",
    "SFNO",
    "SIREN",
    "WNO",
    "AbstractConditioner",
    "AffineModulation",
    "Attention",
    "Block",
    "CPTensor",
    "Cartesian3DEncoder",
    "ConcatConditioner",
    "ConditionedINR",
    "ConvNeXtBlock",
    "CyclicEncoder",
    "DeepVSSGPCore",
    "Deg2Rad",
    "DenseRank1",
    "DenseTensor",
    "DomainPadding",
    "Downsample",
    "FNOBlock",
    "FiLM",
    "FourierFilter",
    "FourierNet",
    "FourierNeuralOperator",
    "GaborFilter",
    "GaborNet",
    "GeneratedSiren",
    "GlobalResponseNorm",
    "HeteroscedasticHead",
    "HybridSphericalSlepianEncoder",
    "HyperLinear",
    "HyperSIREN",
    "LaplaceRandomFeatureCovariance",
    "LayerNormEnsemble",
    "LinearAttention",
    "LonLatScale",
    "MCSigmoidDenseFA",
    "MCSoftmaxDenseFA",
    "MSWTBlock",
    "MultiHeadAttentionBE",
    "MultiScaleWaveletTransformer",
    "NCPContinuousPerturb",
    "NestedResidualUNet",
    "OrthogonalRandomFeatures",
    "RandomFeatureGaussianProcess",
    "Rank1ProjInit",
    "ResnetBlock",
    "SirenDense",
    "SirenLayerSpec",
    "SirenLayerType",
    "SlepianEncoder",
    "SpectralConv",
    "SphericalFNOBlock",
    "SphericalHarmonicEncoder",
    "SphericalHarmonicTransform",
    "SphericalSpectralConv",
    "SphericalWaveletConv",
    "SphericalWaveletTransform",
    "SqueezeExcitation",
    "Stage",
    "StandardizedConv",
    "TTTensor",
    "TuckerTensor",
    "UNet",
    "Upsample",
    "WNOBlock",
    "WaveletAttention",
    "WaveletConv",
    "WaveletNeuralOperator",
    "WindowedAttention",
    "XUNet",
    "__version__",
    "apply_rank1_proj",
    "basis",
    "build_siren_specs",
    "conditioning",
    "encoders",
    "ensemble",
    "fno",
    "geo",
    "hetero_noisy_logits",
    "heteroscedastic",
    "init_factorized_tensor",
    "init_rank1_proj",
    "layers",
    "mfn",
    "mfn_forward",
    "mswt",
    "ncp",
    "orthogonal_blocks",
    "randfeat",
    "rff_cosine_forward",
    "rff_forward",
    "sfno",
    "siren",
    "siren_W_limit",
    "slepian",
    "sngp",
    "unet",
    "vssgp",
    "wno",
]
