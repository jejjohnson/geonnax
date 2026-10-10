---
name: build-models-with-geonnax
description: Build geoscientific neural networks in JAX / Equinox on geonnax — Fourier, spherical and wavelet neural operators (FNO, SFNO, WNO, MSWT), dimension-flexible U-Nets, SIREN and multiplicative filter networks with FiLM / hypernetwork conditioning, lon/lat and spherical-harmonic encoders, spectral and localized bases with their eigenvalues or geometry, random Fourier features, spectral normalisation, SNGP / VSSGP heads, BatchEnsemble and heteroscedastic layers. Use whenever a task builds or trains a neural network on gridded, spherical or coordinate data, writes an FFT / spherical-harmonic / wavelet layer, encodes lon/lat or time, evaluates a basis, or needs a deterministic core for an uncertainty-aware model, in a project that uses (or could use) geonnax.
---

# Geoscience models on geonnax

geonnax is a zoo of deterministic neural-network building blocks for
geoscience in JAX: every model is a plain `equinox.Module` built by an
`init(..., *, key)` classmethod, acting on **one example** (batch with
`jax.vmap`), and every basis is a pure function of arrays. Probabilistic
libraries (pyrox) wrap its cores and kernel libraries (kernellib) build on
its random features and bases. Before writing an FFT convolution, a
spherical harmonic, a SIREN init, random features or a lon/lat encoder, look
it up:

1. **The capability index** lists every public name with a one-line
   summary: <https://jejjohnson.github.io/geonnax/api/capabilities/>. Or list
   the installed version:

   ```python
   import importlib
   import inspect

   for name in ("geonnax", "geonnax.basis", "geonnax.geo", "geonnax.layers"):
       module = importlib.import_module(name)
       for attr in getattr(module, "__all__", []):
           obj = getattr(module, attr)
           doc = (inspect.getdoc(obj) or "").split("\n")[0]
           print(f"{name}.{attr}: {doc}")
   ```

2. Compose what exists. geonnax is not on PyPI:
   `uv add "geonnax @ git+https://github.com/jejjohnson/geonnax.git"` (pin a
   release tag, `@vX.Y.Z`, in a library).

## What lives where

| You need… | Use |
|---|---|
| A neural operator on a grid | `geonnax.FNO` (dense / CP / Tucker / TT spectral weights, `domain_padding` for non-periodic fields), `geonnax.WNO` (wavelets), `geonnax.MSWT` (wavelet-domain attention) |
| An operator on the sphere (lat/lon grid) | `geonnax.SFNO`, `geonnax.SphericalHarmonicTransform`, `geonnax.SphericalSpectralConv`, `geonnax.SphericalWaveletTransform` |
| A U-Net (1-D / 2-D / 3-D) | `geonnax.UNet` (ResNet or ConvNeXt blocks, attention, nested U²-Net stages, MC dropout) |
| Layers to compose your own model | `geonnax.layers` (`StandardizedConv`, `ResnetBlock`, `ConvNeXtBlock`, `Attention`, `WindowedAttention`, `SpectralConv`, `WaveletConv`, `Downsample` / `Upsample`, `dwt` / `idwt`, …) |
| An implicit neural representation | `geonnax.SIREN`, `geonnax.SirenDense`, `geonnax.MultiScaleSIREN`, `geonnax.FourierNet`, `geonnax.GaborNet` |
| Conditioning on a latent or context vector | `geonnax.AffineModulation` (`geonnax.FiLM`), `geonnax.ConcatConditioner`, `geonnax.HyperLinear`, `geonnax.ConditionedINR`, `geonnax.HyperSIREN` |
| Lon/lat and time encodings | `geonnax.geo` (`deg2rad`, `lonlat_scale`, `lonlat_to_cartesian3d`, `cyclic_encode`, `spherical_harmonic_encode`); modules `geonnax.SphericalHarmonicEncoder`, `geonnax.Cartesian3DEncoder`, `geonnax.CyclicEncoder`, `geonnax.GeoContextEncoder`, `geonnax.SlepianEncoder` |
| A basis with its eigenvalues or geometry | `geonnax.basis` (`fourier_basis`, `real_spherical_harmonics`, `slepian_cap_basis`, `graph_laplacian_eigpairs`, `divfree_basis`, `rbf_basis`, `spherical_rbf_basis`, `gabor_frame_grid`, `wavelet_basis_1d`, `eof_basis`) |
| Fourier / seasonal / interaction features | `geonnax.basis.fourier_features`, `geonnax.basis.seasonal_features`, `geonnax.basis.interaction_features`, `geonnax.basis.gaussian_window_features` |
| Random Fourier features | `geonnax.rff_forward`, `geonnax.rff_cosine_forward`, `geonnax.orthogonal_blocks`, `geonnax.OrthogonalRandomFeatures` |
| Distance-aware uncertainty | `geonnax.SpectralNormalization`, `geonnax.RandomFeatureGaussianProcess` (SNGP head), `geonnax.DeepVSSGPCore` |
| Ensembles and probabilistic heads | `geonnax.DenseRank1`, `geonnax.LayerNormEnsemble`, `geonnax.MultiHeadAttentionBE`, `geonnax.MCSoftmaxDenseFA`, `geonnax.MixtureOfGaussiansDenseHead`, `geonnax.LinearChainCRF`, `geonnax.NCPContinuousPerturb` |

Priors, sample sites and Bayesian versions of these layers live in pyrox
(`pyrox_nn` wraps geonnax cores); kernels and kernel methods live in
kernellib.

## The rules your code must keep

- **One example per call.** Dense / INR modules take `(D,)`; layers,
  operators and U-Nets take channel-first `(C, *spatial)`; batch with
  `jax.vmap(model)(xs)`. Bases are the exception: they take the `N`
  evaluation points and return `(N, M)`.
- **Build with `init` and an explicit key.** `geonnax.FNO.init(..., key=k)`;
  split keys (`jr.split`), never reuse one. Parameter-free encoders are
  constructed directly (`geonnax.SphericalHarmonicEncoder(l_max=4)`).
- **Modules are immutable.** Anything that updates returns a new module —
  keep it: `sn, y = sn(x)` for `SpectralNormalization`,
  `gp = gp.update_precision(features)` for the SNGP head. Change parameters
  with `eqx.tree_at` or `eqx.apply_updates`, filtering trainables with
  `eqx.filter(model, eqx.is_inexact_array)`.
- **Explicit randomness at call time.** Monte-Carlo heads take `key=`;
  models with dropout take `key=` per call, or wrap them in
  `eqx.nn.inference_mode` to evaluate deterministically.
- **Units and grids.** Lon/lat arrays are `[lon, lat]`;
  `geonnax.SphericalHarmonicEncoder(input_mode="lonlat")` takes radians
  (`geonnax.geo.deg2rad` first), and the other encoders say their unit (an
  `input_unit` argument or the docstring); `SFNO` /
  `SphericalHarmonicTransform` need `n_lat > l_max` and `n_lon > 2·l_max`;
  an FNO keeps `n_modes` below half of each grid extent.
- **JAX rules.** No Python control flow on traced values; float32 in gives
  float32 out.

## Worked example

```python
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import geonnax
from geonnax.basis import fourier_basis
from geonnax.geo import deg2rad


k_fno, k_x, k_orf, k_lin, k_sn, k_gp = jr.split(jr.key(0), 6)


# 1. A resolution-invariant Fourier Neural Operator on one (C, H, W) field
def field(n):  # a smooth periodic field sampled on an n × n grid, (1, n, n)
    t = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    return einx.add("h, w -> 1 h w", jnp.sin(t), jnp.cos(t))


fno = geonnax.FNO.init(1, 1, (8, 8), key=k_fno, hidden_channels=16, n_layers=2)
y_64 = fno(field(64))  # (1, 64, 64)
y_128 = fno(field(128))  # (1, 128, 128): same parameters, finer grid
y_128_on_64 = jax.image.resize(y_128, y_64.shape, "linear")
resolution_gap = jnp.linalg.norm(y_64 - y_128_on_64) / jnp.linalg.norm(y_64)
batch = jax.vmap(fno)(jnp.stack([field(64)] * 4))  # (4, 1, 64, 64)

# 2. Lon/lat (degrees) → spherical-harmonic features, one point per call
lonlat = jnp.array([[-120.0, 35.0], [0.0, 0.0], [150.0, -60.0]])  # (N, 2)
encoder = geonnax.SphericalHarmonicEncoder(l_max=4, input_mode="lonlat")
sh = jax.vmap(encoder)(deg2rad(lonlat))  # (N, 25), radians in

# 3. A spectral basis returns Φ and its eigenvalues — the half a prior needs
x = einx.id("n -> n 1", jnp.linspace(-2.0, 2.0, 2001))  # (N, 1) in [-L, L]
Phi, lam = fourier_basis(x, num_basis_per_dim=8, L=2.0)  # (N, 8), (8,)
dx = 4.0 / 2000  # grid spacing: Σ Φᵀ Φ dx ≈ ∫ φᵢ φⱼ = δᵢⱼ
basis_gap = jnp.max(jnp.abs(dx * Phi.T @ Phi - jnp.eye(8)))

# 4. Orthogonal random features: φ(x)ᵀφ(y) ≈ exp(-‖x - y‖² / 2ℓ²)
orf = geonnax.OrthogonalRandomFeatures.init(4, 1024, key=k_orf, lengthscale=1.5)
X = jr.normal(k_x, (100, 4))  # (N, D)
F = jax.vmap(orf)(X)  # (N, 2048)
sq_dist = einx.sum("n m [d]", einx.subtract("n d, m d -> n m d", X, X) ** 2)
K = jnp.exp(-0.5 * sq_dist / 1.5**2)  # the RBF kernel, only to check the features
feature_gap = jnp.max(jnp.abs(F @ F.T - K))

# 5. Spectral normalisation: calling returns (updated module, output)
linear = eqx.nn.Linear(32, 32, key=k_lin)
sn = geonnax.SpectralNormalization.init(
    linear, coeff=0.95, n_power_iterations=50, key=k_sn
)
sn, y = sn(jnp.ones(32))  # keep the new module: (u, v) advanced
sigma = jnp.linalg.svd(sn.normalized_layer().weight, compute_uv=False)[0]

# 6. SNGP head: the Laplace variance grows away from the training data
gp = geonnax.RandomFeatureGaussianProcess.init(
    2, 256, 1, key=k_gp, momentum=0.0, ridge=0.1
)  # momentum=0: one full-batch update sets the precision exactly
X_train = 0.3 * jr.normal(k_x, (200, 2))  # clustered around the origin
gp = gp.update_precision(jax.vmap(gp.feature_map)(X_train))  # a new module
_, var_near = gp(jnp.zeros(2), return_cov=True)
_, var_far = gp(jnp.array([4.0, -4.0]), return_cov=True)
```

The same FNO parameters map the field on a 64 × 64 and a 128 × 128 grid to
outputs that agree to about 1 % (relative), the Dirichlet basis is
orthonormal on the grid to within 10⁻⁶, 1024 orthogonal random feature pairs
match the RBF kernel to about 0.06 everywhere, 50 power iterations make the
normalised weight's top singular value exactly the 0.95 asked for, and the
SNGP variance far from the training cluster is about ten times its value at
the centre (≈ 1/ridge, since the far point's features are almost orthogonal
to the training features).

## Don't write it — use geonnax

| Don't write… | Use |
|---|---|
| `jnp.fft.rfftn` → keep low modes → multiply → `irfftn` | `geonnax.layers.SpectralConv`, or the whole `geonnax.FNO` |
| A spherical harmonic transform, Legendre recursions, a Gauss–Legendre grid | `geonnax.SphericalHarmonicTransform`, `geonnax.basis.real_spherical_harmonics`, `geonnax.SFNO` |
| A Haar / Daubechies DWT | `geonnax.layers.dwt` / `idwt`, `geonnax.WaveletConv`, `geonnax.WNO` |
| `sin(ω (W x + b))` layers with hand-tuned uniform inits | `geonnax.SIREN`, `geonnax.SirenDense`, `geonnax.siren_W_limit` |
| FiLM `γ(z) ⊙ h + β(z)` or a hypernetwork | `geonnax.AffineModulation`, `geonnax.HyperLinear`, `geonnax.ConditionedINR` |
| `cos(xW/ℓ)` random features, orthogonal frequency blocks | `geonnax.OrthogonalRandomFeatures`, `geonnax.rff_forward`, `geonnax.orthogonal_blocks` |
| `lon * π / 180`, lon/lat → xyz, `[cos θ, sin θ]` | `geonnax.geo.deg2rad`, `geonnax.geo.lonlat_to_cartesian3d`, `geonnax.geo.cyclic_encode` |
| Laplacian eigenfunctions on a box, RBF / Gabor columns, EOFs | `geonnax.basis.fourier_basis`, `geonnax.basis.rbf_basis`, `geonnax.basis.gabor_frame_grid`, `geonnax.basis.eof_basis` |
| Power iteration for a Lipschitz layer | `geonnax.SpectralNormalization` |
| An SNGP output layer with a Laplace covariance | `geonnax.RandomFeatureGaussianProcess` |
| A U-Net from scratch | `geonnax.UNet` |

## Self-check before you finish

- Every model call takes one example; batches go through `jax.vmap`.
- Every `init` got its own key; no key is used twice.
- Updated modules returned by `SpectralNormalization` / `update_precision`
  are kept, not discarded.
- `eqx.filter_jit` and `eqx.filter_grad` of your loss run and give finite
  gradients.
- A float32 input stays float32.

If geonnax lacks what you need, keep your addition small and shaped like
geonnax (an `eqx.Module` with an `init(..., *, key)` classmethod acting on
one example) and consider proposing it upstream at
<https://github.com/jejjohnson/geonnax/issues>.
