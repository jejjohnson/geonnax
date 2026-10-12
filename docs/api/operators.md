# Operators (FNO / SFNO)

Neural operators learn mappings between discretised functions rather than between
fixed-size vectors. Because the spatial grid is read at call time, an FNO is
**resolution-invariant**: train on one grid and evaluate on another.

A Fourier layer mixes channels in the truncated frequency domain:

$$
\hat{y}[o, k] \;=\; \sum_i \hat{W}[i, o, k]\,\hat{x}[i, k],
\qquad \hat{x} = \mathcal{F}(x),
$$

keeping only the lowest `n_modes` frequencies `k`. The full block adds a local
pointwise skip, $\;\mathrm{act}\!\left(\mathcal{F}^{-1}(\hat{W}\hat{x}) + W x\right)$.

The **Spherical FNO** swaps the planar FFT for a Spherical Harmonic Transform, so the
operator lives on the sphere — the right geometry for global lon/lat fields.

## Fourier Neural Operator

Also exported under the long-form alias `geonnax.FourierNeuralOperator`.

::: geonnax.fno.FNO

::: geonnax.fno.FNOBlock

::: geonnax.fno.DomainPadding

## Spectral convolution

::: geonnax.layers.SpectralConv

## Factorized spectral weights

The spectral weight tensor dominates an FNO's parameter count. Low-rank
factorizations (CP, Tucker, Tensor-Train) shrink it by 10–100× at comparable
accuracy and act as a regulariser — this is the Tensorized FNO (Kossaifi et al., 2023).
Select one through the `factorization=` argument of [`SpectralConv`](#geonnax.layers.SpectralConv)
or [`FNO`](#geonnax.fno.FNO).

::: geonnax.layers.init_factorized_tensor

::: geonnax.layers.DenseTensor

::: geonnax.layers.CPTensor

::: geonnax.layers.TuckerTensor

::: geonnax.layers.TTTensor

## Spherical FNO

::: geonnax.sfno.SFNO

::: geonnax.sfno.SphericalFNOBlock

::: geonnax.layers.SphericalSpectralConv

::: geonnax.layers.SphericalHarmonicTransform

## Manifold neural operator

When the geometry is fixed (an ice-sheet mesh, an ocean basin on the sphere),
the FNO's Fourier modes can be replaced by the domain's own spectral basis.
`ManifoldNO` is NORM (Chen et al., 2024) generalised to any basis: every block
analyses into the $M$ modes of one shared, frozen
[`BasisTransform`](layers.md#geonnax.layers.BasisTransform), mixes channels with
one matrix per mode, and synthesises back at the same $N$ points,

$$
h_\ell = \sigma\big(\Phi\,(W_{\ell,m}\,\hat h_m)_{m=1}^{M} + W_\ell h_{\ell-1}\big),
\qquad \hat h = \Phi^\top \operatorname{diag}(w)\,h_{\ell-1}^\top .
$$

Build the transform from a mesh's cotangent Laplace–Beltrami eigenvectors
(`BasisTransform.from_mesh`, the NORM basis) or from a region Slepian basis
(`BasisTransform.from_slepian`). The weights are tied to that domain's modes, so
the operator does not transfer to another geometry; output at query points other
than the input points is a planned follow-up.

::: geonnax.manifold_no.ManifoldNO

::: geonnax.manifold_no.ManifoldNOBlock

## Convolutional Neural Operator

The CNO (Raonić et al., NeurIPS 2023) is the *local* counterpart of the FNO: a
U-Net of ordinary convolution stencils in which every other operation respects
the band limit, so the discrete network represents a continuous operator. The
sampling rate changes with [`band_limited_resize`](layers.md#geonnax.layers.band_limited_resize)
and the nonlinearity is evaluated on a finer grid by
[`AntiAliasedActivation`](layers.md#geonnax.layers.AntiAliasedActivation),
$\Sigma(v) = R_N\,\sigma(R_{2N} v)$. Four blocks build it: invariant
$I = \Sigma \circ K$, downsampling $D = R_{N/2}\,\sigma\,R_{2N} K$, upsampling
$U = R_{2N}\,\sigma\,R_{2N} K$ (all `CNOBlock`) and residual
$R(v) = v + K \Sigma K v$ (`CNOResBlock`), with R blocks on the skips and at
the bottleneck.

The stencils are tied to the grid spacing, so a CNO is defined on one
computational grid: pass `resolution=` and every input is band-limited-resized
onto it and the output back, which is how the paper evaluates CNO across
resolutions. Local stencils keep sharp and discontinuous features that the
FNO's mode truncation smooths, which also makes the CNO a natural ensemble
partner for it.

Also exported under the long-form alias `geonnax.ConvolutionalNeuralOperator`.

::: geonnax.cno.CNO

::: geonnax.cno.CNOBlock

::: geonnax.cno.CNOResBlock

## Local Neural Operator

The LNO (Li et al., CMAME 2024) is a *usage pattern* for a local operator: a
fully convolutional core with a bounded receptive field, applied
autoregressively as an explicit time-stepper. Each step pads the field by the
halo $p$ with a boundary treatment (`"periodic"`, `"reflect"` or `"zero"`),
applies the core and crops back,

$$u_{t+\Delta t} = \mathrm{crop}_p\big(N(\mathrm{pad}_{p}(u_t))\big),$$

and `rollout` iterates it with `jax.lax.scan`. Because the core is local and
shift-equivariant, one trained LNO runs on domains of any size. The default
core is a shallow [`CNO`](#geonnax.cno.CNO) on its input grid, whose
*effective* receptive radius `cno_receptive_radius` counts the convolution
stencils on its deepest path, $r = h[(5 + 2n)2^L - 3]$ for stencil
half-width $h$, depth $L$ and $n$ residual blocks; the Fourier resampling adds
small global tails, so it is not a hard bound. The receptive range should
match one step's domain of dependence (Ye et al., CMAME 2024): too small
causes oscillation, too large costs accuracy. `receptive_range=` raises when
$r$ exceeds it.

Also exported under the long-form alias `geonnax.LocalNeuralOperator`.

::: geonnax.lno.LNO

::: geonnax.lno.cno_receptive_radius

## Wavelet Neural Operator

The WNO swaps the FNO's Fourier transform for a multi-level discrete wavelet
transform, mixing channels in the wavelet domain. Where Fourier modes are
global, wavelet subbands are spatially localised across scales — better suited
to sharp or non-stationary fields. Analysis and synthesis are exact adjoints,
so an orthonormal wavelet reconstructs perfectly; each spatial extent must be
divisible by `2 ** level`.

Also exported under the long-form alias `geonnax.WaveletNeuralOperator`.

::: geonnax.wno.WNO

::: geonnax.wno.WNOBlock

::: geonnax.layers.WaveletConv

::: geonnax.layers.dwt

::: geonnax.layers.idwt

## Spherical wavelets

A scale-discretised (needlet) wavelet transform on the sphere, built on the
spherical harmonic transform above. Each scale is a smooth harmonic band-pass
window; the squared windows partition unity ($\sum_j g_j(\ell)^2 = 1$), so
analysis followed by synthesis is exact. These axisymmetric needlets give
*localised* multi-scale analysis of global fields — the spherical counterpart
to the planar wavelet transform.

::: geonnax.layers.SphericalWaveletTransform

::: geonnax.layers.SphericalWaveletConv

## Wavelet attention (MSWT)

The Multi-Scale Wavelet Transformer is an *attention-based* operator: it runs
self-attention inside the wavelet domain so the high-frequency detail bands stay
explicit and cannot be smoothed away, mitigating the spectral bias of grid-space
attention on chaotic / long-rollout dynamics (Wang et al., 2026). `MSWT` is a
patch-tokenised U-net of `WaveletAttention` blocks with wavelet down/up-sampling;
`WaveletAttention` is the reusable layer. Attention is global (`Attention`) or,
for large grids, Swin-style `WindowedAttention` (``O(N · window^d)``).

`MSWT` is also exported under the long-form alias
`geonnax.MultiScaleWaveletTransformer`.

::: geonnax.mswt.MSWT

::: geonnax.mswt.MSWTBlock

::: geonnax.layers.WaveletAttention

::: geonnax.layers.WindowedAttention

## Geo-FNO

Geo-FNO (Li et al., JMLR 2023) runs an FNO on per-sample point clouds by
learning a chart $\xi = \varphi(x)$ from the physical domain onto the latent
torus $[0, 1)^d$. `LearnedDeformation` is that chart: an affine map of the
bounding box into $[0, b]^d$, $b = 1/(1 + \mathrm{pad})$, plus a
zero-initialised residual (none, an MLP on NeRF features, or a Dirichlet
Laplacian eigenbasis whose `coeffs` pyrox puts a GP prior on), so it is the
plain box map at init. `jacobian_det` reports $\det \nabla\varphi$ as a
fold-over diagnostic.

`GeoFNO` is the operator. It reads a field at $N$ points with quadrature
weights, lifts it, transforms it onto a uniform latent grid with
`SpectralConv.from_points` at $\xi = \varphi(x)$, runs `FNOBlock`s there
(each plus a positional 1×1 conv of the latent coordinates), and evaluates
the result at $M$ query points with `SpectralConv.to_points` at
$\xi' = \varphi(x')$ before projecting. Ragged clouds batch by padding: pad
each sample to a common $N$ with in-box coordinates, `u = 0` and `w = 0`,
`jax.vmap` the model and mask the loss with `w > 0`; zero-weight points
contribute exact zeros. `jacobian_weights=True` reweights the quadrature by
$|\det \nabla\varphi|$ (off by default, pending
[#85](https://github.com/jejjohnson/geonnax/issues/85)).

::: geonnax.geofno.GeoFNO

::: geonnax.geofno.LearnedDeformation
