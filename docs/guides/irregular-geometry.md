# FNO on curvilinear grids in index space

Many geoscience fields live on a **structured curvilinear grid**: a satellite swath
(scanline × ground pixel), an ORCA tripolar ocean grid, terrain-following sigma levels, a
C-grid around an airfoil. The array is a regular tensor, but the *physical* position of
entry $(i, j)$ is not a regular lattice. You do not need to regrid, and you do not need
to wait for a dedicated irregular-geometry operator. Run the existing
[`FNO`](../api/operators.md#geonnax.fno.FNO) on the **index grid** and give it the
physical coordinates as extra **input channels**.

## Which route for which geometry

| Case | Geometry | Route |
|---|---|---|
| A | Fixed structured curvilinear grid, no holes (swath, ORCA, sigma levels) | `FNO` in index space, coordinates as channels (this page) |
| B | Fixed unstructured mesh or irregular region, same for every sample | A basis fitted to the domain (forthcoming `ManifoldNO`, #88) |
| C | Point cloud or mesh that changes from sample to sample | A deformation onto a periodic latent grid (forthcoming `GeoFNO`, #84) |

Only case A is available today, and it needs no new code.

## Formulation

Let the grid have index coordinates $\xi = (i, j) \in \{0,\dots,n_1-1\} \times
\{0,\dots,n_2-1\}$ and let $\mathbf{x}(\xi) \in \mathbb{R}^2$ be the physical position of
node $\xi$ (a metric $(x, y)$ in km, not raw longitude and latitude). Let
$a(\xi)\in\mathbb{R}^{c}$ be the observed fields. The model input is the channel stack

$$
v_0(\xi) \;=\; P\big([\,a(\xi),\; \mathbf{x}(\xi)\,]\big) \in \mathbb{R}^{d_v},
$$

with $P$ the pointwise lifting (`FNO.lifting`, a $1\times1$ conv), followed by $L$ Fourier layers
(`FNOBlock`) and a two-layer pointwise projection $Q$ (`proj1`, `proj2`):

$$
v_{t+1}(\xi) \;=\; \sigma\!\Big(W v_t(\xi) \;+\; \mathcal{F}^{-1}\big[\,R \cdot \mathcal{F}[v_t]\,\big](\xi)\Big),
\qquad u(\xi) = Q\big(v_L(\xi)\big).
$$

Here $\mathcal{F}$ is the discrete Fourier transform over the *index* axes, $R$ is a
learned complex weight applied to the lowest `n_modes` frequencies only
(`SpectralConv`), $W$ is the local $1\times1$ channel mix, and $\sigma$ is the
activation (the last block is linear). This is exactly the FNO of
Li et al. (2021); nothing in it knows that the grid is curved.

This is also what the reference implementation of Geo-FNO (Li et al., 2023) does on its
structured-mesh experiments: the airfoil C-grid ($221\times51$), the pipe
($129\times129$) and plasticity ($101\times31$) problems are plain FNOs over the index
grid with $(x, y)$ stacked as input channels. The mesh *is* the deformation from the
physical domain to a rectangle, so no chart is learned.

Three consequences follow directly from the formulation.

- **The FFT sees index spacing, not metric spacing.** $\mathcal{F}$ treats neighbouring
  entries as one unit apart. Uneven pixel sizes (for example the across-track growth of
  a nadir-looking imager's footprint) are *not* corrected; they are made available to the
  network through $\mathbf{x}(\xi)$, and $W$ and the lifting can learn to use them. Scale
  coordinates to $O(1)$ (here, divide kilometres by $100$) so the channels are
  comparable to the field.
- **Padding.** The FFT is periodic over the index axes, so it couples the first and last
  scanline (and pixel). A swath is not periodic. Pass `domain_padding` so the field is
  zero-padded by that fraction of each extent before the Fourier layers and cropped back
  afterwards.
- **Resolution.** Each index extent must exceed twice its mode count (after padding),
  e.g. `n_modes=(6, 6)` needs at least $13$ entries per axis.

## Pseudocode

```text
inputs : fields a[c, n1, n2]; lon[n1, n2], lat[n1, n2]
1. (e, n) <- local_tangent_plane(lon, lat; origin = grid centre)   # km, (N, 2)
2. coords[2, n1, n2] <- (e, n) reshaped, divided by a length scale  # ~ O(1)
3. v[c+2, n1, n2]    <- concat(a, coords)                           # channel stack
4. v <- pad(v, domain_padding)                                      # break periodicity
5. v <- lift(v)                                                     # pointwise
6. for block in 1..L:
       v <- act( SpectralConv(v) + W v )                            # FFT over index axes
7. u <- project(v); u <- crop(u)                                    # pointwise, unpad
return u[c_out, n1, n2]
```

## Running it

The example builds a small synthetic swath: a tilted track whose footprint widens away
from nadir. It uses only `geonnax.FNO` and `geonnax.geo.local_tangent_plane`
(`local_tangent_plane(lonlat, origin, *, projection, radius, input_unit)` returns east
and north offsets in the units of `radius`, kilometres by default).

```python
import jax
import jax.numpy as jnp
import jax.random as jr

from geonnax import FNO
from geonnax.geo import local_tangent_plane

n_scan, n_pix = 40, 24

# A synthetic swath: the track is tilted, the pixel footprint grows across track.
s = jnp.linspace(0.0, 1.0, n_scan)[:, None]  # along track
p = jnp.linspace(-1.0, 1.0, n_pix)[None, :]  # across track
width = 0.5 * (1.0 + 0.3 * jnp.abs(p))  # footprint growth away from nadir
lon = 10.0 + 3.0 * s + width * p * 0.9  # (n_scan, n_pix)
lat = 40.0 + 5.0 * s - 0.2 * width * p  # (n_scan, n_pix)
xch4 = jnp.sin(3.0 * s) * jnp.cos(2.0 * p)  # (n_scan, n_pix)

# Metric coordinates (km) about the swath centre, scaled to O(1).
lonlat = jnp.stack([lon.ravel(), lat.ravel()], -1)  # (N, 2)
origin = jnp.array([lon.mean(), lat.mean()])
enu = local_tangent_plane(lonlat, origin)  # (N, 2) km
coords = enu.T.reshape(2, n_scan, n_pix) / 100.0  # (2, n_scan, n_pix)

x = jnp.concatenate([xch4[None], coords], axis=0)  # (3, n_scan, n_pix)
op = FNO.init(
    3, 1, (6, 6), key=jr.PRNGKey(0), hidden_channels=16, n_layers=2,
    domain_padding=0.1,
)
y = op(x)  # (1, n_scan, n_pix)

# A batch of swaths is a vmap over the leading axis.
yb = jax.vmap(op)(jnp.stack([x, x]))  # (2, 1, n_scan, n_pix)
```

As everywhere in geonnax, the operator takes one example (no batch axis); use
`jax.vmap` for a batch. Because the grid is read at call time, the same parameters also
run on a swath with a different number of scanlines, provided the extents stay above
twice the mode count.

## When this breaks

Holes or invalid pixels *inside* the grid (cloud or quality masks) break the tensor
structure: the FFT would mix fill values into valid neighbours. Zero-filling and
masking the loss only hides the problem. The right route for per-sample valid-pixel sets
is case C (`GeoFNO`, #84, with zero weights on invalid pixels), once available. Likewise
a grid that is not a single tensor (a cubed sphere's six panels, an ORCA fold treated as
a seam) is outside this recipe.

## A chart from optimal transport (OTNO recipe)

When the geometry is *not* a structured grid but you can afford a per-shape
preprocessing step, the chart can come from optimal transport instead of a learned
deformation. This is the OTNO recipe of Li, Li, Kovachki and Anandkumar (2025): the
model is the same `FNO` as above, and all of the geometry lives in an index map computed
**offline**.

For a shape with $M$ surface points $\{p_m\}$ and a latent regular grid with $N$ nodes
$\{g_n\}$ (a torus, so the FFT's periodicity is harmless), solve an optimal-transport
problem between the two point sets (an entropic plan via Sinkhorn, or a Monge map) and
keep, for every surface point, the latent node it is sent to:

$$
\pi^\star \in \arg\min_{\pi \in \Pi(\mu, \nu)} \sum_{m,n} \pi_{mn}\,\lVert p_m - g_n \rVert^2,
\qquad
\mathrm{ind}(m) = \arg\max_n\, \pi^\star_{mn}.
$$

Features are scattered onto the latent grid, the operator acts there, and the output is
gathered back through the stored index:

```text
offline, once per shape (data side, not in geonnax):
    pi   <- OT(surface_points, latent_grid_nodes)         # needs POT / geomloss
    ind  <- argmax_n pi[m, n]   for each surface point m  # int array, (M,)
    a_grid[c, n1, n2] <- scatter(a_surface, ind)          # features onto the grid
at train / inference time:
    u_grid <- FNO(concat(a_grid, grid_coords))            # exactly the call above
    u_surface[m] <- u_grid.reshape(c_out, -1)[:, ind[m]]  # a gather
```

geonnax adds nothing to this: the optimal-transport solve needs libraries that are not
geonnax dependencies and belongs in the data pipeline, and the network is the `FNO`
already documented above. The reported gains of OTNO come from this per-shape chart, not
from a different architecture.

## References

- Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart,
  A. Anandkumar. *Fourier Neural Operator for Parametric Partial Differential
  Equations.* ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895).
- Z. Li, D. Z. Huang, B. Liu, A. Anandkumar. *Fourier Neural Operator with Learned
  Deformations for PDEs on General Geometries.* Journal of Machine Learning Research
  24(388):1-26, 2023. [arXiv:2207.05209](https://arxiv.org/abs/2207.05209).
  Reference code: `neuraloperator/Geo-FNO`, `airfoils/naca_geofno.py`.
- Z. Li, B. Li, N. Kovachki, A. Anandkumar. *Geometric Operator Learning with Optimal
  Transport.* NeurIPS 2025. [arXiv:2507.20065](https://arxiv.org/abs/2507.20065).
