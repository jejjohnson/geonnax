"""Fourier spectral convolution (the core Fourier Neural Operator layer).

A `SpectralConv` mixes channels in the truncated Fourier domain: it
`rfft`s the input, keeps the lowest `n_modes` frequencies, applies a
learnable complex channel mixing there, and inverts the transform. Because the
spatial grid size is read from the input at call time (never baked into the
module), the same layer evaluates at any resolution above the mode count — the
resolution-invariance that defines the FNO (Li et al., 2021).

The complex mixing weights can be stored densely or in a low-rank factorised
form (CP / Tucker / TT) via `geonnax.layers._factorized`, which trades a
large parameter count for a compact, regularised one (Kossaifi et al., 2023).

`SpectralConv.from_points` / `SpectralConv.to_points` reuse the same weights
between a point cloud and a uniform latent grid through the weighted
non-uniform transforms of `geonnax.layers._nonuniform` (the Geo-FNO layers of
Li et al., 2023).
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from geonnax.layers._factorized import (
    CPTensor,
    DenseTensor,
    Factorization,
    TTTensor,
    TuckerTensor,
    init_factorized_tensor,
)
from geonnax.layers._nonuniform import nonuniform_irfft, nonuniform_rfft
from geonnax.layers._utils import _corner_slices


FactorizedTensor = DenseTensor | CPTensor | TuckerTensor | TTTensor


class SpectralConv(eqx.Module):
    """Fourier spectral convolution over ``num_spatial_dims = len(n_modes)`` axes.

    Operates on a single example of shape ``(in_channels, *spatial)``; ``vmap``
    over a batch. Each spatial extent must exceed twice its mode count so the
    low/high-frequency corner blocks do not overlap.

    The forward map is, per retained mode ``k``,
    ``ŷ[o, k] = Σ_i Ŵ[i, o, k] · x̂[i, k]`` where ``x̂ = rfft(x)`` and ``Ŵ`` is
    the (optionally factorized) complex weight.

    Attributes:
        weights: One factorized complex weight per spectral corner block; each
            reconstructs to ``(in_channels, out_channels, *n_modes, 2)`` (the
            trailing axis holds real/imag parts).
        bias: Optional spatial-domain bias of shape ``(out_channels, *ones)``.
        in_channels, out_channels: Channel counts (static).
        n_modes: Retained modes per spatial axis (static).
        factorization: Weight parameterisation (static).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import SpectralConv
        >>> sc = SpectralConv.init(4, 6, (8, 8), key=jr.PRNGKey(0))
        >>> sc(jnp.ones((4, 32, 32))).shape
        (6, 32, 32)
        >>> sc(jnp.ones((4, 64, 64))).shape   # resolution-invariant
        (6, 64, 64)

        With a Tensor-Train factorization of the spectral weights:

        >>> sc = SpectralConv.init(8, 8, (4, 4, 4), key=jr.PRNGKey(0),
        ...                        factorization="tt", rank=0.5)
        >>> sc(jnp.ones((8, 16, 16, 16))).shape
        (8, 16, 16, 16)

        The same weights also map a point cloud in $[0, 1)^d$ (with quadrature
        weights) onto a latent grid, and a latent grid onto query points:

        >>> sc = SpectralConv.init(2, 2, (6, 6), key=jr.PRNGKey(0))
        >>> xi = jr.uniform(jr.PRNGKey(1), (500, 2))            # (N, d)
        >>> w = jnp.full((500,), 1 / 500)                       # (N,)
        >>> z = sc.from_points(jnp.ones((2, 500)), xi, w, (32, 32))
        >>> z.shape
        (2, 32, 32)
        >>> sc.to_points(z, xi).shape                           # back to the points
        (2, 500)
    """

    weights: list[FactorizedTensor]
    bias: Float[Array, "C_out *ones"] | None
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: tuple[int, ...] = eqx.field(static=True)
    factorization: Factorization = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        *,
        key: Array,
        factorization: Factorization = "dense",
        rank: float | int = 0.5,
        bias: bool = True,
        init_scale: float | None = None,
    ) -> SpectralConv:
        """Construct a spectral convolution.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            n_modes: Retained Fourier modes per spatial axis; its length is the
                spatial rank.
            key: PRNG key.
            factorization: ``"dense"``, ``"cp"``, ``"tucker"``, or ``"tt"``.
            rank: Fractional (``0 < rank <= 1``) or absolute rank for the
                factorised forms; ignored when ``factorization="dense"``.
            bias: Whether to add a learnable spatial-domain bias.
            init_scale: Std of the (reconstructed) weights; defaults to the
                fan-based ``sqrt(2 / (in + out))``.

        Raises:
            ValueError: If ``n_modes`` is empty or any entry is non-positive.
        """
        if len(n_modes) == 0:
            raise ValueError("n_modes must be non-empty.")
        if any(m < 1 for m in n_modes):
            raise ValueError(f"all n_modes must be >= 1, got {n_modes}.")
        n_modes = tuple(int(m) for m in n_modes)
        scale = (
            math.sqrt(2.0 / (in_channels + out_channels))
            if init_scale is None
            else init_scale
        )
        n_corners = 2 ** (len(n_modes) - 1)
        # Trailing 2 carries real/imag, factorised like any other tensor axis.
        weight_shape = (in_channels, out_channels, *n_modes, 2)
        keys = jax.random.split(key, n_corners)
        weights = [
            init_factorized_tensor(
                weight_shape, factorization, key=keys[c], rank=rank, scale=scale
            )
            for c in range(n_corners)
        ]
        bias_arr = jnp.zeros((out_channels, *(1,) * len(n_modes))) if bias else None
        return cls(
            weights=weights,
            bias=bias_arr,
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            factorization=factorization,
        )

    def _weight(self, corner: int) -> Complex[Array, "C_in C_out *modes"]:
        """Reconstruct corner ``corner``'s weights and view them as complex."""
        w = self.weights[corner].reconstruct()
        return w[..., 0] + 1j * w[..., 1]

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        spatial = x.shape[1:]
        x_ft = jnp.fft.rfftn(x, axes=tuple(range(1, len(self.n_modes) + 1)))
        blocks = [x_ft[(slice(None), *sl)] for sl in _corner_slices(self.n_modes)]
        out = self._scatter_irfftn(self._mix(blocks), spatial)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _mix(
        self, blocks: list[Complex[Array, "C_in *n_modes"]]
    ) -> list[Complex[Array, "C_out *n_modes"]]:
        """Per-corner channel mixing ``ŷ[o, k] = Σ_i Ŵ[i, o, k] · x̂[i, k]``.

        Applies corner ``c``'s reconstructed weight to ``blocks[c]``; shared by
        `__call__`, `from_points` and `to_points`.
        """
        modes = " ".join(f"m{a}" for a in range(len(self.n_modes)))
        return [
            einx.dot(f"i {modes}, i o {modes} -> o {modes}", b, self._weight(c))
            for c, b in enumerate(blocks)
        ]

    def _scatter_irfftn(
        self, blocks: list[Complex[Array, "C_out *n_modes"]], grid: tuple[int, ...]
    ) -> Float[Array, "C_out *grid"]:
        """Scatter corner blocks into a zero rfft spectrum and invert on ``grid``."""
        out_ft = jnp.zeros(
            (self.out_channels, *grid[:-1], grid[-1] // 2 + 1), dtype=blocks[0].dtype
        )
        for sl, block in zip(_corner_slices(self.n_modes), blocks, strict=True):
            out_ft = out_ft.at[(slice(None), *sl)].set(block)
        return jnp.fft.irfftn(out_ft, s=grid, axes=tuple(range(1, len(grid) + 1)))

    def _check_grid(self, grid: tuple[int, ...]) -> None:
        """Raise unless ``grid`` has one extent per mode axis, each above ``2 m``."""
        if len(grid) != len(self.n_modes) or any(
            2 * m >= s for m, s in zip(self.n_modes, grid, strict=True)
        ):
            raise ValueError(
                f"grid must have {len(self.n_modes)} extents, each greater than "
                f"2 * n_modes = {tuple(2 * m for m in self.n_modes)}, got {grid}."
            )

    def from_points(
        self,
        u: Float[Array, "C_in N"],
        xi: Float[Array, "N d"],
        weights: Float[Array, " N"],
        grid: tuple[int, ...],
    ) -> Float[Array, "C_out *grid"]:
        r"""Spectral convolution from a point cloud onto a uniform latent grid.

        The input half of Geo-FNO (Li, Huang, Liu and Anandkumar, *Fourier
        Neural Operator with Learned Deformations for PDEs on General
        Geometries*, JMLR 24, 2023, arXiv:2207.05209), with quadrature
        weights. With $S = \prod_a s_a$ nodes on ``grid`` and $\hat W_k$ the
        same (possibly factorised) weights `__call__` uses, for every retained
        mode $k$:

        $$\hat v_k = \sum_{n=1}^{N} w_n\, u_n\, e^{-2\pi i\, k\cdot\xi_n},
        \qquad \hat y_k = \hat W_k\, \hat v_k,
        \qquad y = \mathrm{irfftn}\big(S\, \hat y,\ s = \mathrm{grid}\big) + b .$$

        $\hat v$ is `nonuniform_rfft`; the factor $S$ converts the continuous
        coefficient to the DFT scale that ``irfftn`` expects.

        ```text
        v̂    = nonuniform_rfft(u, xi, weights, n_modes)  # 2**(d-1) corners (C_in, *m)
        ŷ    = _mix(v̂)                                    # corners (C_out, *m)
        spec = zeros(C_out, *grid[:-1], grid[-1]//2 + 1)
        spec[corners] = S * ŷ
        return irfftn(spec, s=grid) + bias
        ```

        On the grid nodes themselves with $w_n = 1/S$ this equals `__call__`.
        For a scattered cloud the weights should approximate cell volumes
        ($w_n \ge 0$, $\sum_n w_n \approx 1$; $1/N$ for a stratified cloud). A
        point with $w_n = 0$ is ignored, so ragged clouds zero-padded to a
        common $N$ can be ``jax.vmap``-ed (padded coordinates must be finite).

        Args:
            u: Field values at the points, shape ``(in_channels, N)``.
            xi: Latent coordinates in $[0, 1)^d$, shape ``(N, d)`` with
                ``d = len(n_modes)``.
            weights: Quadrature weights, shape ``(N,)``.
            grid: Static output grid extents, one per axis, each greater than
                twice its mode count.

        Returns:
            The field on the uniform grid, shape ``(out_channels, *grid)``.

        Raises:
            ValueError: If ``grid`` has the wrong length or under-resolves
                ``n_modes``, or the shapes of ``u``, ``xi`` and ``weights``
                disagree.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.layers import SpectralConv
            >>> sc = SpectralConv.init(3, 5, (6, 6), key=jr.PRNGKey(0))
            >>> xi = jr.uniform(jr.PRNGKey(1), (400, 2))     # (N, d) in [0, 1)^2
            >>> u = jnp.ones((3, 400))                       # (C_in, N)
            >>> w = jnp.full((400,), 1 / 400)                # sum(w) = 1
            >>> sc.from_points(u, xi, w, (32, 32)).shape
            (5, 32, 32)
        """
        grid = tuple(int(s) for s in grid)
        self._check_grid(grid)
        blocks = self._mix(nonuniform_rfft(u, xi, weights, self.n_modes))
        scale = math.prod(grid)
        out = self._scatter_irfftn([scale * b for b in blocks], grid)
        if self.bias is not None:
            out = out + self.bias
        return out

    def to_points(
        self,
        x: Float[Array, "C_in *grid"],
        xi: Float[Array, "M d"],
    ) -> Float[Array, "C_out M"]:
        r"""Spectral convolution from a uniform latent grid to scattered points.

        The output half of Geo-FNO (Li et al., JMLR 2023, arXiv:2207.05209):
        the convolved field's truncated Fourier series is evaluated at
        arbitrary points instead of at the grid nodes. With
        $S = \prod_a s_a$ and $\mu_k = 1$ if $k_d = 0$, else $2$
        (`nonuniform_irfft`):

        $$\hat x_k = \mathrm{rfftn}(x)_k / S, \qquad
        \hat y_k = \hat W_k\, \hat x_k, \qquad
        y(\xi_m) = \sum_k \mu_k\, \mathrm{Re}\big[\hat y_k\,
        e^{2\pi i\, k\cdot\xi_m}\big] + b .$$

        ```text
        x̂ = corners(rfftn(x)) / S                       # 2**(d-1) corners (C_in, *m)
        ŷ = _mix(x̂)                                     # corners (C_out, *m)
        return nonuniform_irfft(ŷ, xi, n_modes) + bias   # (C_out, M)
        ```

        At the grid nodes this equals `__call__`, flattened to
        ``(out_channels, S)``.

        Args:
            x: Field on a uniform grid of $[0, 1)^d$, shape
                ``(in_channels, *grid)``, each extent greater than twice its
                mode count.
            xi: Query coordinates in $[0, 1)^d$, shape ``(M, d)``.

        Returns:
            The convolved field at the query points, shape
            ``(out_channels, M)``.

        Raises:
            ValueError: If the grid of ``x`` has the wrong rank or
                under-resolves ``n_modes``, or ``xi`` has the wrong number of
                columns.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.layers import SpectralConv
            >>> sc = SpectralConv.init(3, 5, (6, 6), key=jr.PRNGKey(0))
            >>> query = jr.uniform(jr.PRNGKey(1), (250, 2))  # (M, d) in [0, 1)^2
            >>> sc.to_points(jnp.ones((3, 32, 32)), query).shape
            (5, 250)
        """
        grid = tuple(x.shape[1:])
        self._check_grid(grid)
        x_ft = jnp.fft.rfftn(x, axes=tuple(range(1, len(grid) + 1)))
        scale = math.prod(grid)
        blocks = [
            x_ft[(slice(None), *sl)] / scale for sl in _corner_slices(self.n_modes)
        ]
        out = nonuniform_irfft(self._mix(blocks), xi, self.n_modes)
        if self.bias is not None:
            bias = einx.id("o ... -> (o ...)", self.bias)  # (C_out, 1, …) -> (C_out,)
            out = einx.add("o m, o -> o m", out, bias)
        return out


__all__ = ["SpectralConv"]
