r"""Basis spectral convolution: an FNO-style layer in any precomputed basis.

`SphericalHarmonicTransform` / `SphericalSpectralConv` pair a dense, frozen
transform with a learned per-mode channel mixing, hard-wired to spherical
harmonics on a Gauss–Legendre grid. `BasisTransform` / `BasisSpectralConv`
are the same pattern for **any** basis $\Phi \in \mathbb{R}^{N \times M}$
sampled at $N$ points with quadrature weights $w \in \mathbb{R}^N$: region
Slepians (`slepian_region_basis`), EOFs (`eof_basis`), graph-Laplacian
eigenvectors (`graph_laplacian_eigpairs`), cotangent Laplace–Beltrami
eigenvectors of a mesh (`cotangent_laplacian_eigpairs`) or any other table. With
$\Phi_{\text{out}}$ the same basis at the output points,

$$
\hat a = \Phi^\top \operatorname{diag}(w)\,u^\top, \qquad
\hat y_m = W_m\,\hat a_m, \qquad
y = (\Phi_{\text{out}}\,\hat y)^\top + b .
$$

This is the Neural Operator on Riemannian Manifolds (NORM) layer, with the
Laplace–Beltrami eigenbasis replaced by any basis; the SFNO is the special case
of spherical harmonics with $W_m$ tied by degree.

References:
    Chen, Liu, Meng, Chen, Liu & Li (2024), "Learning Neural Operators on
    Riemannian Manifolds", National Science Open; arXiv:2302.08166.
    Simons, Dahlen & Wieczorek (2006), "Spatiospectral Concentration on a
    Sphere", SIAM Review 48(3), 504–536.
    Glorot & Bengio (2010), "Understanding the difficulty of training deep
    feedforward neural networks", AISTATS.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from geonnax._basis._mesh import cotangent_laplacian_eigpairs


if TYPE_CHECKING:
    from geonnax._basis._slepian_region import SlepianBasis


def _check_table(name: str, table: Array, n_modes: int | None = None) -> None:
    if table.ndim != 2:
        raise ValueError(f"{name} must be rank 2 (points, modes); got {table.shape}.")
    if n_modes is not None and table.shape[1] != n_modes:
        raise ValueError(
            f"{name} must have {n_modes} modes (columns); got {table.shape}."
        )


class BasisTransform(eqx.Module):
    r"""Frozen analysis / synthesis pair for a precomputed basis.

    Holds the analysis matrix $A = \Phi^\top \operatorname{diag}(w)$, shape
    ``(M, N)``, and the synthesis matrix $\Phi_{\text{out}}$, shape
    ``(N_out, M)``. Both are array leaves read through
    `jax.lax.stop_gradient`, so they never train; build one transform and
    share it across every `BasisSpectralConv` of a model.

    For a $w$-orthonormal basis ($\Phi^\top \operatorname{diag}(w)\,\Phi = I$)
    with $\Phi_{\text{out}} = \Phi$, ``inverse(forward(u))`` is the
    orthogonal projection of ``u`` onto $\operatorname{span}(\Phi)$, so it
    reproduces any field already in that span.

    Pseudocode:

    ```text
    forward(u):  â = u Aᵀ            # (C, N) → (C, M),     â_m ≈ ⟨u, φ_m⟩_w
    inverse(â):  y = â Φ_outᵀ        # (C, M) → (C, N_out)
    ```

    Attributes:
        analysis: ``(M, N)`` analysis matrix $\Phi^\top \operatorname{diag}(w)$
            (possibly row-normalised, see `from_slepian`).
        synthesis: ``(N_out, M)`` basis evaluated at the output points.
        n_modes: Number of modes ``M`` (static).

    Examples:
        A $w$-orthonormal basis reproduces fields in its span:

        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import BasisTransform
        >>> phi, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(0), (50, 6)))
        >>> transform = BasisTransform.from_matrix(phi, jnp.ones(50))
        >>> u = jr.normal(jr.PRNGKey(1), (2, 6)) @ phi.T   # (C, N) in span(Φ)
        >>> transform.forward(u).shape
        (2, 6)
        >>> bool(jnp.allclose(transform.inverse(transform.forward(u)), u, atol=1e-5))
        True

        A Slepian basis of a 60° polar cap, analysed over the cap and
        synthesised anywhere on the sphere:

        >>> from geonnax.basis import gauss_legendre_grid, slepian_region_basis
        >>> points, weights = gauss_legendre_grid(12)
        >>> cap = points[:, 2] > 0.5
        >>> basis = slepian_region_basis(5, points[cap], weights[cap])
        >>> transform = BasisTransform.from_slepian(
        ...     basis, points[cap], weights[cap], points_out_xyz=points)
        >>> transform.analysis.shape, transform.synthesis.shape
        ((9, 96), (288, 9))
    """

    analysis: Float[Array, "M N"]
    synthesis: Float[Array, "N_out M"]
    n_modes: int = eqx.field(static=True)

    @classmethod
    def from_matrix(
        cls,
        phi: Float[Array, "N M"],
        weights: Float[Array, " N"],
        phi_out: Float[Array, "N_out M"] | None = None,
    ) -> BasisTransform:
        r"""Build the transform from a basis table and quadrature weights.

        $A = \Phi^\top \operatorname{diag}(w)$ and $\Phi_{\text{out}}$ =
        ``phi_out`` (default ``phi``, i.e. output at the input points). The
        analysis is the quadrature $\hat a_m = \sum_n w_n \phi_m(x_n) u(x_n)
        \approx \langle u, \phi_m \rangle$, so it inverts the synthesis only
        when $\Phi$ is $w$-orthonormal; this constructor does not
        orthonormalise.

        Args:
            phi: ``(N, M)`` basis at the ``N`` input points.
            weights: ``(N,)`` quadrature (mass) weights,
                $\sum_n w_n g(x_n) \approx \int g$.
            phi_out: ``(N_out, M)`` the same basis at the output points;
                defaults to ``phi``.

        Returns:
            A `BasisTransform` with ``M`` modes.

        Raises:
            ValueError: If ``phi`` is not rank 2, ``weights`` is not ``(N,)``,
                or ``phi_out`` does not have ``M`` columns.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.layers import BasisTransform
            >>> transform = BasisTransform.from_matrix(jnp.eye(4)[:, :3], jnp.ones(4))
            >>> transform.n_modes, transform.synthesis.shape
            (3, (4, 3))
        """
        phi = jnp.asarray(phi)
        weights = jnp.asarray(weights)
        _check_table("phi", phi)
        if weights.shape != phi.shape[:1]:
            raise ValueError(f"weights must be ({phi.shape[0]},); got {weights.shape}.")
        n_modes = int(phi.shape[1])
        phi_out = phi if phi_out is None else jnp.asarray(phi_out)
        _check_table("phi_out", phi_out, n_modes)
        analysis = einx.multiply("n m, n -> m n", phi, weights)
        return cls(analysis=analysis, synthesis=phi_out, n_modes=n_modes)

    @classmethod
    def from_slepian(
        cls,
        basis: SlepianBasis,
        points_xyz: Float[Array, "N 3"],
        weights: Float[Array, " N"],
        points_out_xyz: Float[Array, "N_out 3"] | None = None,
    ) -> BasisTransform:
        r"""Build the transform from a region Slepian basis.

        ``points_xyz`` / ``weights`` should be a quadrature over the
        **region** (e.g. the one passed to `slepian_region_basis`). Slepian
        functions are orthonormal over the whole sphere but only orthogonal
        over the region, with $\sum_n w_n g_k(x_n) g_j(x_n) = \lambda_k
        \delta_{kj}$ (their concentrations; Simons et al. 2006). The analysis
        therefore divides each row by the region Gram diagonal
        $\lambda_k = \sum_n w_n g_k(x_n)^2$,

        $$
        A = \operatorname{diag}(\lambda)^{-1}\,\Phi^\top \operatorname{diag}(w),
        $$

        which makes ``inverse(forward(u))`` the region-orthogonal projection
        onto the Slepian span: exact for fields in that span. Energy of ``u``
        outside the span (or outside the region's band limit) leaks into the
        poorly concentrated modes, which the $1/\lambda_k$ factor amplifies;
        keep the basis at the Shannon number (the `slepian_region_basis`
        default), where every retained $\lambda_k$ is large.

        The synthesis evaluates the basis with ``basis.evaluate`` at
        ``points_out_xyz`` (default ``points_xyz``), which may be any points on
        the sphere.

        Args:
            basis: A `SlepianBasis` (anything with ``evaluate(xyz) -> (N, K)``).
            points_xyz: ``(N, 3)`` unit vectors of the region quadrature.
            weights: ``(N,)`` region quadrature weights.
            points_out_xyz: ``(N_out, 3)`` unit vectors for the output.

        Returns:
            A `BasisTransform` with ``K = basis.num_modes`` modes.

        Raises:
            ValueError: If the points are not ``(N, 3)`` or ``weights`` is not
                ``(N,)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.basis import gauss_legendre_grid, slepian_region_basis
            >>> from geonnax.layers import BasisTransform
            >>> points, weights = gauss_legendre_grid(12)
            >>> cap = points[:, 2] > 0.5   # a cap of half-angle 60°
            >>> basis = slepian_region_basis(5, points[cap], weights[cap])
            >>> transform = BasisTransform.from_slepian(
            ...     basis, points[cap], weights[cap])
            >>> transform.n_modes, transform.synthesis.shape
            (9, (96, 9))
        """
        points = jnp.asarray(points_xyz)
        weights = jnp.asarray(weights)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"points_xyz must be (N, 3); got {points.shape}.")
        if weights.shape != points.shape[:1]:
            raise ValueError(
                f"weights must be ({points.shape[0]},); got {weights.shape}."
            )
        phi = basis.evaluate(points)
        if points_out_xyz is None:
            phi_out = phi
        else:
            out = jnp.asarray(points_out_xyz)
            if out.ndim != 2 or out.shape[-1] != 3:
                raise ValueError(f"points_out_xyz must be (N_out, 3); got {out.shape}.")
            phi_out = basis.evaluate(out)
        transform = cls.from_matrix(phi, weights, phi_out)
        gram_diag = einx.dot("m n, n m -> m", transform.analysis, phi)
        analysis = einx.divide("m n, m -> m n", transform.analysis, gram_diag)
        return cls(analysis=analysis, synthesis=phi_out, n_modes=transform.n_modes)

    @classmethod
    def from_mesh(
        cls,
        vertices: Float[Array, "V 3"],
        triangles: Int[Array, "T 3"],
        num_basis: int,
    ) -> BasisTransform:
        r"""Build the transform from a mesh's Laplace–Beltrami eigenbasis.

        $\Phi$ holds the ``num_basis`` lowest eigenvectors of the cotangent
        Laplacian (`cotangent_laplacian_eigpairs`), which are orthonormal
        under the lumped vertex-area mass, $\Phi^\top \operatorname{diag}(a)
        \,\Phi = I$. So $A = \Phi^\top \operatorname{diag}(a)$ with the
        vertex areas $a$ as quadrature weights, and the synthesis is $\Phi$:
        input and output are both the ``V`` vertices, and
        ``inverse(forward(u))`` is the $L^2$-orthogonal projection of a vertex
        field onto the low-frequency span. This is the NORM basis (Chen et al.
        2024).

        The eigenvectors come from a dense ``eigh``: their signs, and the basis
        chosen within a degenerate eigenspace (e.g. the $2\ell + 1$ degree-$\ell$
        modes of a sphere), are whatever the solver returns. Per-mode weights
        are tied to that order, so pick ``num_basis`` at the end of a
        degenerate cluster (9 = degrees $\ell \le 2$ on a sphere) and reuse
        the same transform when sharing weights. The construction is eager
        (NumPy, not traceable under ``jit``).

        Args:
            vertices: ``(V, 3)`` vertex positions.
            triangles: ``(T, 3)`` integer vertex indices.
            num_basis: Number of modes ``M``, ``1 <= M <= V``.

        Returns:
            A `BasisTransform` with ``M`` modes, ``(M, V)`` analysis and
            ``(V, M)`` synthesis.

        Raises:
            ValueError: As `cotangent_laplacian_eigpairs` (bad mesh or
                ``num_basis``).

        Examples:
            >>> from geonnax.basis import icosphere
            >>> from geonnax.layers import BasisTransform
            >>> vertices, triangles = icosphere(2)
            >>> transform = BasisTransform.from_mesh(vertices, triangles, num_basis=9)
            >>> transform.analysis.shape, transform.synthesis.shape
            ((9, 162), (162, 9))
        """
        _, phi, areas = cotangent_laplacian_eigpairs(vertices, triangles, num_basis)
        return cls.from_matrix(phi, areas)

    def forward(self, u: Float[Array, "C N"]) -> Float[Array, "C M"]:
        r"""Analyse ``(C, N)`` point values into ``(C, M)`` coefficients.

        Computes $\hat a = u A^\top$.
        """
        analysis = jax.lax.stop_gradient(self.analysis)
        return einx.dot("m n, c n -> c m", analysis, u)

    def inverse(self, a: Float[Array, "C M"]) -> Float[Array, "C N_out"]:
        r"""Synthesise ``(C, M)`` coefficients at the ``N_out`` output points.

        Computes $y = a\,\Phi_{\text{out}}^\top$.
        """
        synthesis = jax.lax.stop_gradient(self.synthesis)
        return einx.dot("p m, c m -> c p", synthesis, a)


class BasisSpectralConv(eqx.Module):
    r"""Spectral convolution in a precomputed basis: per-mode channel mixing.

    $$
    y = \Phi_{\text{out}}\big(W_m\,(A u^\top)_m\big)_{m=1}^{M} + b,
    $$

    with one $C_{\text{out}} \times C_{\text{in}}$ matrix $W_m$ per mode (the
    NORM layer of Chen et al. 2024 in any basis). The shared `BasisTransform`
    is passed at call time, as `SphericalSpectralConv` takes its SHT. Tying
    $W_m = W_{\ell(m)}$ by spherical-harmonic degree on the SHT tables
    recovers `SphericalSpectralConv` exactly.

    Pseudocode:

    ```text
    â = transform.forward(u)            # (C_in, M)
    ŷ[:, m] = weight[m] @ â[:, m]       # (C_out, M)
    y = transform.inverse(ŷ) + bias     # (C_out, N_out)
    ```

    Weights are Glorot-scaled Gaussians, $\mathcal N(0, 2/(C_{\text{in}} +
    C_{\text{out}}))$ (Glorot & Bengio 2010), matching `SphericalSpectralConv`.

    Attributes:
        weight: ``(M, C_out, C_in)`` per-mode mixing matrices.
        bias: ``(C_out, 1)`` bias, or ``None``.
        n_modes: Number of modes ``M`` (static).
        in_channels: ``C_in`` (static).
        out_channels: ``C_out`` (static).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.basis import gauss_legendre_grid, slepian_region_basis
        >>> from geonnax.layers import BasisSpectralConv, BasisTransform
        >>> points, weights = gauss_legendre_grid(12)
        >>> cap = points[:, 2] > 0.5
        >>> basis = slepian_region_basis(5, points[cap], weights[cap])
        >>> transform = BasisTransform.from_slepian(basis, points[cap], weights[cap])
        >>> conv = BasisSpectralConv.init(3, 5, transform.n_modes, key=jr.PRNGKey(0))
        >>> conv(jnp.ones((3, 96)), transform).shape   # 3 -> 5 channels
        (5, 96)
    """

    weight: Float[Array, "M C_out C_in"]
    bias: Float[Array, "C_out 1"] | None
    n_modes: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        *,
        key: Array,
        bias: bool = True,
        init_scale: float | None = None,
    ) -> BasisSpectralConv:
        """Construct a basis spectral convolution with ``n_modes`` modes.

        Args:
            in_channels: Input channels ``C_in``.
            out_channels: Output channels ``C_out``.
            n_modes: Number of basis modes ``M`` (``transform.n_modes``).
            key: PRNG key for the weights.
            bias: Whether to add a learned ``(C_out, 1)`` bias (zeros at init).
            init_scale: Standard deviation of the weights; defaults to
                ``sqrt(2 / (C_in + C_out))``.

        Returns:
            A `BasisSpectralConv`.

        Raises:
            ValueError: If a channel count or ``n_modes`` is below 1.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.layers import BasisSpectralConv
            >>> BasisSpectralConv.init(2, 4, 9, key=jr.PRNGKey(0)).weight.shape
            (9, 4, 2)
        """
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("n_modes", n_modes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}.")
        scale = (
            math.sqrt(2.0 / (in_channels + out_channels))
            if init_scale is None
            else init_scale
        )
        weight = scale * jax.random.normal(key, (n_modes, out_channels, in_channels))
        return cls(
            weight=weight,
            bias=jnp.zeros((out_channels, 1)) if bias else None,
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def __call__(
        self, u: Float[Array, "C_in N"], transform: BasisTransform
    ) -> Float[Array, "C_out N_out"]:
        """Apply the layer to one ``(C_in, N)`` example.

        Raises:
            ValueError: If ``transform.n_modes`` differs from ``n_modes``.
        """
        if transform.n_modes != self.n_modes:
            raise ValueError(
                f"transform has {transform.n_modes} modes but the layer has "
                f"{self.n_modes}."
            )
        coeffs = transform.forward(u)
        mixed = einx.dot("m o i, i m -> o m", self.weight, coeffs)
        out = transform.inverse(mixed)
        if self.bias is not None:
            out = einx.add("o p, o 1 -> o p", out, self.bias)
        return out


__all__ = ["BasisSpectralConv", "BasisTransform"]
