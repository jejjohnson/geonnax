r"""Manifold Neural Operator: an FNO in a fixed domain's own spectral basis.

This is the Neural Operator on Riemannian Manifolds (NORM; Chen et al. 2024),
generalised from the Laplace–Beltrami eigenbasis to any precomputed basis. The
FNO's FFT is replaced by a frozen `BasisTransform` built once for the domain:

- cotangent Laplace–Beltrami eigenvectors of a triangle mesh
  (`BasisTransform.from_mesh`, the NORM basis), for an ice-sheet mesh or an
  icosphere;
- region Slepian functions on the sphere (`BasisTransform.from_slepian`), for
  an ocean basin as a spherical polygon. This variant is new to geonnax: the
  basis is concentrated on the region, so no regridding and no land mask
  is needed;
- any other table with quadrature weights (`BasisTransform.from_matrix`), e.g.
  EOFs or graph-Laplacian eigenvectors.

The structure is the FNO's: lift to a hidden width, apply `ManifoldNOBlock` layers
(a `BasisSpectralConv` summed with a pointwise skip), then project. One
transform is shared by every block. Inputs are single examples of shape
``(channels, N)``, a field at the transform's ``N`` points; ``jax.vmap`` over a
batch.

The weights are tied to *this* domain's modes, so the operator does not
transfer to another geometry. Evaluating the output at query points other than
the input points (a synthesis on other points in the last block) is a planned
follow-up; version 1 keeps input and output points equal.

References:
    Chen, Liu, Meng, Chen, Liu & Li (2024), "Learning Neural Operators on
    Riemannian Manifolds", National Science Open; arXiv:2302.08166.
    Simons, Dahlen & Wieczorek (2006), "Spatiospectral Concentration on a
    Sphere", SIAM Review 48(3), 504–536.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float

from geonnax.layers._basis_conv import BasisSpectralConv, BasisTransform
from geonnax.layers._utils import pointwise


class ManifoldNOBlock(eqx.Module):
    r"""One manifold-operator layer: basis spectral conv + pointwise skip.

    Computes $\sigma\big(\mathcal K(h;\,T) + W h\big)$, with $\mathcal K$ a
    `BasisSpectralConv` in the shared transform $T$ (passed at call time, as
    `SphericalFNOBlock` takes its SHT) and $W$ a ``1×1`` channel mix over the
    point axis.

    Pseudocode:

    ```text
    h = spectral(x, transform) + pointwise(x)   # (C, N)
    return activation(h) if use_activation else h
    ```

    Attributes:
        spectral: Per-mode channel mixing in the basis (`BasisSpectralConv`).
        pointwise: Local ``1×1`` channel-mixing skip (1-D ``eqx.nn.Conv``).
        activation: Pointwise nonlinearity (static).
        use_activation: Whether to apply the activation (static; ``False`` for
            a linear final block).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.basis import icosphere
        >>> from geonnax.layers import BasisTransform
        >>> from geonnax.manifold_no import ManifoldNOBlock
        >>> transform = BasisTransform.from_mesh(*icosphere(2), num_basis=9)
        >>> blk = ManifoldNOBlock.init(8, transform.n_modes, key=jr.PRNGKey(0))
        >>> blk(jnp.ones((8, 162)), transform).shape   # width preserved
        (8, 162)
    """

    spectral: BasisSpectralConv
    pointwise: eqx.nn.Conv
    activation: Callable[[Array], Array] = eqx.field(static=True)
    use_activation: bool = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        channels: int,
        n_modes: int,
        *,
        key: Array,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        use_activation: bool = True,
    ) -> ManifoldNOBlock:
        """Construct a block at fixed channel width ``channels``.

        Args:
            channels: Channel width ``C`` (input and output).
            n_modes: Number of basis modes ``M`` (``transform.n_modes``).
            key: PRNG key.
            activation: Pointwise nonlinearity.
            use_activation: Whether to apply ``activation``.

        Returns:
            A `ManifoldNOBlock`.

        Raises:
            ValueError: If ``channels`` or ``n_modes`` is below 1.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.manifold_no import ManifoldNOBlock
            >>> ManifoldNOBlock.init(4, 9, key=jr.PRNGKey(0)).spectral.weight.shape
            (9, 4, 4)
        """
        k_sp, k_pw = jax.random.split(key)
        spectral = BasisSpectralConv.init(channels, channels, n_modes, key=k_sp)
        skip = pointwise(1, channels, channels, key=k_pw)
        return cls(
            spectral=spectral,
            pointwise=skip,
            activation=activation,
            use_activation=use_activation,
        )

    def __call__(
        self, x: Float[Array, "C N"], transform: BasisTransform
    ) -> Float[Array, "C N"]:
        """Apply the block to one ``(C, N)`` example."""
        h = self.spectral(x, transform) + self.pointwise(x)
        return self.activation(h) if self.use_activation else h


class ManifoldNO(eqx.Module):
    r"""Manifold Neural Operator: lifting → basis blocks → projection.

    With $P$ the lifting, $\mathcal K_\ell$ the `BasisSpectralConv` and $W_\ell$
    the ``1×1`` skip of block $\ell$, $Q_1, Q_2$ the projection and $\sigma$ the
    activation,

    $$
    h_0 = P u, \qquad
    h_\ell = \sigma\big(\mathcal K_\ell(h_{\ell-1};\,T) + W_\ell h_{\ell-1}\big),
    \qquad y = Q_2\,\sigma(Q_1 h_L),
    $$

    where the last block is linear (no $\sigma$), as in `FNO`. Each
    $\mathcal K_\ell$ analyses into the $M$ modes of the shared transform $T$,
    mixes channels with one matrix per mode and synthesises back at the same
    $N$ points (the NORM layer of Chen et al. 2024 in any basis).

    Pseudocode:

    ```text
    h = lifting(u)                              # (H, N)
    for block in blocks:                        # last block linear
        h = block(h, transform)                 # (H, N)
    y = proj2(activation(proj1(h)))             # (C_out, N)
    ```

    Operates on ``(in_channels, N)`` fields at the transform's ``N`` points;
    ``jax.vmap`` over a batch. The transform's tables are read through
    `jax.lax.stop_gradient`, so they never train (their gradient is zero).

    Attributes:
        lifting: Pointwise map ``in_channels -> hidden_channels``.
        blocks: The `ManifoldNOBlock` layers.
        proj1, proj2: Two-layer pointwise projection ``hidden -> proj -> out``.
        transform: The shared, frozen `BasisTransform` (``N`` points in and
            out).
        activation: Pointwise nonlinearity (static).
        in_channels, out_channels, hidden_channels: Static config.

    Examples:
        On the Laplace–Beltrami eigenbasis of a mesh (NORM), here the 162
        vertices of an icosphere with the 9 modes of degree ``l <= 2``:

        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.basis import icosphere
        >>> from geonnax.layers import BasisTransform
        >>> from geonnax.manifold_no import ManifoldNO
        >>> transform = BasisTransform.from_mesh(*icosphere(2), num_basis=9)
        >>> op = ManifoldNO.init(3, 1, transform, key=jr.PRNGKey(0),
        ...                      hidden_channels=16, n_layers=3)
        >>> op(jnp.ones((3, 162))).shape
        (1, 162)
        >>> jax.vmap(op)(jnp.ones((4, 3, 162))).shape   # batched
        (4, 1, 162)

        On a region Slepian basis of a 60° polar cap:

        >>> from geonnax.basis import gauss_legendre_grid, slepian_region_basis
        >>> points, weights = gauss_legendre_grid(12)
        >>> cap = points[:, 2] > 0.5
        >>> basis = slepian_region_basis(5, points[cap], weights[cap])
        >>> transform = BasisTransform.from_slepian(basis, points[cap], weights[cap])
        >>> op = ManifoldNO.init(2, 2, transform, key=jr.PRNGKey(0),
        ...                      hidden_channels=8, n_layers=2)
        >>> op(jnp.ones((2, 96))).shape
        (2, 96)
    """

    lifting: eqx.nn.Conv
    blocks: list[ManifoldNOBlock]
    proj1: eqx.nn.Conv
    proj2: eqx.nn.Conv
    transform: BasisTransform
    activation: Callable[[Array], Array] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        transform: BasisTransform,
        *,
        key: Array,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
    ) -> ManifoldNO:
        """Construct a manifold neural operator on ``transform``'s domain.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            transform: The shared `BasisTransform`; its synthesis must target
                the analysis points (``N_out == N``) so the blocks stack.
            key: PRNG key.
            hidden_channels: Channel width ``H`` of the blocks.
            n_layers: Number of blocks ``L``.
            projection_channels: Hidden width of the projection MLP (defaults
                to ``hidden_channels``).
            activation: Pointwise nonlinearity.

        Returns:
            A `ManifoldNO`.

        Raises:
            ValueError: If ``n_layers < 1`` or the transform's output points
                differ from its input points.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.layers import BasisTransform
            >>> from geonnax.manifold_no import ManifoldNO
            >>> phi, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(0), (50, 6)))
            >>> transform = BasisTransform.from_matrix(phi, jnp.ones(50))
            >>> op = ManifoldNO.init(1, 1, transform, key=jr.PRNGKey(1), n_layers=2)
            >>> len(op.blocks), op.hidden_channels
            (2, 32)
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        n_in = transform.analysis.shape[1]
        n_out = transform.synthesis.shape[0]
        if n_out != n_in:
            raise ValueError(
                "transform must synthesise at its analysis points so the blocks "
                f"stack (N_out == N); got N = {n_in}, N_out = {n_out}."
            )
        projection_channels = projection_channels or hidden_channels

        k_lift, k_p1, k_p2, k_blocks = jax.random.split(key, 4)
        lifting = pointwise(1, in_channels, hidden_channels, key=k_lift)
        bkeys = jax.random.split(k_blocks, n_layers)
        blocks = [
            ManifoldNOBlock.init(
                hidden_channels,
                transform.n_modes,
                key=bkeys[i],
                activation=activation,
                # The last block stays linear before projection, as in FNO.
                use_activation=i < n_layers - 1,
            )
            for i in range(n_layers)
        ]
        proj1 = pointwise(1, hidden_channels, projection_channels, key=k_p1)
        proj2 = pointwise(1, projection_channels, out_channels, key=k_p2)
        return cls(
            lifting=lifting,
            blocks=blocks,
            proj1=proj1,
            proj2=proj2,
            transform=transform,
            activation=activation,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def __call__(self, u: Float[Array, "C_in N"]) -> Float[Array, "C_out N"]:
        """Apply the operator to one ``(C_in, N)`` example."""
        h = self.lifting(u)
        for block in self.blocks:
            h = block(h, self.transform)
        h = self.activation(self.proj1(h))
        return self.proj2(h)


__all__ = ["ManifoldNO", "ManifoldNOBlock"]
