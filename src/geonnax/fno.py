"""Fourier Neural Operator (Li et al., 2021) and its building blocks.

The FNO learns a resolution-invariant mapping between function spaces. It lifts
the input to a higher channel width, applies a stack of Fourier blocks — each a
`SpectralConv` (global, spectral) summed with a pointwise
linear skip (local) — then projects back to the output channels. With
factorised spectral weights (CP / Tucker / TT) it becomes the Tensorized FNO
(Kossaifi et al., 2023).

Optional `DomainPadding` extends a non-periodic input before the FFT and
crops afterwards, so the implicit periodicity of the transform does not wrap
boundary information — useful for the bounded, non-periodic fields common in
geoscience.

Like the rest of geonnax, modules act on a single example of shape
``(channels, *spatial)`` (``num_spatial_dims = len(n_modes)``); ``jax.vmap``
over a batch.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers._factorized import Factorization
from geonnax.layers._spectral import SpectralConv


def _pointwise(
    num_spatial_dims: int, in_channels: int, out_channels: int, *, key: Array
) -> eqx.nn.Conv:
    """A ``1x1`` (pointwise) convolution — a per-position channel-wise linear map."""
    return eqx.nn.Conv(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        key=key,
    )


class DomainPadding(eqx.Module):
    """Zero-pad spatial axes by a fraction of their size, then crop back.

    Applied around an FNO so the FFT's implicit periodicity does not couple
    opposite boundaries of a non-periodic field. Padding amounts are computed
    from the input at call time, preserving resolution invariance.

    Attributes:
        padding: Fraction of each spatial extent to append (e.g. ``0.0625``).
        num_spatial_dims: Number of trailing spatial axes.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.fno import DomainPadding
        >>> dp = DomainPadding(padding=0.25, num_spatial_dims=2)
        >>> padded = dp.pad(jnp.ones((3, 8, 8)))   # 8 -> 8 + round(0.25*8)
        >>> padded.shape
        (3, 10, 10)
        >>> dp.unpad(padded, (8, 8)).shape          # crop back
        (3, 8, 8)
    """

    padding: float = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)

    def pad(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *padded"]:
        """Append ``round(padding * size)`` zeros at the end of each spatial axis."""
        pads = [round(self.padding * s) for s in x.shape[1:]]
        config = [(0, 0), *((0, p) for p in pads)]
        return jnp.pad(x, config)

    def unpad(
        self, x: Float[Array, "C *padded"], spatial: tuple[int, ...]
    ) -> Float[Array, "C *spatial"]:
        """Crop back to the original ``spatial`` extents."""
        return x[(slice(None), *(slice(0, s) for s in spatial))]


class FNOBlock(eqx.Module):
    """One Fourier layer: spectral conv + pointwise skip, then an activation.

    Computes ``act(SpectralConv(x) + W·x)`` — a global spectral term plus a
    local ``1x1`` channel mix, the standard FNO layer (Li et al., 2021).

    Attributes:
        spectral: Global spectral mixing (`SpectralConv`).
        pointwise: Local ``1x1`` channel-mixing skip.
        activation: Pointwise nonlinearity (static).
        use_activation: Whether to apply the activation (static; ``False`` for a
            linear final block).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.fno import FNOBlock
        >>> blk = FNOBlock.init(8, (6, 6), key=jr.PRNGKey(0))
        >>> blk(jnp.ones((8, 32, 32))).shape   # width preserved
        (8, 32, 32)
    """

    spectral: SpectralConv
    pointwise: eqx.nn.Conv
    activation: Callable[[Array], Array] = eqx.field(static=True)
    use_activation: bool = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        channels: int,
        n_modes: tuple[int, ...],
        *,
        key: Array,
        factorization: Factorization = "dense",
        rank: float | int = 0.5,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        use_activation: bool = True,
    ) -> FNOBlock:
        """Construct a Fourier block at fixed channel width ``channels``."""
        k_sp, k_pw = jax.random.split(key)
        spectral = SpectralConv.init(
            channels,
            channels,
            n_modes,
            key=k_sp,
            factorization=factorization,
            rank=rank,
        )
        pointwise = _pointwise(len(n_modes), channels, channels, key=k_pw)
        return cls(
            spectral=spectral,
            pointwise=pointwise,
            activation=activation,
            use_activation=use_activation,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        h = self.spectral(x) + self.pointwise(x)
        return self.activation(h) if self.use_activation else h


class FNO(eqx.Module):
    """Fourier Neural Operator: lifting → Fourier blocks → projection.

    Operates on ``(in_channels, *spatial)`` with ``num_spatial_dims =
    len(n_modes)``; ``jax.vmap`` over a batch. Each spatial extent must exceed
    twice its mode count.

    Because the spatial grid is only read at call time, the same operator
    evaluates at any resolution above the mode count — train on one grid,
    apply on another.

    Attributes:
        lifting: Pointwise map ``in_channels -> hidden_channels``.
        blocks: The Fourier layers.
        proj1, proj2: Two-layer pointwise projection ``hidden -> proj -> out``.
        domain_padding: Optional non-periodic padding wrapper.
        activation: Pointwise nonlinearity (static).
        num_spatial_dims, in_channels, out_channels, hidden_channels: Static.

    Examples:
        A 2D operator mapping a 3-channel field to 1 channel, evaluated at two
        resolutions with the *same* parameters:

        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.fno import FNO
        >>> op = FNO.init(3, 1, (8, 8), key=jr.PRNGKey(0),
        ...               hidden_channels=16, n_layers=3)
        >>> op(jnp.ones((3, 32, 32))).shape
        (1, 32, 32)
        >>> op(jnp.ones((3, 64, 64))).shape   # resolution-invariant
        (1, 64, 64)
        >>> jax.vmap(op)(jnp.ones((4, 3, 32, 32))).shape   # batched
        (4, 1, 32, 32)

        Tensorized weights (CP/Tucker/TT) and domain padding for non-periodic
        fields:

        >>> op = FNO.init(2, 2, (6, 6), key=jr.PRNGKey(0), hidden_channels=8,
        ...               n_layers=2, factorization="tucker", rank=0.5,
        ...               domain_padding=0.1)
        >>> op(jnp.ones((2, 40, 40))).shape
        (2, 40, 40)
    """

    lifting: eqx.nn.Conv
    blocks: list[FNOBlock]
    proj1: eqx.nn.Conv
    proj2: eqx.nn.Conv
    domain_padding: DomainPadding | None
    activation: Callable[[Array], Array] = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        *,
        key: Array,
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        factorization: Factorization = "dense",
        rank: float | int = 0.5,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        domain_padding: float = 0.0,
    ) -> FNO:
        """Construct an FNO.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_modes: Retained Fourier modes per spatial axis; its length sets
                the spatial rank.
            key: PRNG key.
            hidden_channels: Channel width of the Fourier blocks.
            n_layers: Number of Fourier blocks.
            projection_channels: Hidden width of the projection MLP (defaults to
                ``hidden_channels``).
            factorization: Spectral-weight parameterisation (``"dense"`` /
                ``"cp"`` / ``"tucker"`` / ``"tt"``).
            rank: Rank for the factorised forms; ignored when dense.
            activation: Pointwise nonlinearity.
            domain_padding: Fraction of spatial padding for non-periodic inputs
                (``0`` disables it).

        Raises:
            ValueError: If ``n_layers < 1`` or ``n_modes`` is empty.
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if len(n_modes) == 0:
            raise ValueError("n_modes must be non-empty.")
        d = len(n_modes)
        projection_channels = projection_channels or hidden_channels

        k_lift, k_p1, k_p2, k_blocks = jax.random.split(key, 4)
        lifting = _pointwise(d, in_channels, hidden_channels, key=k_lift)
        bkeys = jax.random.split(k_blocks, n_layers)
        blocks = [
            FNOBlock.init(
                hidden_channels,
                n_modes,
                key=bkeys[i],
                factorization=factorization,
                rank=rank,
                activation=activation,
                # The last Fourier block stays linear before projection.
                use_activation=i < n_layers - 1,
            )
            for i in range(n_layers)
        ]
        proj1 = _pointwise(d, hidden_channels, projection_channels, key=k_p1)
        proj2 = _pointwise(d, projection_channels, out_channels, key=k_p2)

        dp = (
            DomainPadding(padding=domain_padding, num_spatial_dims=d)
            if domain_padding > 0.0
            else None
        )
        return cls(
            lifting=lifting,
            blocks=blocks,
            proj1=proj1,
            proj2=proj2,
            domain_padding=dp,
            activation=activation,
            num_spatial_dims=d,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        spatial = x.shape[1:]
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x, spatial)
        return x


FourierNeuralOperator = FNO
"""Verbose alias for `FNO`."""


__all__ = [
    "FNO",
    "DomainPadding",
    "FNOBlock",
    "FourierNeuralOperator",
]
