"""Wavelet Neural Operator (Tripura & Chakraborty, 2023) and its blocks.

The WNO is the FNO with its Fourier transform replaced by a discrete wavelet
transform: it lifts the input to a hidden width, applies a stack of wavelet
blocks — each a `WaveletConv` (global, multi-resolution)
summed with a pointwise linear skip (local) — then projects to the output
channels. Mixing in the wavelet domain captures spatially-localised,
multi-scale structure that the FNO's global Fourier modes smear out, which
suits sharp or non-stationary fields.

Like the rest of geonnax, modules act on a single example of shape
``(channels, *spatial)`` (``num_spatial_dims`` spatial axes); ``jax.vmap`` over
a batch. Each spatial extent must be divisible by ``2 ** level``.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float

from geonnax.layers._wavelet import Wavelet, WaveletConv


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


class WNOBlock(eqx.Module):
    """One wavelet layer: wavelet conv + pointwise skip, then an activation.

    Computes ``act(WaveletConv(x) + W·x)`` — a global multi-resolution term plus
    a local ``1x1`` channel mix, mirroring the FNO's Fourier block.

    Attributes:
        wavelet_conv: Global wavelet-domain mixing (`WaveletConv`).
        pointwise: Local ``1x1`` channel-mixing skip.
        activation: Pointwise nonlinearity (static).
        use_activation: Whether to apply the activation (static; ``False`` for a
            linear final block).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.wno import WNOBlock
        >>> blk = WNOBlock.init(8, 2, key=jr.PRNGKey(0), wavelet="db4", level=2)
        >>> blk(jnp.ones((8, 32, 32))).shape
        (8, 32, 32)
    """

    wavelet_conv: WaveletConv
    pointwise: eqx.nn.Conv
    activation: Callable[[Array], Array] = eqx.field(static=True)
    use_activation: bool = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        wavelet: Wavelet = "db4",
        level: int = 1,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        use_activation: bool = True,
    ) -> WNOBlock:
        """Construct a wavelet block at fixed channel width ``channels``."""
        k_w, k_pw = jax.random.split(key)
        wavelet_conv = WaveletConv.init(
            channels,
            channels,
            num_spatial_dims,
            key=k_w,
            wavelet=wavelet,
            level=level,
        )
        pointwise = _pointwise(num_spatial_dims, channels, channels, key=k_pw)
        return cls(
            wavelet_conv=wavelet_conv,
            pointwise=pointwise,
            activation=activation,
            use_activation=use_activation,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        h = self.wavelet_conv(x) + self.pointwise(x)
        return self.activation(h) if self.use_activation else h


class WNO(eqx.Module):
    """Wavelet Neural Operator: lifting → wavelet blocks → projection.

    Operates on ``(in_channels, *spatial)`` with ``num_spatial_dims`` spatial
    axes; ``jax.vmap`` over a batch. Each spatial extent must be divisible by
    ``2 ** level``.

    Attributes:
        lifting: Pointwise map ``in_channels -> hidden_channels``.
        blocks: The wavelet layers.
        proj1, proj2: Two-layer pointwise projection ``hidden -> proj -> out``.
        activation: Pointwise nonlinearity (static).
        num_spatial_dims, in_channels, out_channels, hidden_channels: Static.

    Examples:
        A 2D wavelet operator mapping a 3-channel field to 1 channel (each axis
        divisible by ``2 ** level``):

        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.wno import WNO
        >>> op = WNO.init(3, 1, num_spatial_dims=2, key=jr.PRNGKey(0),
        ...               hidden_channels=16, n_layers=3, wavelet="db4", level=2)
        >>> op(jnp.ones((3, 64, 64))).shape
        (1, 64, 64)
        >>> jax.vmap(op)(jnp.ones((4, 3, 64, 64))).shape
        (4, 1, 64, 64)
    """

    lifting: eqx.nn.Conv
    blocks: list[WNOBlock]
    proj1: eqx.nn.Conv
    proj2: eqx.nn.Conv
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
        num_spatial_dims: int,
        *,
        key: Array,
        hidden_channels: int = 32,
        n_layers: int = 4,
        wavelet: Wavelet = "db4",
        level: int = 1,
        projection_channels: int | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
    ) -> WNO:
        """Construct a WNO.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_spatial_dims: Number of spatial axes (1, 2, or 3).
            key: PRNG key.
            hidden_channels: Channel width of the wavelet blocks.
            n_layers: Number of wavelet blocks.
            wavelet: Wavelet name — ``"haar"``, Daubechies ``"db2"``..``"db5"``,
                Symlets ``"sym4"``..``"sym6"``, or Coiflets ``"coif1"``/``"coif2"``.
            level: Number of DWT levels per block.
            projection_channels: Hidden width of the projection MLP (defaults to
                ``hidden_channels``).
            activation: Pointwise nonlinearity.

        Raises:
            ValueError: If ``n_layers < 1`` or ``num_spatial_dims < 1``.
        """
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims must be >= 1, got {num_spatial_dims}.")
        d = num_spatial_dims
        projection_channels = projection_channels or hidden_channels

        k_lift, k_p1, k_p2, k_blocks = jax.random.split(key, 4)
        lifting = _pointwise(d, in_channels, hidden_channels, key=k_lift)
        bkeys = jax.random.split(k_blocks, n_layers)
        blocks = [
            WNOBlock.init(
                hidden_channels,
                d,
                key=bkeys[i],
                wavelet=wavelet,
                level=level,
                activation=activation,
                # The last wavelet block stays linear before projection.
                use_activation=i < n_layers - 1,
            )
            for i in range(n_layers)
        ]
        proj1 = _pointwise(d, hidden_channels, projection_channels, key=k_p1)
        proj2 = _pointwise(d, projection_channels, out_channels, key=k_p2)
        return cls(
            lifting=lifting,
            blocks=blocks,
            proj1=proj1,
            proj2=proj2,
            activation=activation,
            num_spatial_dims=d,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation(self.proj1(x))
        x = self.proj2(x)
        return x


WaveletNeuralOperator = WNO
"""Verbose alias for `WNO`."""


__all__ = ["WNO", "WNOBlock", "WaveletNeuralOperator"]
