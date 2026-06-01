"""Normalization layers beyond Equinox's built-ins."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class GlobalResponseNorm(eqx.Module):
    """Global Response Normalization (Woo et al., 2023, "ConvNeXt-V2").

    Computes a per-channel global response (the L2 norm over spatial
    positions), divides it by its channel-mean, then re-scales and -shifts each
    channel with learnable ``gamma``/``beta`` and adds an identity residual.
    Encourages feature diversity across channels at negligible cost and is
    initialised to the identity transform.

    Attributes:
        gamma: Per-channel scale, shape ``(C, *ones)``.
        beta: Per-channel shift, shape ``(C, *ones)``.
        num_spatial_dims: Number of trailing spatial axes.
        eps: Numerical floor.
    """

    gamma: Float[Array, "C *ones"]
    beta: Float[Array, "C *ones"]
    num_spatial_dims: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @classmethod
    def init(
        cls, channels: int, num_spatial_dims: int, *, eps: float = 1e-6
    ) -> GlobalResponseNorm:
        """Construct a GRN initialised to the identity transform."""
        shape = (channels,) + (1,) * num_spatial_dims
        return cls(
            gamma=jnp.zeros(shape),
            beta=jnp.zeros(shape),
            num_spatial_dims=num_spatial_dims,
            eps=eps,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        axes = tuple(range(1, self.num_spatial_dims + 1))
        gx = jnp.sqrt(jnp.sum(x**2, axis=axes, keepdims=True) + self.eps)
        nx = gx / (jnp.mean(gx, axis=0, keepdims=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


__all__ = ["GlobalResponseNorm"]
