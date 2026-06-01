"""Monte Carlo dropout for inference-time uncertainty."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class MCDropout(eqx.Module):
    """Always-on dropout for Monte Carlo uncertainty estimation.

    Unlike standard dropout, :class:`MCDropout` stays active at
    inference time — repeated forward passes with different keys
    produce a distribution of outputs whose spread approximates
    predictive uncertainty (Gal & Ghahramani, 2016).

    The stochasticity comes from the explicit PRNG ``key`` argument; no
    sample sites or learnable parameters are involved.

    Attributes:
        rate: Dropout probability in :math:`[0, 1)`.
    """

    rate: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"rate must be in [0, 1), got {self.rate}.")

    def __call__(
        self,
        x: Float[Array, ...],
        *,
        key: Array,
    ) -> Float[Array, ...]:
        """Apply dropout, scaling survivors by ``1 / (1 - rate)``."""
        keep = jax.random.bernoulli(key, 1.0 - self.rate, x.shape)
        return jnp.where(keep, x / (1.0 - self.rate), 0.0)


__all__ = ["MCDropout"]
