"""Deterministic linear core for variational dense layers.

Pure ``equinox.Module`` core: ``y = x @ W + b``. The Bayesian dense
family (reparameterisation, Flipout, hierarchical, NCP, …) in the
consuming library reuses this core as a thin wrapper that swaps the
stored ``W`` / ``b`` for sample sites with different priors.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearCore(eqx.Module):
    r"""Deterministic linear layer ``y = x @ W + b``.

    Shared deterministic core for the variational-dense family. The
    weight matrix has shape ``(in_features, out_features)`` and the
    optional bias has shape ``(out_features,)``. Weights are initialised
    from ``U(-1/sqrt(in_features), 1/sqrt(in_features))`` (LeCun-style
    fan-in scaling); biases are initialised to zero.

    Attributes:
        W: Weight matrix of shape ``(in_features, out_features)``.
        b: Optional bias vector of shape ``(out_features,)`` or ``None``
            when ``bias=False``.
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether a bias term is included.
    """

    W: Float[Array, "in_features out_features"]
    b: Float[Array, " out_features"] | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: Array,
        bias: bool = True,
    ) -> LinearCore:
        """Construct a ``LinearCore`` with LeCun-uniform weights and zero bias."""
        if in_features <= 0:
            raise ValueError(f"in_features must be > 0, got {in_features}.")
        if out_features <= 0:
            raise ValueError(f"out_features must be > 0, got {out_features}.")
        limit = 1.0 / math.sqrt(in_features)
        W = jax.random.uniform(
            key, (in_features, out_features), minval=-limit, maxval=limit
        )
        b: Float[Array, " out_features"] | None
        b = jnp.zeros((out_features,)) if bias else None
        return cls(
            W=W,
            b=b,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        out = einx.dot("... din, din dout -> ... dout", x, self.W)
        if self.b is not None:
            out = out + self.b
        return out


__all__ = ["LinearCore"]
