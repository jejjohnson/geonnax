"""Noise Contrastive Prior building blocks (deterministic parts).

Only the input-side stochastic perturbation lives here. The full
``DenseNCP`` and ``NCPNormalOutput`` layers register NumPyro sample
sites for the perturbation weights and the output-side KL factor, so
they stay as Bayesian wrappers in the consuming library.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, Float


class NCPContinuousPerturb(eqx.Module):
    r"""Input perturbation for the Noise Contrastive Prior pattern.

    Adds Gaussian noise scaled by a fixed positive scale to the input:

    .. math::

        \tilde{x} = x + \sigma \epsilon, \qquad
        \epsilon \sim \mathcal{N}(0, I).

    Place before a deterministic network to inject input uncertainty;
    pair with a Bayesian ``DenseNCP`` head for the full NCP
    architecture (Hafner et al., 2019).

    Stochasticity comes from the explicit PRNG ``key`` argument.

    Attributes:
        scale: Perturbation scale :math:`\sigma`.
    """

    scale: float | Float[Array, ""] = 1.0

    def __call__(
        self,
        x: Float[Array, " D"],
        *,
        key: Array,
    ) -> Float[Array, " D"]:
        eps = jax.random.normal(key, x.shape, dtype=x.dtype)
        return x + self.scale * eps


__all__ = ["NCPContinuousPerturb"]
