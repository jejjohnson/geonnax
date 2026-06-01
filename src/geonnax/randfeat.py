"""Random Fourier feature primitives.

Shared deterministic helpers (``rff_forward``, ``rff_cosine_forward``,
``orthogonal_blocks``) and the lone fully-deterministic feature map
(:class:`OrthogonalRandomFeatures`). The Bayesian feature-map family
(``RBF/Matern/Laplace {Fourier,Cosine}Features`` etc.) lives in the
consuming library as thin sample-wrappers built on top of these
helpers.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def rff_forward(
    W: Float[Array, "D_in n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch D_rff"]:
    """Shared RFF feature map: ``sqrt(1/D) [cos(xW/l), sin(xW/l)]``."""
    z = einx.dot("... din, din f -> ... f", x, W) / lengthscale
    scale = jnp.sqrt(1.0 / n_features)
    return scale * jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)


def rff_cosine_forward(
    W: Float[Array, "D_in n_features"],
    b: Float[Array, " n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch n_features"]:
    """Shared single-cosine RFF feature map: ``sqrt(2/D) cos(xW/l + b)``."""
    proj = einx.dot("... din, din f -> ... f", x, W) / lengthscale + b
    return jnp.sqrt(2.0 / n_features) * jnp.cos(proj)


def orthogonal_blocks(
    in_features: int,
    n_blocks: int,
    *,
    key: jax.Array,
) -> Float[Array, "D_in D_orf"]:
    """Build the ORF frequency matrix as ``[Q_1 S_1, ..., Q_K S_K]``.

    Each block is a Haar-orthogonal ``D x D`` matrix with its *columns*
    scaled by independent chi-distributed magnitudes. The RFF forward uses
    ``z = x @ W`` so columns of ``W`` carry the per-feature frequencies:
    scaling columns (not rows) preserves the ORF construction where each
    frequency vector is an orthogonal unit direction times its own chi
    magnitude.
    """
    D = in_features
    blocks: list[Float[Array, "D_in D_in"]] = []
    for _ in range(n_blocks):
        key, k_qr, k_chi = jax.random.split(key, 3)
        G = jax.random.normal(k_qr, (D, D))
        Q, _ = jnp.linalg.qr(G)
        # Chi-distributed scale per column (sqrt of chi-squared with df=D).
        chi = jnp.sqrt(jnp.sum(jax.random.normal(k_chi, (D, D)) ** 2, axis=-1))  # (D,)
        blocks.append(Q * chi[None, :])
    return jnp.concatenate(blocks, axis=-1)  # (D, n_blocks * D)


class OrthogonalRandomFeatures(eqx.Module):
    r"""Orthogonal Random Features (Yu et al., 2016) — variance-reduced RFF.

    Frequencies are drawn from blocks of Haar-orthogonal matrices scaled by
    independent chi-distributed magnitudes, giving the same RBF kernel
    approximation as plain ``RBFFourierFeatures`` *in expectation* but
    with provably lower variance for finite ``n_features``.

    Frozen at construction time — no priors, no SVI on ``W``. The frequency
    matrix is built once from a ``key`` and stored as a static array.

    Attributes:
        in_features: Input dimension :math:`D`.
        n_features: Number of feature pairs. Must satisfy
            ``n_features % in_features == 0`` so that ORF blocks tile cleanly.
        lengthscale: Fixed kernel lengthscale (no prior; pass a value).
        W: Pre-built frequency matrix of shape ``(in_features, n_features)``.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    lengthscale: Float[Array, ""]
    W: Float[Array, "D_in D_orf"]

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        key: jax.Array,
        lengthscale: float = 1.0,
    ) -> OrthogonalRandomFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if n_features % in_features != 0:
            raise ValueError(
                f"n_features ({n_features}) must be divisible by in_features "
                f"({in_features}) so ORF blocks tile cleanly."
            )
        n_blocks = n_features // in_features
        W = orthogonal_blocks(in_features, n_blocks, key=key)
        return cls(
            in_features=in_features,
            n_features=n_features,
            lengthscale=jnp.asarray(lengthscale),
            W=W,
        )

    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        return rff_forward(self.W, self.lengthscale, self.n_features, x)


__all__ = [
    "OrthogonalRandomFeatures",
    "orthogonal_blocks",
    "rff_cosine_forward",
    "rff_forward",
]
