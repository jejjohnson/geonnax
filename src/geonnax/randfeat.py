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
    x: Float[Array, " D_in"],
) -> Float[Array, " D_rff"]:
    r"""Shared RFF feature map: ``sqrt(1/D) [cos(xW/l), sin(xW/l)]``.

    Concatenating cos and sin doubles the width, so the output has
    ``2 * n_features`` entries.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.randfeat import rff_forward
        >>> W = jr.normal(jr.PRNGKey(0), (3, 4))
        >>> rff_forward(W, 1.0, 4, jnp.ones(3)).shape
        (8,)
    """
    # z = xW / ℓ ; (D_in,)·(D_in, n_features) -> (n_features,)
    z = einx.dot("din, din f -> f", x, W) / lengthscale
    scale = jnp.sqrt(1.0 / n_features)
    # [cos z, sin z] : (n_features,) ++ (n_features,) -> (2 * n_features,)
    return scale * jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)


def rff_cosine_forward(
    W: Float[Array, "D_in n_features"],
    b: Float[Array, " n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, " D_in"],
) -> Float[Array, " n_features"]:
    r"""Shared single-cosine RFF feature map: ``sqrt(2/D) cos(xW/l + b)``.

    Single cosine map (no sin), so the output width equals
    ``n_features``.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.randfeat import rff_cosine_forward
        >>> W = jr.normal(jr.PRNGKey(0), (3, 4))
        >>> b = jnp.zeros(4)
        >>> rff_cosine_forward(W, b, 1.0, 4, jnp.ones(3)).shape
        (4,)
    """
    # proj = xW / ℓ + b ; (D_in,)·(D_in, n_features) -> (n_features,)
    proj = einx.dot("din, din f -> f", x, W) / lengthscale + b
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

    The result has shape ``(D, n_blocks * D)`` — each Haar block
    contributes ``D`` frequency columns.

    Examples:
        >>> import jax.random as jr
        >>> from geonnax.randfeat import orthogonal_blocks
        >>> W = orthogonal_blocks(3, 2, key=jr.PRNGKey(0))
        >>> W.shape
        (3, 6)
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

    The feature map is the shared RFF map
    :math:`\phi(x) = \sqrt{1/D}\,[\cos(W^\top x/\ell),\,\sin(W^\top x/\ell)]`,
    so calling the module on a ``(in_features,)`` vector yields a
    ``(2 * n_features,)`` feature vector.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.randfeat import OrthogonalRandomFeatures
        >>> orf = OrthogonalRandomFeatures.init(4, 8, key=jr.PRNGKey(0))
        >>> orf(jnp.ones(4)).shape
        (16,)
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
        """Build the frozen ORF frequency matrix and wrap it in the module.

        ``n_features`` must be divisible by ``in_features`` so the
        Haar-orthogonal blocks tile cleanly.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.randfeat import OrthogonalRandomFeatures
            >>> orf = OrthogonalRandomFeatures.init(4, 8, key=jr.PRNGKey(0))
            >>> orf.W.shape
            (4, 8)
        """
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if in_features <= 0 or n_features <= 0:
            raise ValueError(
                "in_features and n_features must be > 0; got "
                f"in_features={in_features}, n_features={n_features}."
            )
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

    def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, " D_rff"]:
        r"""Map ``x`` to its ORF feature vector ``(2 * n_features,)``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.randfeat import OrthogonalRandomFeatures
            >>> orf = OrthogonalRandomFeatures.init(2, 4, key=jr.PRNGKey(0))
            >>> orf(jnp.ones(2)).shape
            (8,)
        """
        # (D_in,) -> (2 * n_features,)
        return rff_forward(self.W, self.lengthscale, self.n_features, x)


__all__ = [
    "OrthogonalRandomFeatures",
    "orthogonal_blocks",
    "rff_cosine_forward",
    "rff_forward",
]
