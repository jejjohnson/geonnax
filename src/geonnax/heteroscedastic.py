"""Heteroscedastic output layers (Collier et al., 2021).

Deterministic ``equinox.Module`` cores. The probabilistic wrappers
that promote each weight array to a ``pyrox_param`` site live in the
consuming probabilistic library and reuse the pure helpers exposed
here (``hetero_noisy_logits``, ``HeteroscedasticHead``).

* :class:`HeteroscedasticHead` — shared base with the deterministic
  logit-noise math.
* :class:`MCSoftmaxDenseFA` — multi-class output with input-dependent
  low-rank-plus-diagonal logit noise, MC-averaged softmax probabilities.
* :class:`MCSigmoidDenseFA` — same noise model, sigmoid output for
  multi-label / binary classification.

Both share the same heteroscedastic logit-noise model — given an
input-dependent low-rank factor :math:`V(x) \\in \\mathbb{R}^{C \\times r}`
and diagonal :math:`\\sigma(x) \\in \\mathbb{R}^C`,

.. math::

    \\eta(x) = W_\\mu x + b_\\mu + \\epsilon, \\qquad
    \\Sigma(x) = V(x) V(x)^\\top + \\mathrm{diag}\\!\\bigl(\\sigma^2(x)\\bigr),
    \\;\\; \\epsilon \\sim \\mathcal{N}(0, \\Sigma(x)).

Predictions average a small number of Monte Carlo softmax / sigmoid
samples.
"""

from __future__ import annotations

import math
from typing import Self

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def _glorot_uniform(
    key: PRNGKeyArray, in_features: int, out_features: int
) -> Float[Array, "D_in D_out"]:
    lim = math.sqrt(6.0 / (in_features + out_features))
    return jr.uniform(
        key,
        (in_features, out_features),
        minval=-lim,
        maxval=lim,
    )


def hetero_noisy_logits(
    layer: HeteroscedasticHead,
    x: Float[Array, " D_in"],
    *,
    key: PRNGKeyArray,
) -> Float[Array, "S C"]:
    """Sample ``num_mc_samples`` heteroscedastic logits for one example.

    Computes mean logits, low-rank factor, and diagonal scale via three
    deterministic linear maps, then draws Monte Carlo logit
    perturbations using the supplied PRNG ``key``.

    Args:
        layer: A :class:`HeteroscedasticHead` (or subclass) carrying
            the weight / bias arrays.
        x: Input feature vector of shape ``(D_in,)``.
        key: PRNG key used to draw the Gaussian noise samples.

    Returns:
        Tensor of shape ``(S, C)`` of MC logit samples; the ``S``
        sample axis is intrinsic to the head, the data axis was stripped
        (``vmap`` to batch).

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from geonnax.heteroscedastic import MCSoftmaxDenseFA
        >>> layer = MCSoftmaxDenseFA.init(
        ...     in_features=4, num_classes=3, rank=2, key=jr.PRNGKey(0),
        ...     num_mc_samples=7,
        ... )
        >>> logits = hetero_noisy_logits(layer, jnp.ones(4), key=jr.PRNGKey(2))
        >>> logits.shape  # (S, C) = (num_mc_samples, num_classes)
        (7, 3)
    """
    C = layer.num_classes
    r = layer.rank
    S = layer.num_mc_samples

    mu = einx.dot("d, d c -> c", x, layer.W_loc) + layer.b_loc  # (C,)

    # Low-rank factor: project to (C·r,) then split the trailing axis.
    raw_scale = einx.dot("d, d k -> k", x, layer.W_scale) + layer.b_scale
    V = einx.id("(c r) -> c r", raw_scale, r=r)

    sigma = jnp.exp(einx.dot("d, d c -> c", x, layer.W_diag) + layer.b_diag)  # (C,)

    kz, ku = jr.split(key)
    z = jr.normal(kz, (S, r), dtype=x.dtype)
    u = jr.normal(ku, (S, C), dtype=x.dtype)
    # Low-rank + diagonal noise: V z contracts the rank axis r, the diagonal
    # term broadcasts σ over the S sample axis. Both land at (S, C).
    eps = einx.dot("c r, s r -> s c", V, z) + einx.multiply("c, s c -> s c", sigma, u)
    return einx.add("c, s c -> s c", mu, eps)


class HeteroscedasticHead(eqx.Module):
    """Shared state and init for the FA-noise heteroscedastic layers.

    Stores the six deterministic linear-factor arrays
    (``W_loc``/``b_loc`` for the mean logits, ``W_scale``/``b_scale``
    for the low-rank covariance factor, ``W_diag``/``b_diag`` for the
    log-diagonal scale) as real :mod:`equinox` array fields. Subclasses
    override ``__call__`` to wrap :func:`hetero_noisy_logits` with the
    appropriate link function (softmax / sigmoid).

    Attributes:
        W_loc: Mean-logit weight matrix, shape ``(D_in, C)``.
        b_loc: Mean-logit bias, shape ``(C,)``.
        W_scale: Low-rank factor weight matrix, shape ``(D_in, C * r)``.
        b_scale: Low-rank factor bias, shape ``(C * r,)``.
        W_diag: Diagonal log-scale weight matrix, shape ``(D_in, C)``.
        b_diag: Diagonal log-scale bias, shape ``(C,)``.
        in_features: Input dimension :math:`D_\\mathrm{in}`.
        num_classes: Number of classes :math:`C`.
        rank: Rank :math:`r` of the low-rank factor :math:`V(x)`.
        num_mc_samples: Number of MC samples :math:`S` per forward call.
        diag_init_bias: Initial value for the diagonal-scale bias
            ``b_diag`` (a small negative number keeps initial noise
            small).

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> head = HeteroscedasticHead.init(
        ...     in_features=4, num_classes=3, rank=2, key=jr.PRNGKey(0),
        ... )
        >>> out = head(jnp.ones(4), key=jr.PRNGKey(1))  # MC mean of raw logits
        >>> out.shape
        (3,)
    """

    W_loc: Float[Array, "D_in C"]
    b_loc: Float[Array, " C"]
    W_scale: Float[Array, "D_in Cr"]
    b_scale: Float[Array, " Cr"]
    W_diag: Float[Array, "D_in C"]
    b_diag: Float[Array, " C"]
    in_features: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    num_mc_samples: int = eqx.field(static=True, default=10)
    diag_init_bias: float = eqx.field(static=True, default=-3.0)

    @classmethod
    def init(
        cls,
        in_features: int,
        num_classes: int,
        rank: int,
        *,
        key: PRNGKeyArray,
        num_mc_samples: int = 10,
        diag_init_bias: float = -3.0,
        scale_init_factor: float = 0.1,
    ) -> Self:
        """Construct the layer with Glorot-init linear factors."""
        if in_features <= 0 or num_classes <= 0 or rank <= 0:
            raise ValueError(
                "in_features, num_classes, rank must all be > 0; "
                f"got {in_features=}, {num_classes=}, {rank=}."
            )
        if num_mc_samples <= 0:
            raise ValueError(f"num_mc_samples must be > 0; got {num_mc_samples}.")
        kl, ks = jr.split(key)
        W_loc = _glorot_uniform(kl, in_features, num_classes)
        # Small init for the low-rank factor so it does not dominate the
        # mean logits before training.
        W_scale = scale_init_factor * _glorot_uniform(
            ks, in_features, num_classes * rank
        )
        W_diag = jnp.zeros((in_features, num_classes))
        b_loc = jnp.zeros(num_classes)
        b_scale = jnp.zeros(num_classes * rank)
        b_diag = jnp.full((num_classes,), float(diag_init_bias))
        return cls(
            W_loc=W_loc,
            b_loc=b_loc,
            W_scale=W_scale,
            b_scale=b_scale,
            W_diag=W_diag,
            b_diag=b_diag,
            in_features=in_features,
            num_classes=num_classes,
            rank=rank,
            num_mc_samples=num_mc_samples,
            diag_init_bias=diag_init_bias,
        )

    def __call__(
        self,
        x: Float[Array, " D_in"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, " C"]:
        """Default forward returns the MC mean of the raw logits.

        Subclasses override this to apply a link function (softmax /
        sigmoid) before averaging.
        """
        logits = hetero_noisy_logits(self, x, key=key)
        return jnp.mean(logits, axis=0)


class MCSoftmaxDenseFA(HeteroscedasticHead):
    r"""Heteroscedastic multi-class output layer (FA noise + softmax).

    Implements Collier et al. (2021): the logit covariance is
    input-dependent low-rank-plus-diagonal,

    .. math::

        \eta(x) = W_\mu x + b_\mu + \epsilon, \qquad
        \Sigma(x) = V(x) V(x)^\top + \operatorname{diag}\!\bigl(\sigma^2(x)\bigr),
        \;\; \epsilon \sim \mathcal{N}(0, \Sigma(x)),

    where :math:`V(x) = \mathrm{reshape}(W_V x + b_V, [C, r])` and
    :math:`\sigma(x) = \exp(W_\sigma x + b_\sigma)`. Output is the
    Monte Carlo average of softmaxed perturbed logits

    .. math::

        \hat{p}(y = k \mid x) \approx
        \frac{1}{S}\sum_{s=1}^{S}
        \mathrm{softmax}_k\!\bigl(\eta(x) + \epsilon_s\bigr).

    All linear factors are real :mod:`equinox` array fields — the
    layer is heteroscedastic but not Bayesian over its weights. Use it
    as a drop-in head for classification when label noise is known to
    be input-dependent (label disagreement, fine-grained categories).

    The consuming probabilistic library can promote each array to a
    parameter site without changing the forward math; geonnax exposes
    :func:`hetero_noisy_logits` as the reusable pure helper.

    Attributes:
        in_features: Input dimension :math:`D_\mathrm{in}`.
        num_classes: Number of classes :math:`C`.
        rank: Rank :math:`r` of the low-rank factor :math:`V(x)`.
        num_mc_samples: Number of MC softmax samples :math:`S` per
            forward call.
        diag_init_bias: Initial value for the diagonal-scale bias
            ``b_diag`` (a small negative number keeps initial noise
            small).

    Example:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> layer = MCSoftmaxDenseFA.init(
        ...     in_features=4, num_classes=3, rank=2, key=jr.PRNGKey(0),
        ... )
        >>> x = jnp.ones(4)
        >>> probs = layer(x, key=jr.PRNGKey(1))
        >>> probs.shape
        (3,)
        >>> bool(jnp.allclose(probs.sum(axis=-1), 1.0))
        True

    References:
        Collier, M., Mustafa, B., Kokiopoulou, E., Jenatton, R., &
        Berent, J. (2021). *Correlated Input-Dependent Label Noise in
        Large-Scale Image Classification.* CVPR.
    """

    def __call__(
        self,
        x: Float[Array, " D_in"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, " C"]:
        logits = hetero_noisy_logits(self, x, key=key)
        return jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)


class MCSigmoidDenseFA(HeteroscedasticHead):
    r"""Heteroscedastic multi-label output layer (FA noise + sigmoid).

    Identical low-rank-plus-diagonal logit-noise model as
    :class:`MCSoftmaxDenseFA`, but the per-class outputs are
    independent Bernoullis — final probabilities are the MC average of
    *element-wise* sigmoids, not a softmax. Use this for multi-label
    classification or independent binary heads.

    .. math::

        \hat{p}(y_k = 1 \mid x) \approx
        \frac{1}{S}\sum_{s=1}^{S}
        \sigma\!\bigl(\eta(x) + \epsilon_s\bigr)_k.

    See :class:`MCSoftmaxDenseFA` for the noise model, init API, and
    references.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> layer = MCSigmoidDenseFA.init(
        ...     in_features=4, num_classes=3, rank=2, key=jr.PRNGKey(0),
        ... )
        >>> probs = layer(jnp.ones(4), key=jr.PRNGKey(1))
        >>> probs.shape
        (3,)
        >>> bool(jnp.all((probs >= 0.0) & (probs <= 1.0)))  # per-class Bernoulli
        True
    """

    def __call__(
        self,
        x: Float[Array, " D_in"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, " C"]:
        logits = hetero_noisy_logits(self, x, key=key)
        return jnp.mean(jax.nn.sigmoid(logits), axis=0)


__all__ = [
    "HeteroscedasticHead",
    "MCSigmoidDenseFA",
    "MCSoftmaxDenseFA",
    "hetero_noisy_logits",
]
