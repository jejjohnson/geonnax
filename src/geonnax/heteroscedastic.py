"""Heteroscedastic output layers (Collier et al., 2021).

Deterministic ``equinox.Module`` cores. The probabilistic wrappers
that promote each weight array to a ``pyrox_param`` site live in the
consuming probabilistic library and reuse the pure helpers exposed
here (``hetero_noisy_logits``, ``HeteroscedasticHead``).

* `HeteroscedasticHead` — shared base with the deterministic
  logit-noise math.
* `MCSoftmaxDenseFA` — multi-class output with input-dependent
  low-rank-plus-diagonal logit noise, MC-averaged softmax probabilities.
* `MCSigmoidDenseFA` — same noise model, sigmoid output for
  multi-label / binary classification.
* `GaussianRegressionHead` — regression head returning a mean and an
  input-dependent, clamped log standard deviation.

Both share the same heteroscedastic logit-noise model — given an
input-dependent low-rank factor $V(x) \\in \\mathbb{R}^{C \\times r}$
and diagonal $\\sigma(x) \\in \\mathbb{R}^C$,

$$
\\eta(x) = W_\\mu x + b_\\mu + \\epsilon, \\qquad
\\Sigma(x) = V(x) V(x)^\\top + \\mathrm{diag}\\!\\bigl(\\sigma^2(x)\\bigr),
\\;\\; \\epsilon \\sim \\mathcal{N}(0, \\Sigma(x)).
$$


Predictions average a small number of Monte Carlo softmax / sigmoid
samples. The regression head needs no sampling: it returns the two
parameters $(\\mu(x), \\log\\sigma(x))$ of a Gaussian, and the likelihood
is built downstream.
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
        layer: A `HeteroscedasticHead` (or subclass) carrying
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
    log-diagonal scale) as real `equinox` array fields. Subclasses
    override ``__call__`` to wrap `hetero_noisy_logits` with the
    appropriate link function (softmax / sigmoid).

    Attributes:
        W_loc: Mean-logit weight matrix, shape ``(D_in, C)``.
        b_loc: Mean-logit bias, shape ``(C,)``.
        W_scale: Low-rank factor weight matrix, shape ``(D_in, C * r)``.
        b_scale: Low-rank factor bias, shape ``(C * r,)``.
        W_diag: Diagonal log-scale weight matrix, shape ``(D_in, C)``.
        b_diag: Diagonal log-scale bias, shape ``(C,)``.
        in_features: Input dimension $D_\\mathrm{in}$.
        num_classes: Number of classes $C$.
        rank: Rank $r$ of the low-rank factor $V(x)$.
        num_mc_samples: Number of MC samples $S$ per forward call.
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

    $$
    \eta(x) = W_\mu x + b_\mu + \epsilon, \qquad
    \Sigma(x) = V(x) V(x)^\top + \operatorname{diag}\!\bigl(\sigma^2(x)\bigr),
    \;\; \epsilon \sim \mathcal{N}(0, \Sigma(x)),
    $$


    where $V(x) = \mathrm{reshape}(W_V x + b_V, [C, r])$ and
    $\sigma(x) = \exp(W_\sigma x + b_\sigma)$. Output is the
    Monte Carlo average of softmaxed perturbed logits

    $$
    \hat{p}(y = k \mid x) \approx
    \frac{1}{S}\sum_{s=1}^{S}
    \mathrm{softmax}_k\!\bigl(\eta(x) + \epsilon_s\bigr).
    $$


    All linear factors are real `equinox` array fields — the
    layer is heteroscedastic but not Bayesian over its weights. Use it
    as a drop-in head for classification when label noise is known to
    be input-dependent (label disagreement, fine-grained categories).

    The consuming probabilistic library can promote each array to a
    parameter site without changing the forward math; geonnax exposes
    `hetero_noisy_logits` as the reusable pure helper.

    Attributes:
        in_features: Input dimension $D_\mathrm{in}$.
        num_classes: Number of classes $C$.
        rank: Rank $r$ of the low-rank factor $V(x)$.
        num_mc_samples: Number of MC softmax samples $S$ per
            forward call.
        diag_init_bias: Initial value for the diagonal-scale bias
            ``b_diag`` (a small negative number keeps initial noise
            small).

    Examples:
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
    `MCSoftmaxDenseFA`, but the per-class outputs are
    independent Bernoullis — final probabilities are the MC average of
    *element-wise* sigmoids, not a softmax. Use this for multi-label
    classification or independent binary heads.

    $$
    \hat{p}(y_k = 1 \mid x) \approx
    \frac{1}{S}\sum_{s=1}^{S}
    \sigma\!\bigl(\eta(x) + \epsilon_s\bigr)_k.
    $$


    See `MCSoftmaxDenseFA` for the noise model, init API, and
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


# Clamp range of `GaussianRegressionHead`'s log standard deviation.
_LOG_SIGMA_MIN = -7.0
_LOG_SIGMA_MAX = 3.0


class GaussianRegressionHead(eqx.Module):
    r"""Heteroscedastic Gaussian regression head (mean plus log-scale).

    Maps a feature vector $h \in \mathbb{R}^{D_\mathrm{in}}$ to the
    parameters of an input-dependent Gaussian observation model,

    $$
    y \sim \mathcal{N}\bigl(\mu(h),\ \sigma(h)^2\bigr), \qquad
    \mu = W_\mu^\top h + b_\mu, \qquad
    \log\sigma = \operatorname{clip}\bigl(W_s^\top h + b_s,\
    -7,\ 3\bigr),
    $$

    with one output per target, $\mu, \log\sigma \in \mathbb{R}^K$
    (Nix & Weigend, 1994). The clamp keeps $\sigma \in
    [e^{-7}, e^{3}] \approx [9 \times 10^{-4}, 20]$, so the Gaussian
    negative log-likelihood cannot diverge by collapsing $\sigma$ onto a
    fitted point. The hard clip is deliberate: it pins $\sigma$ to exact,
    documented bounds, at the cost of a zero gradient through
    $\log\sigma$ while the pre-activation is outside the range (a
    saturated output stops learning until the mean branch moves the
    residuals back). A soft clamp, e.g.
    $\log\sigma_{\min} + (\log\sigma_{\max} - \log\sigma_{\min})
    \,\mathrm{sigmoid}(\cdot)$ or a softplus floor, keeps a gradient
    everywhere but no longer reproduces $\sigma_\mathrm{init}$ exactly
    from $b_s$; wrap the head if you need it.

    **Initialisation.** $W_\mu$ is Glorot-uniform and $b_\mu = 0$;
    $W_s = \texttt{scale\_init\_factor} \cdot \text{Glorot}$ (zero by
    default) and $b_s = \log\sigma_\mathrm{init}$, so the head starts
    homoscedastic at $\sigma_\mathrm{init}$ and the variance branch only
    departs from it as the data ask. Nix & Weigend (1994) train in two
    phases, the mean first and the variance after; a flat variance at init
    approximates that schedule inside a single joint fit. Seitzer et al.
    (2022) show that the $1/\sigma^2$ weighting of the Gaussian NLL lets
    the mean underfit the regions the variance branch has already marked
    as noisy (they propose the $\beta$-NLL); starting homoscedastic keeps
    that weighting uniform early on. The small-scale init follows the
    ``scale_init_factor`` knob of `HeteroscedasticHead`, with a different
    default: that head uses ``0.1``, this one ``0.0`` so the start is
    exactly homoscedastic. A zero $W_s$ still receives gradient
    ($\partial \log\sigma / \partial W_s = h$).

    The head returns parameters, not a distribution: build the likelihood
    downstream. With a known per-observation error $\sigma_i$, combine the
    two as $\sigma_i^2 + \sigma(h)^2$. A Bayesian wrapper samples the four
    array fields ``W_loc``, ``b_loc``, ``W_log_scale`` and ``b_log_scale``.

    Attributes:
        W_loc: Mean weight matrix $W_\mu$, shape ``(D_in, K)``.
        b_loc: Mean bias $b_\mu$, shape ``(K,)``.
        W_log_scale: Log-scale weight matrix $W_s$, shape ``(D_in, K)``.
        b_log_scale: Log-scale bias $b_s$, shape ``(K,)``.
        in_features: Input dimension $D_\mathrm{in}$.
        out_features: Number of targets $K$.
        log_sigma_min: Lower clamp of $\log\sigma$ (``-7.0``).
        log_sigma_max: Upper clamp of $\log\sigma$ (``3.0``).

    Examples:
        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> head = GaussianRegressionHead.init(
        ...     in_features=8, out_features=2, sigma_init=0.5, key=jr.PRNGKey(0)
        ... )
        >>> mu, log_sigma = head(jnp.ones(8))
        >>> mu.shape, log_sigma.shape
        ((2,), (2,))
        >>> [round(float(s), 3) for s in jnp.exp(log_sigma)]  # homoscedastic at init
        [0.5, 0.5]
        >>> mu_b, log_sigma_b = jax.vmap(head)(jnp.ones((5, 8)))
        >>> mu_b.shape
        (5, 2)

    References:
        Nix, D. A., & Weigend, A. S. (1994). *Estimating the mean and
        variance of the target probability distribution.* ICNN.

        Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). *On the
        Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic
        Neural Networks.* ICLR.
    """

    W_loc: Float[Array, "D_in K"]
    b_loc: Float[Array, " K"]
    W_log_scale: Float[Array, "D_in K"]
    b_log_scale: Float[Array, " K"]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    log_sigma_min: float = eqx.field(static=True, default=_LOG_SIGMA_MIN)
    log_sigma_max: float = eqx.field(static=True, default=_LOG_SIGMA_MAX)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int = 1,
        *,
        key: PRNGKeyArray,
        sigma_init: float = 1.0,
        scale_init_factor: float = 0.0,
    ) -> Self:
        r"""Build the head with a Glorot mean branch and a flat log-scale.

        Args:
            in_features: Input dimension $D_\mathrm{in}$; must be > 0.
            out_features: Number of targets $K$; must be > 0.
            key: PRNG key, split between the mean and log-scale weights.
            sigma_init: Initial standard deviation $\sigma_\mathrm{init}$;
                $\log\sigma_\mathrm{init}$ must lie strictly inside
                $(-7, 3)$, so the head does not start on the clip.
            scale_init_factor: Multiplier on the Glorot draw for $W_s$;
                ``0.0`` (the default) makes the initial head exactly
                homoscedastic. Must be >= 0.

        Returns:
            The initialised head.

        Raises:
            ValueError: If a dimension is non-positive, ``sigma_init`` is not
                positive or not strictly inside $(e^{-7}, e^{3})$, or
                ``scale_init_factor`` is negative.

        Examples:
            >>> import jax.random as jr
            >>> head = GaussianRegressionHead.init(4, 3, key=jr.PRNGKey(0))
            >>> head.W_loc.shape, head.W_log_scale.shape, head.b_log_scale.shape
            ((4, 3), (4, 3), (3,))
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "in_features and out_features must be > 0; "
                f"got {in_features=}, {out_features=}."
            )
        if not sigma_init > 0:
            raise ValueError(f"sigma_init must be > 0; got {sigma_init}.")
        log_sigma_init = math.log(sigma_init)
        # Strict: at a tie jnp.clip splits the gradient (0.5 to b_s), so a
        # head initialised on the bound would start half-saturated.
        if not _LOG_SIGMA_MIN < log_sigma_init < _LOG_SIGMA_MAX:
            raise ValueError(
                "log(sigma_init) must lie strictly inside "
                f"({_LOG_SIGMA_MIN}, {_LOG_SIGMA_MAX}); "
                f"got sigma_init={sigma_init} (log = {log_sigma_init:.3f})."
            )
        if scale_init_factor < 0:
            raise ValueError(
                f"scale_init_factor must be >= 0; got {scale_init_factor}."
            )
        k_loc, k_scale = jr.split(key)
        return cls(
            W_loc=_glorot_uniform(k_loc, in_features, out_features),
            b_loc=jnp.zeros(out_features),
            W_log_scale=scale_init_factor
            * _glorot_uniform(k_scale, in_features, out_features),
            b_log_scale=jnp.full((out_features,), log_sigma_init),
            in_features=in_features,
            out_features=out_features,
        )

    def __call__(
        self, h: Float[Array, " D_in"]
    ) -> tuple[Float[Array, " K"], Float[Array, " K"]]:
        r"""Map one feature vector to $(\mu, \log\sigma)$.

        Args:
            h: Feature vector of shape ``(D_in,)``; ``vmap`` to batch.

        Returns:
            ``(mu, log_sigma)``, each of shape ``(K,)``; ``log_sigma`` is
            clamped to ``[log_sigma_min, log_sigma_max]``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> head = GaussianRegressionHead.init(4, key=jr.PRNGKey(0))
            >>> mu, log_sigma = head(jnp.ones(4))
            >>> mu.shape, float(log_sigma[0])
            ((1,), 0.0)
        """
        mu = einx.dot("d, d k -> k", h, self.W_loc) + self.b_loc
        raw = einx.dot("d, d k -> k", h, self.W_log_scale) + self.b_log_scale
        log_sigma = jnp.clip(raw, self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


__all__ = [
    "GaussianRegressionHead",
    "HeteroscedasticHead",
    "MCSigmoidDenseFA",
    "MCSoftmaxDenseFA",
    "hetero_noisy_logits",
]
