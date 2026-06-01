"""Spectral-Normalized Gaussian Process (SNGP) output layer.

Deterministic ``equinox.Module`` cores. The probabilistic wrapper
that promotes each weight array to a ``pyrox_param`` site lives in
the consuming probabilistic library.

* :class:`LaplaceRandomFeatureCovariance` — pure-functional container
  for the Laplace-approximation precision matrix used at SNGP test
  time. Updated via the EMA of feature outer products during training.
* :class:`RandomFeatureGaussianProcess` — SNGP output head (Liu et al.,
  2020). RFF feature map :math:`\\phi(x)` plus a linear mean head and
  a Laplace covariance over the linear weights.

This module implements *just the SNGP head* — spectral normalisation
of upstream dense layers is a separate concern and the user is
responsible for that.
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


def _glorot_normal(
    key: PRNGKeyArray,
    fan_in: int,
    fan_out: int,
    scale: float = 1.0,
) -> Float[Array, "F_in F_out"]:
    std = scale * math.sqrt(2.0 / (fan_in + fan_out))
    return std * jr.normal(key, (fan_in, fan_out))


class LaplaceRandomFeatureCovariance(eqx.Module):
    r"""Laplace-approximation precision for an SNGP output head.

    Stores the precision matrix :math:`\hat{\Lambda} \in \mathbb{R}^{D \times D}`
    over the linear weights of the output layer. Updated as an
    exponential moving average of feature outer products during
    training:

    .. math::

        \hat{\Lambda}_{t+1} \leftarrow m\,\hat{\Lambda}_t
        + (1 - m)\,\frac{1}{B} \sum_{b=1}^{B} \phi(x_b)\,\phi(x_b)^\top.

    At test time the predictive variance for a feature vector
    :math:`\phi(x_*)` is

    .. math::

        \sigma^2(x_*) = \phi(x_*)^\top \hat{\Sigma}\, \phi(x_*),
        \qquad \hat{\Sigma} = \hat{\Lambda}^{-1},

    computed stably via a Cholesky solve.

    The container is *pure-functional*: :meth:`update` returns a new
    instance with an updated precision rather than mutating ``self``,
    matching how Equinox composes immutable PyTrees with optimisers.
    A small ridge :math:`\lambda` initialises the precision at
    :math:`\lambda I` and is *also* added at solve-time inside
    :meth:`covariance` and :meth:`variance_at` so the Cholesky stays
    numerically well-conditioned even after many EMA steps with low
    momentum (which would otherwise let the ridge contribution decay
    geometrically and the precision approach singularity).
    Equivalently :math:`\hat\Sigma = (\hat\Lambda + \lambda I)^{-1}` —
    the Bayesian-linear-regression interpretation of SNGP, where
    :math:`\lambda I` is a Gaussian prior precision on the head weights.

    Attributes:
        precision: Current precision matrix :math:`\hat{\Lambda}`.
        momentum: EMA momentum :math:`m \in [0, 1]`. Higher values give
            slower updates; ``0.999`` works well for most settings.
        ridge: Diagonal ridge :math:`\lambda`. Used both as the init
            value of ``precision`` and as a solve-time jitter to keep
            the Cholesky well-defined.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.sngp import LaplaceRandomFeatureCovariance
        >>> cov = LaplaceRandomFeatureCovariance.init(4, ridge=1.0)
        >>> cov.precision.shape
        (4, 4)
        >>> cov.variance_at(jnp.eye(4)).shape
        (4,)
    """

    precision: Float[Array, "D D"]
    momentum: float = eqx.field(static=True, default=0.999)
    ridge: float = eqx.field(static=True, default=1.0)

    @classmethod
    def init(
        cls,
        num_features: int,
        *,
        momentum: float = 0.999,
        ridge: float = 1.0,
    ) -> LaplaceRandomFeatureCovariance:
        r"""Construct a fresh covariance container with ``ridge * I`` precision.

        Examples:
            >>> from geonnax.sngp import LaplaceRandomFeatureCovariance
            >>> cov = LaplaceRandomFeatureCovariance.init(4, momentum=0.9)
            >>> cov.precision.shape
            (4, 4)
        """
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0; got {num_features}.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must lie in [0, 1]; got {momentum}.")
        if ridge <= 0:
            raise ValueError(f"ridge must be > 0; got {ridge}.")
        return cls(
            precision=ridge * jnp.eye(num_features),
            momentum=momentum,
            ridge=ridge,
        )

    def update(self, features: Float[Array, "B D"]) -> LaplaceRandomFeatureCovariance:
        r"""Return a new container with EMA-updated precision.

        :math:`\hat\Lambda \leftarrow m\,\hat\Lambda + (1-m)\,\Phi^\top\Phi/B`.

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.sngp import LaplaceRandomFeatureCovariance
            >>> cov = LaplaceRandomFeatureCovariance.init(4)
            >>> new = cov.update(jnp.ones((8, 4)))
            >>> new is cov
            False
        """
        B = features.shape[0]
        # Feature Gram ΦᵀΦ / B: contract the batch axis b → (D, D).
        outer = einx.dot("b d, b e -> d e", features, features) / B
        new_precision = self.momentum * self.precision + (1.0 - self.momentum) * outer
        return eqx.tree_at(lambda c: c.precision, self, new_precision)

    def _chol(self) -> Float[Array, "D D"]:
        # Symmetrise to absorb floating-point asymmetry in the EMA, then
        # add ridge jitter so the matrix is guaranteed positive-definite
        # regardless of how the EMA has evolved.
        sym = 0.5 * (self.precision + einx.id("i j -> j i", self.precision))
        D = sym.shape[0]
        return jnp.linalg.cholesky(sym + self.ridge * jnp.eye(D, dtype=sym.dtype))

    def covariance(self) -> Float[Array, "D D"]:
        r"""Inverse of the precision matrix (one-shot Cholesky inversion).

        :math:`\hat\Sigma = (\hat\Lambda + \lambda I)^{-1}`, shape ``(D, D)``.

        Examples:
            >>> from geonnax.sngp import LaplaceRandomFeatureCovariance
            >>> cov = LaplaceRandomFeatureCovariance.init(4)
            >>> cov.covariance().shape
            (4, 4)
        """
        L = self._chol()
        D = self.precision.shape[0]
        return jax.scipy.linalg.cho_solve((L, True), jnp.eye(D))

    def variance_at(self, features: Float[Array, "N D"]) -> Float[Array, " N"]:
        r"""Per-row predictive variance :math:`\phi(x_n)^\top \hat{\Sigma}\,\phi(x_n)`.

        Computed via a triangular solve to avoid materialising the full
        :math:`D \times D` covariance:

        .. math::

            y = L^{-1} \phi(x_n)^\top, \qquad
            \sigma^2(x_n) = \lVert y \rVert_2^2
            = \phi(x_n)^\top (L L^\top)^{-1} \phi(x_n).

        Examples:
            >>> import jax.numpy as jnp
            >>> from geonnax.sngp import LaplaceRandomFeatureCovariance
            >>> cov = LaplaceRandomFeatureCovariance.init(4)
            >>> cov.variance_at(jnp.eye(4)).shape
            (4,)
        """
        L = self._chol()
        # Solve L y = Φᵀ ; (D, D)\(D, N) -> (D, N), one column per row of Φ.
        y = jax.scipy.linalg.solve_triangular(
            L, einx.id("n d -> d n", features), lower=True
        )
        # σ²(xₙ) = ‖yₙ‖² ; sum over D -> (N,)
        return einx.sum("[d] n", y * y)


class RandomFeatureGaussianProcess(eqx.Module):
    r"""SNGP output layer (Liu et al., 2020) — deterministic core.

    A random Fourier feature map followed by a learnable linear head,
    plus a Laplace-approximation covariance over the linear weights.
    The forward pass returns the mean prediction and (optionally) a
    per-input predictive variance summarising distance from the
    training distribution.

    Forward (mean):

    .. math::

        \phi(x) = \sqrt{\tfrac{2}{D}}\,\cos\!\bigl(W\, x / \ell + b\bigr),
        \qquad \mu(x) = \phi(x)\, H + b_H.

    The frequencies :math:`W` and bias :math:`b` of the RFF map are
    *frozen* (they implicitly define the kernel approximation): they
    are stored as real array fields then guarded with
    :func:`jax.lax.stop_gradient` inside :meth:`feature_map` so
    SGD-style optimisers leave them untouched. The lengthscale
    :math:`\ell`, the linear head :math:`H, b_H`, and the Laplace
    precision are the trainable / updated quantities.

    Predictive variance — when :math:`\hat{\Lambda}` is the current
    precision matrix:

    .. math::

        \sigma^2(x_*) = \phi(x_*)^\top \hat{\Lambda}^{-1}\, \phi(x_*).

    Training pattern (one minibatch):

    1. ``mean = layer(x)`` returns the mean prediction. Compute the
       loss, take a gradient step on the trainable arrays as usual.
    2. After the gradient step, call
       ``new_layer = layer.update_precision(features)`` where
       ``features`` is the result of :meth:`feature_map` evaluated on
       the same minibatch using the *updated* parameters. This returns
       a new layer with the LRFC's precision EMA-updated.

    At inference, ``mean, var = layer(x, return_cov=True)`` produces
    the mean and the Laplace per-input predictive variance.

    The consuming probabilistic library can promote ``W`` / ``bias`` /
    ``lengthscale`` / ``output_linear`` / ``output_bias`` to parameter
    sites without changing the forward math.

    Attributes:
        W: Frozen RFF frequencies, shape ``(D_in, D)``, drawn from a
            standard Normal (the RBF spectral density).
        bias: Frozen RFF biases, shape ``(D,)``, drawn from
            ``Uniform(0, 2 pi)``.
        lengthscale: RFF lengthscale :math:`\ell`. Scalar, trainable.
        output_linear: Linear head, shape ``(D, D_out)``.
        output_bias: Output bias, shape ``(D_out,)``.
        covariance: The :class:`LaplaceRandomFeatureCovariance` instance.
        in_features: Input dimension :math:`D_\mathrm{in}`.
        num_features: Number of random Fourier features :math:`D`.
        out_features: Output dimension :math:`D_\mathrm{out}`.

    References:
        Liu, J. Z., et al. (2020). *Simple and Principled Uncertainty
        Estimation with Deterministic Deep Learning via Distance
        Awareness.* NeurIPS.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.sngp import RandomFeatureGaussianProcess
        >>> layer = RandomFeatureGaussianProcess.init(
        ...     in_features=3, num_features=16, out_features=2, key=jr.PRNGKey(0)
        ... )
        >>> layer(jnp.ones(3)).shape
        (2,)
        >>> mean, var = layer(jnp.ones(3), return_cov=True)
        >>> mean.shape, var.shape
        ((2,), ())
    """

    W: Float[Array, "D_in D"]
    bias: Float[Array, " D"]
    lengthscale: Float[Array, ""]
    output_linear: Float[Array, "D D_out"]
    output_bias: Float[Array, " D_out"]
    covariance: LaplaceRandomFeatureCovariance
    in_features: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        num_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        init_lengthscale: float = 1.0,
        momentum: float = 0.999,
        ridge: float = 1.0,
        head_scale: float = 0.01,
    ) -> Self:
        r"""Construct an SNGP head with frozen RFF freqs and an empty precision.

        Frequencies ``W`` are drawn from :math:`\mathcal N(0, 1)` (the RBF
        spectral density) and biases ``b`` from :math:`\mathrm{Uniform}(0, 2\pi)`;
        both are frozen. The linear head is Glorot-scaled and the Laplace
        precision starts at ``ridge * I``.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.sngp import RandomFeatureGaussianProcess
            >>> layer = RandomFeatureGaussianProcess.init(
            ...     in_features=3, num_features=16, out_features=2, key=jr.PRNGKey(0)
            ... )
            >>> layer.W.shape
            (3, 16)
        """
        if in_features <= 0 or num_features <= 0 or out_features <= 0:
            raise ValueError(
                "in_features, num_features, out_features must all be > 0; "
                f"got {in_features=}, {num_features=}, {out_features=}."
            )
        if init_lengthscale <= 0:
            raise ValueError(f"init_lengthscale must be > 0; got {init_lengthscale}.")
        kw, kb, kh = jr.split(key, 3)
        W = jr.normal(kw, (in_features, num_features))
        bias = jr.uniform(kb, (num_features,), minval=0.0, maxval=2 * jnp.pi)
        output_linear = _glorot_normal(kh, num_features, out_features, scale=head_scale)
        output_bias = jnp.zeros(out_features)
        cov = LaplaceRandomFeatureCovariance.init(
            num_features, momentum=momentum, ridge=ridge
        )
        return cls(
            W=W,
            bias=bias,
            lengthscale=jnp.asarray(init_lengthscale),
            output_linear=output_linear,
            output_bias=output_bias,
            covariance=cov,
            in_features=in_features,
            num_features=num_features,
            out_features=out_features,
        )

    def feature_map(self, x: Float[Array, " D_in"]) -> Float[Array, " D"]:
        r"""Random Fourier feature map: :math:`\phi(x) = \sqrt{2/D}\,\cos(Wx/\ell + b)`.

        Single-example: ``x: (D_in,)`` → ``(D,)``. Frequencies and bias
        are guarded with :func:`jax.lax.stop_gradient` so gradient-based
        optimisers leave them frozen at their init values. The
        lengthscale is the active bandwidth control.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.sngp import RandomFeatureGaussianProcess
            >>> layer = RandomFeatureGaussianProcess.init(
            ...     in_features=3, num_features=16, out_features=2, key=jr.PRNGKey(0)
            ... )
            >>> layer.feature_map(jnp.ones(3)).shape
            (16,)
        """
        W = jax.lax.stop_gradient(self.W)
        b = jax.lax.stop_gradient(self.bias)
        # z = Wx/ℓ + b ; (D_in,)·(D_in, D) -> (D,)
        z = einx.dot("d, d f -> f", x, W) / self.lengthscale + b
        # φ(x) = sqrt(2/D) cos(z) ; (D,)
        return jnp.sqrt(2.0 / self.num_features) * jnp.cos(z)

    def __call__(
        self,
        x: Float[Array, " D_in"],
        *,
        return_cov: bool = False,
    ) -> Float[Array, " D_out"] | tuple[Float[Array, " D_out"], Float[Array, ""]]:
        r"""Mean prediction (and optional Laplace variance) for one input.

        Returns ``mean: (D_out,)``, or ``(mean, var)`` with scalar ``var``
        when ``return_cov=True``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.sngp import RandomFeatureGaussianProcess
            >>> layer = RandomFeatureGaussianProcess.init(
            ...     in_features=3, num_features=16, out_features=2, key=jr.PRNGKey(0)
            ... )
            >>> layer(jnp.ones(3)).shape
            (2,)
        """
        features = self.feature_map(x)  # (D_in,) -> (D,)
        # μ(x) = φ(x) H + b_H ; (D,)·(D, D_out) -> (D_out,)
        mean = einx.dot("f, f o -> o", features, self.output_linear) + self.output_bias
        if return_cov:
            # variance_at takes (N, D); single example becomes a row of 1.
            var = self.covariance.variance_at(features[None, :])[0]
            return mean, var
        return mean

    def update_precision(self, features: Float[Array, "B D"]) -> Self:
        """Return a new layer with an EMA-updated Laplace precision.

        Pure-functional: ``self`` is unchanged. Pass a *batch* of features
        ``(B, D)`` computed on the current minibatch — e.g. by
        ``jax.vmap(self.feature_map)(x_batch)`` — and the update folds
        the empirical second moment into the EMA. Call this once per
        training batch *after* the gradient step.

        Examples:
            >>> import jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.sngp import RandomFeatureGaussianProcess
            >>> layer = RandomFeatureGaussianProcess.init(
            ...     in_features=3, num_features=8, out_features=1, key=jr.PRNGKey(0)
            ... )
            >>> feats = jax.vmap(layer.feature_map)(jnp.ones((4, 3)))
            >>> new = layer.update_precision(feats)
            >>> new.covariance.precision.shape
            (8, 8)
        """
        new_cov = self.covariance.update(features)
        return eqx.tree_at(lambda layer: layer.covariance, self, new_cov)


__all__ = [
    "LaplaceRandomFeatureCovariance",
    "RandomFeatureGaussianProcess",
]
