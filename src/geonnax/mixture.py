r"""Mixture-density-network output head (Bishop, 1994).

`MixtureOfGaussiansDenseHead` turns a feature vector into the parameters of a
diagonal-covariance Gaussian mixture — gate weights, means, and log standard
deviations. It returns *parameters*, not a distribution object, so the head
stays free of any probabilistic-framework dependency; construct the mixture
downstream (e.g. ``numpyro.distributions.MixtureSameFamily``).

The Bayesian variant — priors on the projection weights — lives downstream.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MixtureOfGaussiansDenseHead(eqx.Module):
    r"""Deterministic mixture-density-network output head.

    For a feature vector $x \in \mathbb{R}^d$ and $K$ components the head
    parameterises

    $$
    p(y \mid x) = \sum_{k=1}^{K} \pi_k(x)\,
    \mathcal{N}\!\bigl(y \mid \mu_k(x), \Sigma_k(x)\bigr),
    $$

    with $\pi(x) = \mathrm{softmax}(W_\pi x + b_\pi)$,
    $\mu_k(x) = W_{\mu,k} x + b_{\mu,k}$, and
    $\log \sigma_k(x) = W_{\sigma,k} x + b_{\sigma,k}$. Covariances are
    diagonal, $\Sigma_k = \mathrm{diag}(\sigma_k^2)$; full covariance is out
    of scope for now.

    Each component owns independent mean and log-scale projections, matching
    the Edward2 `MixtureOfGaussiansDense` layout.

    Attributes:
        pi_proj: Gate projection, ``in_features -> num_components``.
        mu_projs: Per-component mean projections, ``in_features -> num_outputs``.
        log_sigma_projs: Per-component log-scale projections.
        in_features: Input feature dimension $d$.
        num_components: Number of mixture components $K$.
        num_outputs: Output dimension $D$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.mixture import MixtureOfGaussiansDenseHead
        >>> head = MixtureOfGaussiansDenseHead.init(64, 5, 1, key=jr.PRNGKey(0))
        >>> pi, mu, log_sigma = head(jnp.ones(64))
        >>> pi.shape, mu.shape, log_sigma.shape
        ((5,), (5, 1), (5, 1))
        >>> bool(jnp.abs(pi.sum() - 1.0) < 1e-6)  # gates are normalised
        True
    """

    pi_proj: eqx.nn.Linear
    mu_projs: tuple[eqx.nn.Linear, ...]
    log_sigma_projs: tuple[eqx.nn.Linear, ...]
    in_features: int = eqx.field(static=True)
    num_components: int = eqx.field(static=True)
    num_outputs: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        num_components: int,
        num_outputs: int,
        *,
        key: PRNGKeyArray,
    ) -> MixtureOfGaussiansDenseHead:
        """Build the gate projection plus per-component mean / log-scale heads.

        Args:
            in_features: Input feature dimension $d$; must be > 0.
            num_components: Number of components $K$; must be > 0.
            num_outputs: Output dimension $D$; must be > 0.
            key: PRNG key, split across the ``2 * num_components + 1``
                projections.

        Returns:
            The initialised head.

        Raises:
            ValueError: If any of the three dimensions is non-positive.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.mixture import MixtureOfGaussiansDenseHead
            >>> head = MixtureOfGaussiansDenseHead.init(8, 3, 2, key=jr.PRNGKey(0))
            >>> len(head.mu_projs), len(head.log_sigma_projs)
            (3, 3)
        """
        for name, value in (
            ("in_features", in_features),
            ("num_components", num_components),
            ("num_outputs", num_outputs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")
        k_pi, k_mu, k_sigma = jax.random.split(key, 3)
        mu_keys = jax.random.split(k_mu, num_components)
        sigma_keys = jax.random.split(k_sigma, num_components)
        return cls(
            pi_proj=eqx.nn.Linear(in_features, num_components, key=k_pi),
            mu_projs=tuple(
                eqx.nn.Linear(in_features, num_outputs, key=k) for k in mu_keys
            ),
            log_sigma_projs=tuple(
                eqx.nn.Linear(in_features, num_outputs, key=k) for k in sigma_keys
            ),
            in_features=in_features,
            num_components=num_components,
            num_outputs=num_outputs,
        )

    def __call__(
        self,
        x: Float[Array, " in_features"],
    ) -> tuple[
        Float[Array, " K"],
        Float[Array, "K D"],
        Float[Array, "K D"],
    ]:
        r"""Map a feature vector to $(\pi, \mu, \log\sigma)$.

        Args:
            x: Feature vector of shape ``(in_features,)``.

        Returns:
            Tuple ``(pi, mu, log_sigma)`` with shapes ``(K,)``, ``(K, D)``,
            ``(K, D)``. ``pi`` is softmax-normalised; ``log_sigma`` is
            unconstrained — exponentiate it to get the standard deviations.

        Examples:
            >>> import jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.mixture import MixtureOfGaussiansDenseHead
            >>> head = MixtureOfGaussiansDenseHead.init(4, 3, 2, key=jr.PRNGKey(0))
            >>> pi, mu, log_sigma = head(jnp.ones(4))
            >>> pi.shape, mu.shape, log_sigma.shape
            ((3,), (3, 2), (3, 2))
            >>> # vmap for a batch of features
            >>> batched = jax.vmap(head)(jnp.ones((7, 4)))
            >>> [a.shape for a in batched]
            [(7, 3), (7, 3, 2), (7, 3, 2)]
        """
        pi = jax.nn.softmax(self.pi_proj(x))
        mu = jnp.stack([proj(x) for proj in self.mu_projs])
        log_sigma = jnp.stack([proj(x) for proj in self.log_sigma_projs])
        return pi, mu, log_sigma


__all__ = ["MixtureOfGaussiansDenseHead"]
