r"""Spectral normalisation — $c$-Lipschitz linear layers via power iteration.

`SpectralNormalization` wraps any linear `equinox.Module` exposing a
``.weight`` matrix and rescales that matrix to a fixed spectral norm, so the
wrapped map is (approximately) $c$-Lipschitz (Miyato et al., 2018). This is
the wrapper half of the SNGP / DUE recipe:
`geonnax.sngp.RandomFeatureGaussianProcess` supplies the distance-aware
output head, and spectral normalisation of the upstream dense layers is what
makes the feature extractor distance-preserving enough for that head to mean
anything.

The top singular value is estimated by power iteration on state $(u, v)$
carried as ordinary array fields, so the module stays a plain pytree. The
forward pass is pure: it returns a *new* module with $(u, v)$ advanced,
alongside the layer output.

The estimate approaches $\sigma(W)$ from below, so the realised Lipschitz
constant is $c\,\sigma(W)/\hat\sigma(W) \ge c$ until the iterates converge —
approximate normalisation, matching the reference implementations rather than
a provable bound. `SpectralNormalization` documents the contract in full.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


# Floor for the sigma divisor. An all-zero weight yields sigma == 0, and
# 0 * (coeff / 0) is NaN; flooring turns that into 0 * (coeff / eps) == 0.
_SIGMA_FLOOR = 1e-12


def _l2_normalize(x: Float[Array, " n"], eps: float = 1e-12) -> Float[Array, " n"]:
    """Return ``x / ||x||`` with a floor on the norm to keep gradients finite."""
    return x / jnp.maximum(jnp.linalg.norm(x), eps)


class SpectralNormalization(eqx.Module):
    r"""Wrap an Equinox linear layer to be $c$-Lipschitz via power iteration.

    For a weight matrix $W \in \mathbb{R}^{m \times n}$ and a bound $c > 0$
    the layer is evaluated with

    $$
    \hat W = W \, \frac{c}{\sigma(W)},
    \qquad
    \sigma(W) = \sup_{\|x\| = 1} \|W x\| ,
    $$

    so if $\sigma(W)$ were exact, $\|\hat W x\| \le c \|x\|$ and a stack of
    $L$ such layers would be $c^L$-Lipschitz. Miyato et al. (2018) and the
    SNGP / DUE line of work use $c \approx 0.95$. What is actually computed
    is an *estimate* of $\sigma(W)$ — see the contract note below.

    $\sigma(W)$ is estimated by power iteration on persistent state
    $u \in \mathbb{R}^m$, $v \in \mathbb{R}^n$:

    $$
    v \leftarrow \frac{W^{\top} u}{\|W^{\top} u\|},
    \qquad
    u \leftarrow \frac{W v}{\|W v\|},
    \qquad
    \hat\sigma(W) = |u^{\top} W v| \approx \sigma(W) .
    $$

    One iteration per forward pass is the standard setting — the estimate
    tracks $W$ as it moves during training. The iterates are treated as
    constants by autodiff (`jax.lax.stop_gradient`), so gradients flow
    through $W$ only, exactly as in the reference implementation.

    !!! warning "The bound is approximate, not enforced"
        Power iteration approaches $\sigma(W)$ **from below**: for any unit
        $u, v$, $|u^{\top} W v| \le \sigma(W)$. The realised spectral norm of
        $\hat W$ is therefore

        $$
        \sigma(\hat W) = c \, \frac{\sigma(W)}{\hat\sigma(W)} \; \ge \; c ,
        $$

        with equality only once the iterates converge. With the default
        ``n_power_iterations=1`` and a weight that keeps moving under the
        optimiser, the layer is *approximately* rather than provably
        $c$-Lipschitz — the same contract as
        [`torch.nn.utils.spectral_norm`](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html)
        and the reference SNGP / DUE implementations, which is what makes the
        recipe cheap enough to run every step.

        The gap closes at rate $(\sigma_2/\sigma_1)^{2k}$ in the number of
        iterations $k$, so it is set by the weight's spectral gap. Raise
        ``n_power_iterations`` if you need a tighter estimate per step. For a
        hard guarantee, rescale by an exact
        ``jnp.linalg.svd(W, compute_uv=False)[0]`` yourself — correct, but
        $\mathcal{O}(mn\min(m,n))$ per forward pass.

    !!! note "Rescaling is unconditional"
        $W$ is multiplied by $c / \hat\sigma(W)$ whether or not
        $\hat\sigma(W)$ exceeds $c$, so the normalisation drives the norm
        *to* $c$ rather than clipping it at $c$. Bias, if the wrapped layer
        has one, is left untouched: it shifts the output but does not change
        the Lipschitz constant.

        An all-zero weight is the one exception — it stays zero instead of
        producing ``NaN``, since a zero map already satisfies every Lipschitz
        bound.

    Attributes:
        layer: The wrapped layer; must expose ``.weight`` of shape ``(m, n)``.
        u: Left singular-vector estimate, shape ``(m,)``.
        v: Right singular-vector estimate, shape ``(n,)``.
        coeff: Lipschitz bound $c$.
        n_power_iterations: Power iterations performed per forward call.

    Examples:
        >>> import equinox as eqx, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.spectral_norm import SpectralNormalization
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> layer = eqx.nn.Linear(8, 4, use_bias=False, key=k1)
        >>> sn = SpectralNormalization.init(layer, coeff=0.95, key=k2)
        >>> new_sn, y = sn(jnp.ones(8))
        >>> y.shape
        (4,)
        >>> # (u, v) advanced; the module itself is unchanged.
        >>> bool(jnp.any(new_sn.u != sn.u))
        True
    """

    layer: eqx.Module
    u: Float[Array, " m"]
    v: Float[Array, " n"]
    coeff: float = eqx.field(static=True)
    n_power_iterations: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        layer: eqx.Module,
        *,
        coeff: float = 0.95,
        n_power_iterations: int = 1,
        key: PRNGKeyArray,
    ) -> SpectralNormalization:
        """Wrap ``layer`` and seed the power-iteration state from ``key``.

        Args:
            layer: Layer to wrap; must expose a 2D ``.weight`` of shape
                ``(m, n)`` (the `equinox.nn.Linear` convention:
                ``out_features`` first).
            coeff: Lipschitz bound $c$; must be > 0.
            n_power_iterations: Iterations per forward call; must be >= 1.
            key: PRNG key used to draw the initial $(u, v)$.

        Returns:
            The wrapped, spectrally normalised layer.

        Raises:
            ValueError: If ``layer`` has no ``.weight``, the weight is not 2D,
                ``coeff <= 0``, or ``n_power_iterations < 1``.

        Examples:
            >>> import equinox as eqx, jax.random as jr
            >>> from geonnax.spectral_norm import SpectralNormalization
            >>> k1, k2 = jr.split(jr.PRNGKey(0))
            >>> sn = SpectralNormalization.init(
            ...     eqx.nn.Linear(6, 3, key=k1), coeff=0.9, key=k2
            ... )
            >>> sn.u.shape, sn.v.shape
            ((3,), (6,))
        """
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise ValueError(
                f"layer must expose a `.weight` attribute, got {type(layer).__name__}."
            )
        if jnp.ndim(weight) != 2:
            raise ValueError(
                f"layer.weight must be 2D of shape (m, n), got shape "
                f"{jnp.shape(weight)}."
            )
        if coeff <= 0:
            raise ValueError(f"coeff must be > 0, got {coeff}.")
        if n_power_iterations < 1:
            raise ValueError(
                f"n_power_iterations must be >= 1, got {n_power_iterations}."
            )
        m, n = jnp.shape(weight)
        k_u, k_v = jr.split(key)
        return cls(
            layer=layer,
            u=_l2_normalize(jr.normal(k_u, (m,))),
            v=_l2_normalize(jr.normal(k_v, (n,))),
            coeff=coeff,
            n_power_iterations=n_power_iterations,
        )

    @property
    def weight(self) -> Float[Array, "m n"]:
        """The wrapped layer's raw (un-normalised) weight matrix."""
        return self.layer.weight  # ty: ignore[unresolved-attribute]

    def _power_iterate(
        self,
    ) -> tuple[Float[Array, " m"], Float[Array, " n"]]:
        """Advance $(u, v)$ by ``n_power_iterations`` steps, detached from autodiff."""
        weight = jax.lax.stop_gradient(self.weight)
        u, v = jax.lax.stop_gradient(self.u), jax.lax.stop_gradient(self.v)
        for _ in range(self.n_power_iterations):
            v = _l2_normalize(weight.T @ u)
            u = _l2_normalize(weight @ v)
        return u, v

    def estimated_sigma(self) -> Float[Array, ""]:
        r"""Current power-iteration estimate $|u^{\top} W v|$ of $\sigma(W)$.

        Uses the stored $(u, v)$ without advancing them, so repeated calls on
        the same module agree. The absolute value keeps the estimate
        non-negative for an unconverged $(u, v)$ — notably the random pair
        from `init`, before any forward pass has run — without changing
        anything at convergence, where $u^{\top} W v > 0$ already.

        This is a *lower* bound on $\sigma(W)$ for any unit $u, v$, tight only
        once the iterates converge; see the class docstring on what that means
        for the Lipschitz contract.

        Returns:
            Scalar estimate of the top singular value, always >= 0.

        Examples:
            >>> import equinox as eqx, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.spectral_norm import SpectralNormalization
            >>> k1, k2 = jr.split(jr.PRNGKey(0))
            >>> sn = SpectralNormalization.init(
            ...     eqx.nn.Linear(5, 5, key=k1), key=k2
            ... )
            >>> for _ in range(30):
            ...     sn, _ = sn(jnp.ones(5))
            >>> true = jnp.linalg.svd(sn.weight, compute_uv=False)[0]
            >>> bool(jnp.abs(sn.estimated_sigma() - true) < 1e-3)
            True
        """
        return jnp.abs(self.u @ self.weight @ self.v)

    def normalized_layer(self) -> eqx.Module:
        r"""Return the wrapped layer with $W$ replaced by $c\,W/\hat\sigma(W)$.

        The rescaling uses the *current* $(u, v)$ — call `__call__` (or
        rebuild from its returned module) to advance them first. The realised
        spectral norm is $c\,\sigma(W)/\hat\sigma(W) \ge c$, reaching $c$ as
        the iterates converge; see the class docstring.

        A zero weight is passed through unchanged rather than turned into
        ``NaN``: the divisor is floored at a small epsilon, so ``0 * (c/eps)``
        stays ``0``.

        Returns:
            A copy of the wrapped layer whose weight has spectral norm
            approximately $c$ (exactly $c$ at convergence, ``0`` for a zero
            weight).

        Examples:
            >>> import equinox as eqx, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.spectral_norm import SpectralNormalization
            >>> k1, k2 = jr.split(jr.PRNGKey(0))
            >>> sn = SpectralNormalization.init(
            ...     eqx.nn.Linear(4, 4, use_bias=False, key=k1),
            ...     coeff=0.95, key=k2,
            ... )
            >>> for _ in range(30):
            ...     sn, _ = sn(jnp.ones(4))
            >>> sigma = jnp.linalg.svd(
            ...     sn.normalized_layer().weight, compute_uv=False
            ... )[0]
            >>> bool(jnp.abs(sigma - 0.95) < 1e-3)
            True
        """
        # Floor the divisor so an all-zero weight scales to 0 rather than NaN.
        sigma = jnp.maximum(self.estimated_sigma(), _SIGMA_FLOOR)
        return eqx.tree_at(
            lambda m: m.weight,
            self.layer,
            self.weight * (self.coeff / sigma),
        )

    def __call__(
        self,
        x: Float[Array, " n"],
    ) -> tuple[SpectralNormalization, Float[Array, " m"]]:
        r"""Advance $(u, v)$, then apply the layer with $\hat W = c W/\sigma(W)$.

        Args:
            x: Input vector of shape ``(n,)``.

        Returns:
            Tuple ``(updated_module, output)``. The returned module carries
            the advanced power-iteration state; ``self`` is unchanged. Thread
            it forward so the estimate keeps tracking $W$ during training.

        Examples:
            >>> import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.spectral_norm import SpectralNormalization
            >>> k1, k2 = jr.split(jr.PRNGKey(0))
            >>> sn = SpectralNormalization.init(
            ...     eqx.nn.Linear(8, 4, use_bias=False, key=k1), key=k2
            ... )
            >>> _, y = sn(jnp.ones(8))
            >>> y.shape
            (4,)
            >>> # Map over a batch by keeping only the output.
            >>> jax.vmap(lambda z: sn(z)[1])(jnp.ones((16, 8))).shape
            (16, 4)
        """
        u, v = self._power_iterate()
        advanced = eqx.tree_at(lambda m: (m.u, m.v), self, (u, v))
        return advanced, advanced.normalized_layer()(x)


__all__ = ["SpectralNormalization"]
