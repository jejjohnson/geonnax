r"""Linear-chain conditional random field — forward algorithm and Viterbi.

`LinearChainCRF` holds the learnable pairwise-potential matrix of a
linear-chain CRF (Sutton & McCallum, 2012) and provides the two deterministic
kernels a sequence-tagging model needs: the log-partition function via the
forward recursion, and MAP decoding via Viterbi. Unary potentials are supplied
per call — they are normally the per-step output of an upstream feature
extractor, which keeps this module independent of the backbone.

Both recursions run in $\mathcal{O}(T K^2)$ under `jax.lax.scan`, so they are
`jit`-, `vmap`-, and `grad`-friendly. The Bayesian variant — priors on the
pairwise potentials — lives downstream.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray


class LinearChainCRF(eqx.Module):
    r"""Deterministic linear-chain CRF — forward algorithm plus Viterbi decoding.

    With unary potentials $\phi_t(y_t)$ (supplied per call) and a learnable
    pairwise matrix $\psi(y_t, y_{t+1})$, the model is

    $$
    p(y_{1:T} \mid h_{1:T}) = \frac{1}{Z}
    \exp\!\Bigl( \sum_{t=1}^{T} \phi_t(y_t)
    + \sum_{t=1}^{T-1} \psi(y_t, y_{t+1}) \Bigr).
    $$

    There are no separate start / stop potentials: fold any such bias into the
    unary potentials of the first and last steps.

    The forward recursion
    $\alpha_{t+1}(j) = \phi_{t+1}(j)
    + \operatorname{logsumexp}_i \bigl[\alpha_t(i) + \psi(i, j)\bigr]$
    gives $\log Z = \operatorname{logsumexp}_j \alpha_T(j)$; replacing the
    log-sum-exp with a max and recording the argmax gives Viterbi.

    Attributes:
        pairwise: Transition potentials $\psi$ of shape ``(K, K)``, indexed
            ``[from_label, to_label]``.
        num_labels: Label-alphabet size $K$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.crf import LinearChainCRF
        >>> crf = LinearChainCRF.init(num_labels=4, key=jr.PRNGKey(0))
        >>> unary = jr.normal(jr.PRNGKey(1), (6, 4))
        >>> path = crf.viterbi(unary)
        >>> path.shape
        (6,)
        >>> # A log-probability is never positive.
        >>> bool(crf.log_prob(unary, path) <= 0.0)
        True
    """

    pairwise: Float[Array, "K K"]
    num_labels: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        num_labels: int,
        *,
        key: PRNGKeyArray,
        scale: float = 0.01,
    ) -> LinearChainCRF:
        """Build a CRF with small random transition potentials.

        Args:
            num_labels: Label-alphabet size $K$; must be > 0.
            key: PRNG key for the pairwise matrix.
            scale: Standard deviation of the initial pairwise potentials.
                The default keeps the chain close to independent at init.

        Returns:
            The initialised CRF.

        Raises:
            ValueError: If ``num_labels <= 0`` or ``scale < 0``.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.crf import LinearChainCRF
            >>> crf = LinearChainCRF.init(num_labels=10, key=jr.PRNGKey(0))
            >>> crf.pairwise.shape
            (10, 10)
        """
        if num_labels <= 0:
            raise ValueError(f"num_labels must be > 0, got {num_labels}.")
        if scale < 0:
            raise ValueError(f"scale must be >= 0, got {scale}.")
        return cls(
            pairwise=scale * jr.normal(key, (num_labels, num_labels)),
            num_labels=num_labels,
        )

    def _score(
        self,
        unary: Float[Array, "T K"],
        y: Int[Array, " T"],
    ) -> Float[Array, ""]:
        r"""Unnormalised score $\sum_t \phi_t(y_t) + \sum_t \psi(y_t, y_{t+1})$."""
        unary_score = unary[jnp.arange(unary.shape[0]), y].sum()
        # Empty slices for T == 1 sum to zero, which is the correct no-edge case.
        pair_score = self.pairwise[y[:-1], y[1:]].sum()
        return unary_score + pair_score

    def log_partition(self, unary: Float[Array, "T K"]) -> Float[Array, ""]:
        r"""Log-partition $\log Z$ via the forward recursion.

        Args:
            unary: Unary potentials of shape ``(T, K)``, ``T >= 1``.

        Returns:
            Scalar $\log Z$.

        Examples:
            >>> import jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.crf import LinearChainCRF
            >>> crf = LinearChainCRF.init(num_labels=3, key=jr.PRNGKey(0))
            >>> # A single step has no transitions: log Z = logsumexp of the unary.
            >>> unary = jnp.array([[0.0, 1.0, 2.0]])
            >>> bool(jnp.allclose(
            ...     crf.log_partition(unary), jax.nn.logsumexp(unary[0])
            ... ))
            True
        """

        def step(
            alpha: Float[Array, " K"], phi: Float[Array, " K"]
        ) -> tuple[Float[Array, " K"], None]:
            # α_{t+1}(j) = φ_{t+1}(j) + logsumexp_i [α_t(i) + ψ(i, j)].
            return phi + jax.nn.logsumexp(alpha[:, None] + self.pairwise, axis=0), None

        alpha, _ = jax.lax.scan(step, unary[0], unary[1:])
        return jax.nn.logsumexp(alpha)

    def log_prob(
        self,
        unary: Float[Array, "T K"],
        y: Int[Array, " T"],
    ) -> Float[Array, ""]:
        r"""Log-likelihood $\log p(y_{1:T} \mid h_{1:T})$ of a label sequence.

        Negate it for the usual training loss.

        Args:
            unary: Unary potentials of shape ``(T, K)``, ``T >= 1``.
            y: Label sequence of shape ``(T,)`` with entries in ``[0, K)``.

        Returns:
            Scalar log-probability, ``score(y) - log_partition(unary)``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.crf import LinearChainCRF
            >>> crf = LinearChainCRF.init(num_labels=3, key=jr.PRNGKey(0))
            >>> unary = jr.normal(jr.PRNGKey(1), (4, 3))
            >>> # Log-probabilities over all sequences sum to 1.
            >>> import itertools
            >>> total = sum(
            ...     float(jnp.exp(crf.log_prob(unary, jnp.array(y))))
            ...     for y in itertools.product(range(3), repeat=4)
            ... )
            >>> bool(abs(total - 1.0) < 1e-5)
            True
        """
        return self._score(unary, y) - self.log_partition(unary)

    def viterbi(self, unary: Float[Array, "T K"]) -> Int[Array, " T"]:
        r"""MAP label sequence $\arg\max_y p(y_{1:T} \mid h_{1:T})$.

        Args:
            unary: Unary potentials of shape ``(T, K)``, ``T >= 1``.

        Returns:
            Integer label sequence of shape ``(T,)``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.crf import LinearChainCRF
            >>> # With zero transitions the MAP path is the per-step argmax.
            >>> crf = LinearChainCRF.init(num_labels=3, key=jr.PRNGKey(0), scale=0.0)
            >>> unary = jnp.array([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
            >>> crf.viterbi(unary).tolist()
            [1, 0]
        """

        def step(
            delta: Float[Array, " K"], phi: Float[Array, " K"]
        ) -> tuple[Float[Array, " K"], Int[Array, " K"]]:
            # δ_{t+1}(j) = φ_{t+1}(j) + max_i [δ_t(i) + ψ(i, j)]; keep the argmax.
            scores = delta[:, None] + self.pairwise
            return phi + jnp.max(scores, axis=0), jnp.argmax(scores, axis=0)

        delta, backptrs = jax.lax.scan(step, unary[0], unary[1:])
        last = jnp.argmax(delta)

        def backtrace(
            nxt: Int[Array, ""], bp: Int[Array, " K"]
        ) -> tuple[Int[Array, ""], Int[Array, ""]]:
            prev = bp[nxt]
            return prev, prev

        # Walk the pointers back from step T-1; `prevs` holds labels 0 … T-2.
        _, prevs = jax.lax.scan(backtrace, last, backptrs, reverse=True)
        return jnp.concatenate([prevs, last[None]])


__all__ = ["LinearChainCRF"]
