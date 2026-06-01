"""Deterministic core for Deep Variational SSGP (Cutajar et al. 2017).

A stack of random Fourier feature (RFF) maps composed with linear
projections. The Bayesian wrapper in the consuming library swaps the
stored ``W_freq`` / ``W_proj`` / ``lengthscale`` arrays for sample sites
with appropriate priors.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.randfeat import rff_forward


class DeepVSSGPCore(eqx.Module):
    r"""Deterministic deep RFF spectral core.

    Stacks :math:`L` random Fourier feature layers, each composed of an
    RFF map :math:`\Phi_l(z;\, \Omega_l, \ell_l) =
    \sqrt{1/M}\,[\cos(z\,\Omega_l/\ell_l), \sin(z\,\Omega_l/\ell_l)]`
    followed by a linear projection :math:`W_l \in
    \mathbb{R}^{2M \times d_{\mathrm{out}, l}}`. The deterministic
    forward composes these layers without sampling.

    Attributes:
        W_freqs: Per-layer RFF frequency matrices. Layer ``l`` has shape
            ``(d_in_l, n_features)`` where ``d_in_0 = in_features`` and
            ``d_in_l = hidden_features`` for ``l >= 1``.
        W_projs: Per-layer projection matrices. Layer ``l`` has shape
            ``(2 * n_features, d_out_l)`` where ``d_out_l =
            hidden_features`` for ``l < depth - 1`` and ``out_features``
            for the readout.
        lengthscales: Per-layer kernel lengthscale; shape ``(depth,)``.
        in_features: Input dimension.
        hidden_features: Inter-layer dimension.
        out_features: Output dimension.
        n_features: Per-layer Fourier-feature pair count :math:`M`.
        depth: Total number of stacked RFF layers :math:`L`. Must be
            :math:`\ge 1`.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.vssgp import DeepVSSGPCore
        >>> core = DeepVSSGPCore.init(
        ...     in_features=2,
        ...     hidden_features=4,
        ...     out_features=1,
        ...     depth=3,
        ...     key=jr.PRNGKey(0),
        ...     n_features=8,
        ... )
        >>> core(jnp.zeros(2)).shape
        (1,)
    """

    W_freqs: list[Float[Array, "d_in n_features"]]
    W_projs: list[Float[Array, "d_rff d_out"]]
    lengthscales: Float[Array, " depth"]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: Array,
        n_features: int = 64,
        lengthscale: float = 1.0,
        prior_std: float = 1.0,
    ) -> DeepVSSGPCore:
        """Construct a :class:`DeepVSSGPCore`.

        Frequencies are drawn from :math:`\\mathcal{N}(0, 1)` (the RBF
        spectral density in lengthscale-1 units); projections are drawn
        from :math:`\\mathcal{N}(0, \\sigma_W^2)`; lengthscales are
        broadcast from the scalar ``lengthscale`` argument.

        Args:
            in_features: Input dimension. Must be :math:`\\ge 1`.
            hidden_features: Hidden dimension. Must be :math:`\\ge 1`.
            out_features: Output dimension. Must be :math:`\\ge 1`.
            depth: Total stacked RFF layers (including readout). Must be
                :math:`\\ge 1`.
            key: PRNG key.
            n_features: Per-layer Fourier-feature pair count. Must be
                :math:`\\ge 1`.
            lengthscale: Initial value for each layer's lengthscale.
                Must be :math:`> 0`.
            prior_std: Standard deviation used to scale the initial
                projection weights. Must be :math:`> 0`.

        Returns:
            Initialised :class:`DeepVSSGPCore`.

        Raises:
            ValueError: If ``depth``, any feature dimension, or
                ``n_features`` is :math:`< 1`, or if ``lengthscale`` /
                ``prior_std`` is :math:`\\le 0`.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.vssgp import DeepVSSGPCore
            >>> core = DeepVSSGPCore.init(
            ...     in_features=2,
            ...     hidden_features=4,
            ...     out_features=1,
            ...     depth=3,
            ...     key=jr.PRNGKey(0),
            ...     n_features=8,
            ... )
            >>> core.W_freqs[0].shape  # (in_features, n_features)
            (2, 8)
            >>> core.W_projs[0].shape  # (2 * n_features, hidden_features)
            (16, 4)
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}.")
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0, got {prior_std}.")
        for name, dim in (
            ("in_features", in_features),
            ("hidden_features", hidden_features),
            ("out_features", out_features),
        ):
            if dim < 1:
                raise ValueError(f"{name} must be >= 1, got {dim}.")

        W_freqs: list[Float[Array, "d_in n_features"]] = []
        W_projs: list[Float[Array, "d_rff d_out"]] = []
        for layer_idx in range(depth):
            in_dim = in_features if layer_idx == 0 else hidden_features
            out_dim = out_features if layer_idx == depth - 1 else hidden_features
            key, k_freq, k_proj = jax.random.split(key, 3)
            W_freqs.append(jax.random.normal(k_freq, (in_dim, n_features)))
            W_projs.append(
                prior_std * jax.random.normal(k_proj, (2 * n_features, out_dim))
            )
        lengthscales = jnp.full((depth,), lengthscale)
        return cls(
            W_freqs=W_freqs,
            W_projs=W_projs,
            lengthscales=lengthscales,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            n_features=n_features,
            depth=depth,
        )

    def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, " D_out"]:
        r"""Compose the ``depth`` RFF-then-project layers on a single input.

        Each layer maps ``z -> Φ_l(z) -> z W_l`` so the running activation
        flows ``in_features -> hidden_features -> ... -> out_features``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.vssgp import DeepVSSGPCore
            >>> core = DeepVSSGPCore.init(
            ...     in_features=3,
            ...     hidden_features=5,
            ...     out_features=2,
            ...     depth=1,
            ...     key=jr.PRNGKey(1),
            ...     n_features=4,
            ... )
            >>> core(jnp.ones(3)).shape
            (2,)
        """
        z = x
        for layer_idx in range(self.depth):
            W_freq = self.W_freqs[layer_idx]
            W_proj = self.W_projs[layer_idx]
            ls = self.lengthscales[layer_idx]
            # Φ_l(z): (d_in_l,) -> (2 * n_features,)
            phi = rff_forward(W_freq, ls, self.n_features, z)
            # project: (2 * n_features,)·(2 * n_features, d_out_l) -> (d_out_l,)
            z = einx.dot("f, f o -> o", phi, W_proj)
        return z


__all__ = ["DeepVSSGPCore"]
