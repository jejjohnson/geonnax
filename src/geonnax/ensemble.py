"""Ensemble-style layers for ``geonnax``.

Deterministic ``equinox.Module`` cores for BatchEnsemble / Rank-1 BNN
architectures:

* `DenseRank1` — rank-1 ensemble dense layer (Wen et al., 2020;
  Dusenberry et al., 2020). A shared full-rank kernel $W$ plus
  per-ensemble-member rank-1 multiplicative perturbations $r_i$,
  $s_i$. The deterministic (BatchEnsemble) forward; a Bayesian
  wrapper in the consuming probabilistic library swaps $r, s$
  for sample sites.
* `LayerNormEnsemble` — per-ensemble-member LayerNorm. Required
  drop-in replacement for ``LayerNorm`` inside BatchEnsemble / Rank1
  architectures.
* `MultiHeadAttentionBE` — multi-head attention with
  BatchEnsemble per-member rank-1 perturbations on each of the four
  Q / K / V / O projections. Output gains a leading ensemble axis.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def _glorot_uniform(
    key: PRNGKeyArray, in_features: int, out_features: int
) -> Float[Array, "D_in D_out"]:
    """Glorot-uniform init for a shared dense kernel."""
    lim = math.sqrt(6.0 / (in_features + out_features))
    return jr.uniform(
        key,
        (in_features, out_features),
        minval=-lim,
        maxval=lim,
    )


def _rs_init(
    key: PRNGKeyArray,
    ensemble_size: int,
    feature_dim: int,
    init_scale: float,
) -> Float[Array, "M D"]:
    """Per-member rank-1-vector init: ``1 + init_scale * N(0, I)``.

    Centered on 1.0 so the rank-1 perturbation is the identity in
    expectation. ``init_scale`` controls per-member diversity.
    """
    return 1.0 + init_scale * jr.normal(key, (ensemble_size, feature_dim))


class Rank1ProjInit(NamedTuple):
    """Per-projection BatchEnsemble inits used by `MultiHeadAttentionBE`.

    Bundles the four arrays needed for one rank-1 projection
    ($W$ shared, $r, s, b$ per-member) into a single
    PyTree leaf so the parent module can carry one such NamedTuple
    per Q/K/V/O projection.

    Examples:
        >>> import jax.random as jr
        >>> proj = init_rank1_proj(
        ...     jr.PRNGKey(0), in_features=4, out_features=2,
        ...     ensemble_size=3, init_scale=0.5,
        ... )
        >>> isinstance(proj, Rank1ProjInit)
        True
        >>> proj.W.shape  # shared kernel (D_in, D_out)
        (4, 2)
    """

    W: Float[Array, "D_in D_out"]
    r: Float[Array, "M D_out"]
    s: Float[Array, "M D_in"]
    b: Float[Array, "M D_out"]


def init_rank1_proj(
    key: PRNGKeyArray,
    in_features: int,
    out_features: int,
    ensemble_size: int,
    init_scale: float,
) -> Rank1ProjInit:
    """Glorot-init shared kernel + per-member rank-1 vectors + zero bias.

    Examples:
        >>> import jax.random as jr
        >>> proj = init_rank1_proj(
        ...     jr.PRNGKey(0), in_features=4, out_features=2,
        ...     ensemble_size=3, init_scale=0.5,
        ... )
        >>> proj.r.shape, proj.s.shape  # (M, D_out), (M, D_in)
        ((3, 2), (3, 4))
    """
    kw, kr, ks = jr.split(key, 3)
    return Rank1ProjInit(
        W=_glorot_uniform(kw, in_features, out_features),
        r=_rs_init(kr, ensemble_size, out_features, init_scale),
        s=_rs_init(ks, ensemble_size, in_features, init_scale),
        b=jnp.zeros((ensemble_size, out_features)),
    )


def apply_rank1_proj(
    x: Float[Array, ...],
    proj: Rank1ProjInit,
    ensemble_size: int,
    bias: bool,
    *,
    has_ensemble: bool,
) -> Float[Array, ...]:
    r"""Apply ``y_i = ((x ⊙ s_i) @ W) ⊙ r_i + b_i`` per ensemble member.

    Two modes selected by the *explicit* ``has_ensemble`` flag rather
    than a shape heuristic — heuristics on ``x.shape[0] == ensemble_size``
    silently mis-classify inputs whose sequence length happens to equal
    the ensemble size (e.g. self-attention with ``T = M``):

    * ``has_ensemble=False``: ``x`` has shape ``(..., D_in)`` (any
      intrinsic-to-the-layer axes, e.g. a sequence axis, but no data
      batch axis) and the output gains a leading ``M`` axis:
      ``(M, ..., D_out)``. Use this for the Q/K/V projections, whose
      inputs are un-ensembled.
    * ``has_ensemble=True``: ``x`` already carries an ``M`` leading
      axis (``(M, ..., D_in)``); the per-member projection flows through
      unchanged. Use this for the O projection after attention has
      already added the ensemble axis.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> proj = init_rank1_proj(
        ...     jr.PRNGKey(0), in_features=4, out_features=2,
        ...     ensemble_size=3, init_scale=0.5,
        ... )
        >>> out = apply_rank1_proj(  # (D_in,) -> (M, D_out)
        ...     jnp.ones(4), proj, ensemble_size=3, bias=True,
        ...     has_ensemble=False,
        ... )
        >>> out.shape
        (3, 2)
    """
    if has_ensemble:
        if x.ndim < 2 or x.shape[0] != ensemble_size:
            raise ValueError(
                f"Expected x.shape[0] == ensemble_size ({ensemble_size}) when "
                f"has_ensemble=True; got x.shape = {x.shape}."
            )
        # x already carries the M axis: scale per member over batch dims.
        x_scaled = einx.multiply("m ... d, m d -> m ... d", x, proj.s)
    else:
        # x is un-ensembled: the M axis is introduced by the per-member scale.
        x_scaled = einx.multiply("... d, m d -> m ... d", x, proj.s)
    h = einx.dot("m ... d, d o -> m ... o", x_scaled, proj.W)
    out = einx.multiply("m ... o, m o -> m ... o", h, proj.r)
    if bias:
        out = einx.add("m ... o, m o -> m ... o", out, proj.b)
    return out


class DenseRank1(eqx.Module):
    r"""Rank-1 ensemble dense layer (deterministic BatchEnsemble core).

    Implements the BatchEnsemble (Wen et al., 2020) / rank-1 BNN
    (Dusenberry et al., 2020) parameterization: a single shared kernel
    $W \in \mathbb{R}^{D_\mathrm{in} \times D_\mathrm{out}}$ and
    per-member rank-1 multiplicative perturbations
    $s_i \in \mathbb{R}^{D_\mathrm{in}}$,
    $r_i \in \mathbb{R}^{D_\mathrm{out}}$ for
    $i = 1, \ldots, M$. The per-member effective weight is

    $$
    W_i = (s_i \otimes r_i) \circ W,
    $$


    and the efficient forward pass avoids materialising $W_i$:

    $$
    y_i = \bigl((x \circ s_i)\, W\bigr) \circ r_i + b_i.
    $$


    This is the deterministic core. Per-member diversity comes purely
    from the random initialisation of $r_i, s_i$. A Bayesian
    wrapper in the consuming probabilistic library may swap
    $r, s$ for sample sites with Normal priors centered at the
    per-member init values to recover the rank-1 BNN of
    Dusenberry et al. (2020).

    Attributes:
        W: Shared kernel of shape ``(D_in, D_out)``.
        r: Per-member output-side perturbation of shape ``(M, D_out)``.
        s: Per-member input-side perturbation of shape ``(M, D_in)``.
        b: Per-member bias of shape ``(M, D_out)``. Always present as
            an array (zeros) but only added when ``bias=True``.
        in_features: Input dimension $D_\mathrm{in}$.
        out_features: Output dimension $D_\mathrm{out}$.
        ensemble_size: Number of ensemble members $M$.
        bias: Whether to add the per-member bias in the forward.

    Examples:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> layer = DenseRank1.init(
        ...     jr.PRNGKey(0),
        ...     in_features=4,
        ...     out_features=2,
        ...     ensemble_size=3,
        ... )
        >>> x = jnp.ones(4)
        >>> y = layer(x)
        >>> y.shape
        (3, 2)

    References:
        Wen, Y., Tran, D., & Ba, J. (2020). *BatchEnsemble: An
        Alternative Approach to Efficient Ensemble and Lifelong
        Learning.* ICLR.

        Dusenberry, M. W., et al. (2020). *Efficient and Scalable
        Bayesian Neural Nets with Rank-1 Factors.* ICML.
    """

    W: Float[Array, "D_in D_out"]
    r: Float[Array, "M D_out"]
    s: Float[Array, "M D_in"]
    b: Float[Array, "M D_out"]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True, default=True)

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        *,
        bias: bool = True,
        init_scale: float = 0.5,
    ) -> DenseRank1:
        """Construct a layer with random per-member init vectors."""
        if in_features <= 0 or out_features <= 0 or ensemble_size <= 0:
            raise ValueError(
                "in_features, out_features, ensemble_size must all be > 0; "
                f"got {in_features=}, {out_features=}, {ensemble_size=}."
            )
        if init_scale < 0:
            raise ValueError(f"init_scale must be >= 0; got {init_scale}.")
        kw, kr, ks = jr.split(key, 3)
        W = _glorot_uniform(kw, in_features, out_features)
        r = _rs_init(kr, ensemble_size, out_features, init_scale)
        s = _rs_init(ks, ensemble_size, in_features, init_scale)
        b = jnp.zeros((ensemble_size, out_features))
        return cls(
            W=W,
            r=r,
            s=s,
            b=b,
            in_features=in_features,
            out_features=out_features,
            ensemble_size=ensemble_size,
            bias=bias,
        )

    def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, "M D_out"]:
        # yᵢ = ((x ∘ sᵢ) W) ∘ rᵢ + bᵢ. Single-example forward — the M axis
        # is intrinsic to the BatchEnsemble layer and is introduced by the
        # per-member input-scaling step. The user vmaps over a data axis.
        # Avoids materialising the per-member effective kernel
        # Wᵢ = (sᵢ ⊗ rᵢ) ∘ W and the manual reshapes that broadcasting
        # rᵢ / bᵢ would otherwise need.
        #   x:        (D_in,)
        #   x_scaled: (M, D_in)
        #   h, out:   (M, D_out)
        x_scaled = einx.multiply("d, m d -> m d", x, self.s)
        h = einx.dot("m d, d o -> m o", x_scaled, self.W)
        out = einx.multiply("m o, m o -> m o", h, self.r)
        if self.bias:
            out = einx.add("m o, m o -> m o", out, self.b)
        return out


class LayerNormEnsemble(eqx.Module):
    r"""Per-ensemble-member LayerNorm (deterministic core).

    Drop-in replacement for ``LayerNorm`` inside BatchEnsemble / Rank1
    architectures. Computes the standard LayerNorm normalisation over
    the trailing feature dimension and applies a *per-member* affine
    transform — each ensemble member $i \in \{1, \ldots, M\}$
    gets its own learnable scale $\gamma_i \in \mathbb{R}^D$
    and bias $\beta_i \in \mathbb{R}^D$:

    $$
    \hat{x}_i = \frac{x_i - \mu(x_i)}{\sqrt{\sigma^2(x_i) + \epsilon}},
    \qquad
    y_i = \gamma_i \odot \hat{x}_i + \beta_i,
    $$


    where $\mu$ and $\sigma^2$ are the empirical mean and
    variance over the trailing feature axis (computed independently
    for each member-batch slice). Without per-member scale/bias,
    sharing a single LayerNorm across the ensemble would couple all
    members and erase the diversity introduced by `DenseRank1`
    or any other BatchEnsemble layer upstream.

    Input is expected to carry a leading ensemble axis of size
    ``ensemble_size`` — that axis is intrinsic to BatchEnsemble layers,
    not a data batch — followed by a trailing feature axis of size
    ``feature_dim``. The user ``jax.vmap`` s over any data batch axis.

    Attributes:
        scales: Per-member scale of shape ``(M, D)``.
        biases: Per-member bias of shape ``(M, D)``.
        ensemble_size: Number of ensemble members $M$.
        feature_dim: Trailing feature dimension $D$ over which
            the normalisation is computed.
        eps: Small positive constant added to the variance for
            numerical stability.

    Examples:
        >>> import jax.numpy as jnp
        >>> ln = LayerNormEnsemble.init(ensemble_size=3, feature_dim=4)
        >>> x = jnp.ones((3, 4))  # (M, D)
        >>> y = ln(x)
        >>> y.shape
        (3, 4)
    """

    scales: Float[Array, "M D"]
    biases: Float[Array, "M D"]
    ensemble_size: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-5)

    @classmethod
    def init(
        cls,
        ensemble_size: int,
        feature_dim: int,
        *,
        eps: float = 1e-5,
    ) -> LayerNormEnsemble:
        """Construct a layer with default unit-scale, zero-bias inits."""
        if ensemble_size <= 0:
            raise ValueError(f"ensemble_size must be > 0; got {ensemble_size}.")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0; got {feature_dim}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}.")
        scales = jnp.ones((ensemble_size, feature_dim))
        biases = jnp.zeros((ensemble_size, feature_dim))
        return cls(
            scales=scales,
            biases=biases,
            ensemble_size=ensemble_size,
            feature_dim=feature_dim,
            eps=eps,
        )

    def __call__(self, x: Float[Array, "M D"]) -> Float[Array, "M D"]:
        if x.ndim != 2:
            raise ValueError(
                f"x must have shape (M, D); got shape {x.shape}. "
                "vmap over any data batch axis."
            )
        if x.shape[0] != self.ensemble_size:
            raise ValueError(
                f"x.shape[0] = {x.shape[0]} does not match "
                f"ensemble_size = {self.ensemble_size}."
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"x.shape[-1] = {x.shape[-1]} does not match "
                f"feature_dim = {self.feature_dim}."
            )

        # Per-member mean/var over the trailing feature axis.
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) * jax.lax.rsqrt(var + self.eps)

        # Per-member affine.
        #   x_hat: (M, D)   γ, β: (M, D)
        scaled = einx.multiply("m d, m d -> m d", x_hat, self.scales)
        return einx.add("m d, m d -> m d", scaled, self.biases)


class MultiHeadAttentionBE(eqx.Module):
    r"""Multi-head attention with BatchEnsemble rank-1 projections.

    Standard scaled-dot-product multi-head attention where each of the
    four linear projections — query, key, value, and output — uses a
    BatchEnsemble parameterisation: a shared full-rank kernel plus
    per-ensemble-member rank-1 multiplicative perturbations. So for
    member $i \in \{1, \ldots, M\}$ and projection
    $P \in \{Q, K, V, O\}$,

    $$
    W_i^{(P)} = (s_i^{(P)} \otimes r_i^{(P)}) \circ W^{(P)},
    $$


    and the attention itself is the usual

    $$
    \mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!
    \Bigl(\frac{Q K^\top}{\sqrt{d_k}}\Bigr) V.
    $$


    The forward consumes un-ensembled inputs (``query``, ``key``,
    ``value`` of shape ``(T, D)`` / ``(S, D)``), adds the ensemble
    axis when projecting to ``Q``, ``K``, ``V``, runs per-member
    attention in parallel, and returns the per-member output of
    shape ``(M, T, D)``. Equivalent to running ``M`` independent
    attention heads with rank-1 weight perturbations and stacking
    their outputs.

    Attributes:
        q_proj / k_proj / v_proj / o_proj: Per-projection
            BatchEnsemble arrays bundled as `Rank1ProjInit`.
        embed_dim: Total feature dimension $D$ of query / key /
            value (must be divisible by ``num_heads``).
        num_heads: Number of attention heads $H$. Each head sees
            ``embed_dim // num_heads`` features.
        ensemble_size: Number of ensemble members $M$.
        bias: Whether each of the four projections adds a per-member
            bias.

    Examples:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> mha = MultiHeadAttentionBE.init(
        ...     jr.PRNGKey(0),
        ...     embed_dim=8, num_heads=2, ensemble_size=3,
        ... )
        >>> x = jnp.ones((5, 8))
        >>> y = mha(x, x, x)         # self-attention
        >>> y.shape
        (3, 5, 8)
    """

    q_proj: Rank1ProjInit
    k_proj: Rank1ProjInit
    v_proj: Rank1ProjInit
    o_proj: Rank1ProjInit
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    ensemble_size: int = eqx.field(static=True)
    bias: bool = eqx.field(static=True, default=True)

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        embed_dim: int,
        num_heads: int,
        ensemble_size: int,
        *,
        bias: bool = True,
        init_scale: float = 0.5,
    ) -> MultiHeadAttentionBE:
        """Construct an MHA-BE layer with random Q/K/V/O projection inits."""
        if embed_dim <= 0 or num_heads <= 0 or ensemble_size <= 0:
            raise ValueError(
                "embed_dim, num_heads, ensemble_size must all be > 0; "
                f"got {embed_dim=}, {num_heads=}, {ensemble_size=}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if init_scale < 0:
            raise ValueError(f"init_scale must be >= 0; got {init_scale}.")
        kq, kk, kv, ko = jr.split(key, 4)
        return cls(
            q_proj=init_rank1_proj(kq, embed_dim, embed_dim, ensemble_size, init_scale),
            k_proj=init_rank1_proj(kk, embed_dim, embed_dim, ensemble_size, init_scale),
            v_proj=init_rank1_proj(kv, embed_dim, embed_dim, ensemble_size, init_scale),
            o_proj=init_rank1_proj(ko, embed_dim, embed_dim, ensemble_size, init_scale),
            embed_dim=embed_dim,
            num_heads=num_heads,
            ensemble_size=ensemble_size,
            bias=bias,
        )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def __call__(
        self,
        query: Float[Array, "T D"],
        key: Float[Array, "S D"],
        value: Float[Array, "S D"],
    ) -> Float[Array, "M T D"]:
        if query.ndim != 2 or query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"query must be (T, embed_dim={self.embed_dim}); got {query.shape}."
            )
        if key.ndim != 2 or key.shape[-1] != self.embed_dim:
            raise ValueError(
                f"key must be (S, embed_dim={self.embed_dim}); got {key.shape}."
            )
        if value.shape != key.shape:
            raise ValueError(
                f"key and value must share shape; got {key.shape} vs {value.shape}."
            )

        M = self.ensemble_size
        H = self.num_heads
        d = self.head_dim

        # Project inputs (no ensemble axis on input → M added on output).
        Q = apply_rank1_proj(
            query, self.q_proj, M, self.bias, has_ensemble=False
        )  # (M, T, D)
        K = apply_rank1_proj(
            key, self.k_proj, M, self.bias, has_ensemble=False
        )  # (M, S, D)
        V = apply_rank1_proj(
            value, self.v_proj, M, self.bias, has_ensemble=False
        )  # (M, S, D)

        # Per-head split + transpose in one shot: (M, ·, H·d) → (M, H, ·, d).
        Q = einx.id("m t (h d) -> m h t d", Q, h=H)  # (M, H, T, d)
        K = einx.id("m s (h d) -> m h s d", K, h=H)  # (M, H, S, d)
        V = einx.id("m s (h d) -> m h s d", V, h=H)  # (M, H, S, d)

        scores = einx.dot("m h t d, m h s d -> m h t s", Q, K) / math.sqrt(d)
        weights = jax.nn.softmax(scores, axis=-1)
        attn = einx.dot("m h t s, m h s d -> m h t d", weights, V)  # (M, H, T, d)

        # Merge heads back to (M, T, embed_dim): transpose + reshape in one shot.
        attn = einx.id("m h t d -> m t (h d)", attn)

        # Output projection: input already has the M axis.
        return apply_rank1_proj(attn, self.o_proj, M, self.bias, has_ensemble=True)


__all__ = [
    "DenseRank1",
    "LayerNormEnsemble",
    "MultiHeadAttentionBE",
    "Rank1ProjInit",
    "apply_rank1_proj",
    "init_rank1_proj",
]
