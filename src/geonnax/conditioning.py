"""Unified conditioning primitives for ``geonnax``.

A conditioner is a layer ``c(h, z) -> y`` that transforms an inner
activation ``h`` based on a context / latent code ``z``. Three concrete
conditioners cover the literature:

* :class:`ConcatConditioner` — ``y = Linear([h ‖ z])``. Cheapest baseline,
  parameter-heavy in ``cond_dim``.
* :class:`AffineModulation` (also exported as :data:`FiLM`) — feature-wise
  affine ``y = γ(z) ⊙ h + β(z)`` with a single ``eqx.nn.Linear`` generator.
  The ``gamma_activation="exp"`` mode exposes :meth:`AffineModulation.log_det`
  so flowjax-style code can use it as a bijection wrapper.
* :class:`HyperLinear` — full hypernetwork, ``(W, b) = g(z); y = W x + b``.
  Generator is a single ``eqx.nn.Linear`` of width
  ``target_out * target_in + target_out``.

The composite :class:`ConditionedINR` wraps any inner network exposing a
``layers`` sequence (e.g. :class:`geonnax.SIREN`) with a per-layer
conditioner. The :func:`HyperSIREN` constructor sugar builds the NIF
ShapeNet/ParameterNet composite (Pan, Brunton, Kutz — JMLR 2023) by
special-casing per-layer init-scale calibration so the generated weights
match Sitzmann's variance-preservation property.
"""

from __future__ import annotations

import math
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from geonnax.siren import SIREN, siren_W_limit


GammaActivation = Literal["one_plus_tanh", "exp", "softplus", "identity"]
ConditionedMode = Literal["input", "feature"]


_GAMMA_ACTIVATIONS: tuple[str, ...] = ("one_plus_tanh", "exp", "softplus", "identity")


def _apply_gamma(raw: Array, kind: str) -> Array:
    if kind == "one_plus_tanh":
        return 1.0 + jnp.tanh(raw)
    if kind == "exp":
        return jnp.exp(raw)
    if kind == "softplus":
        return jax.nn.softplus(raw)
    if kind == "identity":
        return raw
    raise ValueError(
        f"gamma_activation must be one of {_GAMMA_ACTIVATIONS}; got {kind!r}."
    )


def _broadcast_z(z: Array, n_rows: int) -> Array:
    """Broadcast a single context ``(K,)`` to ``(N, K)``; pass-through otherwise."""
    if z.ndim == 1:
        return einx.id("k -> n k", z, n=n_rows)
    return z


def _atleast_2d_pair(h: Array, z: Array) -> tuple[Array, Array, bool]:
    """Promote ``h``, ``z`` to 2D for the inner kernel; record whether to squeeze.

    Conditioners accept either ``(C,)`` or ``(N, C)`` for ``h`` and
    correspondingly ``(K,)`` or ``(N, K)`` for ``z``. Higher-rank batch
    shapes are rejected here with a clear error rather than silently
    misbroadcasting.
    """
    if h.ndim not in (1, 2):
        raise ValueError(
            f"Conditioners accept h of shape (C,) or (N, C); got h.ndim={h.ndim}. "
            "For higher-rank batches, flatten leading axes first."
        )
    if z.ndim not in (1, 2):
        raise ValueError(
            f"Conditioners accept z of shape (K,) or (N, K); got z.ndim={z.ndim}. "
            "For higher-rank batches, flatten leading axes first."
        )
    squeeze = h.ndim == 1
    if squeeze:
        h = einx.id("c -> 1 c", h)
    z = _broadcast_z(z, h.shape[0])
    return h, z, squeeze


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class AbstractConditioner(eqx.Module):
    """Duck-typed protocol for ``(h, z) -> y`` conditioning layers.

    Concrete subclasses share the contract ``__call__(h, z) -> Array``
    where ``h.shape[-1] == num_features`` and ``z.shape[-1] == cond_dim``.
    There is no ``abstractmethod`` enforcement — subclasses simply
    implement ``__call__``.

    Attributes:
        num_features: Output channel count, matching ``h.shape[-1]``.
        cond_dim: Latent / context dimension, matching ``z.shape[-1]``.
    """

    num_features: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)

    def __call__(self, h: Array, z: Array, /) -> Array:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concat conditioner: y = Linear([h ‖ z])
# ---------------------------------------------------------------------------


class ConcatConditioner(AbstractConditioner):
    """Concatenate ``h`` and ``z`` then apply a single ``Linear``.

    Cheapest, most expressive in principle, but parameter count grows
    linearly with ``cond_dim``: ``(num_features + cond_dim) * num_features
    + num_features`` (the bias). No init ceremony required — uses
    ``eqx.nn.Linear`` defaults.

    Attributes:
        proj: Linear projection ``R^{C+K} -> R^{C}``.
        num_features: Output channels ``C``.
        cond_dim: Context dimension ``K``.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> layer = ConcatConditioner.init(num_features=8, cond_dim=4, key=jr.key(0))
        >>> y = layer(jnp.ones((5, 8)), jnp.ones((5, 4)))
        >>> y.shape
        (5, 8)
    """

    proj: eqx.nn.Linear

    @classmethod
    def init(
        cls,
        num_features: int,
        cond_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> ConcatConditioner:
        """Build a :class:`ConcatConditioner` with default ``eqx.nn.Linear`` init.

        Args:
            num_features: Output channel count.
            cond_dim: Context dimension.
            key: PRNG key for the projection's init.

        Returns:
            Initialised :class:`ConcatConditioner`.

        Raises:
            ValueError: If ``num_features`` or ``cond_dim`` is non-positive.
        """
        if num_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "num_features and cond_dim must be positive; got "
                f"num_features={num_features}, cond_dim={cond_dim}."
            )
        proj = eqx.nn.Linear(num_features + cond_dim, num_features, key=key)
        return cls(
            num_features=num_features,
            cond_dim=cond_dim,
            proj=proj,
        )

    def __call__(
        self,
        h: Float[Array, "*batch C"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch C"]:
        if h.shape[-1] != self.num_features:
            raise ValueError(
                f"h.shape[-1]={h.shape[-1]} does not match "
                f"num_features={self.num_features}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        h2d, z2d, squeeze = _atleast_2d_pair(h, z)
        # Concatenate on the feature axis, then per-row Linear via vmap.
        cat = jnp.concatenate([h2d, z2d], axis=-1)
        out = jax.vmap(self.proj)(cat)
        return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Affine modulation (FiLM): y = γ(z) ⊙ h + β(z)
# ---------------------------------------------------------------------------


class AffineModulation(AbstractConditioner):
    r"""Feature-wise Linear Modulation (FiLM): ``y = γ(z) ⊙ h + β(z)``.

    A single ``eqx.nn.Linear`` of output size ``2 * num_features``
    produces the concatenated ``(raw_β, raw_γ)`` from the context vector.
    The two halves are split on the feature axis via
    :func:`einx.id` (no raw ``jnp.split``), then ``γ`` is passed
    through the chosen activation:

    * ``"one_plus_tanh"`` (default): ``γ = 1 + tanh(raw_γ)`` — identity at
      init when the generator's bias is zero. The choice that gives FiLM
      its "does nothing until trained" property.
    * ``"exp"``: ``γ = exp(raw_γ)`` — strictly positive, required for
      bijection use. In this mode :meth:`log_det` returns
      ``sum(raw_γ, axis=-1)``, the closed-form log-Jacobian of an
      element-wise scale.
    * ``"softplus"``: ``γ = softplus(raw_γ)`` — strictly positive, slower
      to leave the prior than ``exp``.
    * ``"identity"``: ``γ = raw_γ`` — no shape guarantee, rarely useful.

    Attributes:
        generator: Linear ``R^K -> R^{2C}``.
        num_features: Output channels ``C``.
        cond_dim: Context dimension ``K``.
        gamma_activation: Parameterisation of ``γ`` (see above).

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> film = AffineModulation.init(num_features=8, cond_dim=4, key=jr.key(0))
        >>> y = film(jnp.ones((5, 8)), jnp.ones((5, 4)))
        >>> y.shape
        (5, 8)
    """

    generator: eqx.nn.Linear
    gamma_activation: GammaActivation = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        num_features: int,
        cond_dim: int,
        *,
        key: PRNGKeyArray,
        gamma_activation: GammaActivation = "one_plus_tanh",
    ) -> AffineModulation:
        """Build :class:`AffineModulation` with the default 2-output Linear generator.

        Args:
            num_features: Output channel count.
            cond_dim: Context dimension.
            key: PRNG key for the generator's init.
            gamma_activation: Parameterisation of ``γ``; see the class docstring.

        Returns:
            Initialised :class:`AffineModulation`.

        Raises:
            ValueError: If ``num_features`` or ``cond_dim`` is non-positive,
                or if ``gamma_activation`` is not a recognised value.
        """
        if num_features <= 0 or cond_dim <= 0:
            raise ValueError(
                "num_features and cond_dim must be positive; got "
                f"num_features={num_features}, cond_dim={cond_dim}."
            )
        if gamma_activation not in _GAMMA_ACTIVATIONS:
            raise ValueError(
                f"gamma_activation must be one of {_GAMMA_ACTIVATIONS}; "
                f"got {gamma_activation!r}."
            )
        generator = eqx.nn.Linear(cond_dim, 2 * num_features, key=key)
        assert generator.bias is not None
        # Bias-only zero-init: with bias=0 and "one_plus_tanh", γ=1 and β=0
        # → the layer is identity at init (Perez et al. 2018 default).
        generator = eqx.tree_at(
            lambda m: m.bias,
            generator,
            jnp.zeros_like(generator.bias),
        )
        return cls(
            num_features=num_features,
            cond_dim=cond_dim,
            generator=generator,
            gamma_activation=gamma_activation,
        )

    def _gamma_beta(self, z: Array) -> tuple[Array, Array, Array]:
        """Compute ``(raw_γ, γ, β)`` from a context array of shape ``(N, K)``."""
        raw = jax.vmap(self.generator)(z)  # (N, 2C)
        # Split on the feature axis: first half = β, second half = raw_γ.
        # einx.id keeps the (two, c) split explicit and avoids jnp.split.
        split = einx.id("n (two c) -> two n c", raw, two=2)
        beta, raw_gamma = split[0], split[1]
        gamma = _apply_gamma(raw_gamma, self.gamma_activation)
        return raw_gamma, gamma, beta

    def __call__(
        self,
        h: Float[Array, "*batch C"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch C"]:
        if h.shape[-1] != self.num_features:
            raise ValueError(
                f"h.shape[-1]={h.shape[-1]} does not match "
                f"num_features={self.num_features}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        h2d, z2d, squeeze = _atleast_2d_pair(h, z)
        _raw_gamma, gamma, beta = self._gamma_beta(z2d)
        out = gamma * h2d + beta
        return out[0] if squeeze else out

    def log_det(
        self,
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, " *batch"]:
        """Sum of ``log γ`` across the feature axis.

        Only valid when ``gamma_activation="exp"`` — that's the only
        parameterisation for which ``log γ = raw_γ`` exactly. For other
        modes this raises :class:`NotImplementedError`; callers that need
        a generic Jacobian must compute it manually.

        Args:
            z: Context array of shape ``(N, K)`` or ``(K,)``.

        Returns:
            Log-determinant of the diagonal scaling, shape ``(N,)`` (or
            scalar for 1-D ``z``).

        Raises:
            NotImplementedError: If ``gamma_activation != "exp"``.
        """
        if self.gamma_activation != "exp":
            raise NotImplementedError(
                "log_det is only defined for gamma_activation='exp'; got "
                f"{self.gamma_activation!r}. Use exp parameterisation for "
                "bijection wrappers."
            )
        squeeze = z.ndim == 1
        z2d = einx.id("k -> 1 k", z) if squeeze else z
        raw_gamma, _gamma, _beta = self._gamma_beta(z2d)
        ldj = einx.sum("n [c]", raw_gamma)
        return ldj[0] if squeeze else ldj


#: Backwards-compatible alias for :class:`AffineModulation`.
FiLM = AffineModulation


# ---------------------------------------------------------------------------
# Hypernetwork: (W, b) = g(z); y = W x + b
# ---------------------------------------------------------------------------


class HyperLinear(AbstractConditioner):
    """Generate a target ``Linear``'s ``(W, b)`` from ``z``, then apply.

    A single ``eqx.nn.Linear`` of output size ``target_out * target_in +
    target_out`` produces the flat parameter vector for an ad-hoc linear
    layer; ``W`` and ``b`` are split out via :func:`einx.id`.
    The forward dispatches on ``z.ndim``:

    * ``z.shape == (K,)`` — *shared* path: one ``(W, b)`` generated and
      reused across every row of ``x``. Cheap (one small affine + one
      matmul).
    * ``z.shape == (N, K)`` — *per-sample* path: ``(W, b)`` generated for
      each row, applied via ``einx.dot``. Costs ``N * C * C_in``
      flops per call.

    The generator weight scale is multiplied by ``init_scale`` so the
    generated ``W`` magnitude starts small and the composite is near-zero
    at init. Default ``init_scale=0.1`` matches NIF (Pan et al. 2023).

    Attributes:
        generator: Linear ``R^K -> R^{C_out * C_in + C_out}``.
        target_in: Inner ``Linear``'s input dim ``C_in``.
        target_out: Inner ``Linear``'s output dim ``C_out`` (= num_features).
        cond_dim: Context dimension ``K``.
        num_features: Alias for ``target_out`` (satisfies the protocol).

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> hyper = HyperLinear.init(
        ...     target_in=4, target_out=8, cond_dim=3, key=jr.key(0)
        ... )
        >>> y_shared = hyper(jnp.ones((6, 4)), jnp.ones((3,)))
        >>> y_persample = hyper(jnp.ones((6, 4)), jnp.ones((6, 3)))
        >>> (y_shared.shape, y_persample.shape)
        ((6, 8), (6, 8))
    """

    generator: eqx.nn.Linear
    target_in: int = eqx.field(static=True)
    target_out: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        target_in: int,
        target_out: int,
        cond_dim: int,
        *,
        key: PRNGKeyArray,
        init_scale: float = 0.1,
    ) -> HyperLinear:
        """Build a :class:`HyperLinear` with a small-magnitude generator init.

        Args:
            target_in: Input dimension of the generated ``Linear``.
            target_out: Output dimension of the generated ``Linear``.
            cond_dim: Context dimension.
            key: PRNG key for generator init.
            init_scale: Multiplicative factor on the generator weights so
                the generated ``W`` stays small at init. Default ``0.1``.

        Returns:
            Initialised :class:`HyperLinear`.

        Raises:
            ValueError: If any of ``target_in``, ``target_out``,
                ``cond_dim``, or ``init_scale`` is non-positive.
        """
        if target_in <= 0 or target_out <= 0 or cond_dim <= 0:
            raise ValueError(
                "target_in, target_out, and cond_dim must all be positive; got "
                f"target_in={target_in}, target_out={target_out}, "
                f"cond_dim={cond_dim}."
            )
        if init_scale <= 0:
            raise ValueError(f"init_scale must be > 0; got {init_scale}.")
        flat_size = target_out * target_in + target_out
        gen = eqx.nn.Linear(cond_dim, flat_size, key=key)
        # Scale weights and zero the bias so the generated (W, b) are small at init.
        gen = eqx.tree_at(
            lambda m: m.weight,
            gen,
            gen.weight * init_scale,
        )
        gen = eqx.tree_at(
            lambda m: m.bias,
            gen,
            jnp.zeros_like(gen.bias),
        )
        return cls(
            num_features=target_out,
            cond_dim=cond_dim,
            generator=gen,
            target_in=target_in,
            target_out=target_out,
        )

    def _split_params(self, flat: Array) -> tuple[Array, Array]:
        """Split flat ``(out * in + out,)`` into ``W: (out, in)`` and ``b: (out,)``."""
        w_size = self.target_out * self.target_in
        flat_W, flat_b = flat[:w_size], flat[w_size:]
        W = einx.id(
            "(c c_in) -> c c_in", flat_W, c=self.target_out, c_in=self.target_in
        )
        return W, flat_b

    def __call__(
        self,
        x: Float[Array, "*batch C_in"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch C_out"]:
        if x.shape[-1] != self.target_in:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]} does not match target_in={self.target_in}."
            )
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        squeeze = x.ndim == 1
        if squeeze:
            x = einx.id("c -> 1 c", x)

        if z.ndim == 1:
            # Shared (W, b): generate once, reuse across all rows.
            flat = self.generator(z)
            W, b = self._split_params(flat)
            out = einx.dot("c c_in, n c_in -> n c", W, x) + b
        else:
            # Per-sample (W, b): generate per row of z, contract per row.
            flats = jax.vmap(self.generator)(z)  # (N, out*in + out)
            w_size = self.target_out * self.target_in
            flat_W = flats[:, :w_size]
            flat_b = flats[:, w_size:]
            W = einx.id(
                "n (c c_in) -> n c c_in",
                flat_W,
                c=self.target_out,
                c_in=self.target_in,
            )
            out = einx.dot("n c c_in, n c_in -> n c", W, x) + flat_b

        return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Composite: any inner-network + per-layer conditioning
# ---------------------------------------------------------------------------


class ConditionedINR(eqx.Module):
    """Wrap an inner network's per-layer activations with conditioners.

    Given an ``inner`` network exposing a ``layers`` sequence (true for
    :class:`geonnax.SIREN` and any module that holds a list of callables
    named ``layers``), :class:`ConditionedINR` runs the inner forward and
    inserts a conditioner after each non-readout layer:

    .. code:: text

        z_0 = layer_0(x)
        z_0 = cond_0(z_0, c)
        z_1 = layer_1(z_0)
        z_1 = cond_1(z_1, c)
        ...
        y   = layer_{L-1}(z_{L-2})        # readout, not conditioned

    The ``mode="input"`` shortcut concatenates ``c`` to the input
    *before* running ``inner`` — useful for inner networks that don't
    expose a ``layers`` sequence (e.g. plain ``eqx.nn.MLP`` instances).

    Conditioners must be ``AbstractConditioner`` instances whose
    ``num_features`` matches the corresponding ``inner`` layer's output
    width.

    Attributes:
        inner: Inner network with a ``layers`` attribute (for
            ``mode="feature"``) or any callable (for ``mode="input"``).
        conditioners: Per-layer conditioner list. Length equals
            ``len(inner.layers) - 1`` for ``"feature"`` (no readout
            conditioning) or 1 for ``"input"`` (a single
            ``ConcatConditioner``-style head).
        cond_dim: Context dimension shared by all conditioners.
        mode: ``"feature"`` for per-layer modulation;
            ``"input"`` for input-side concatenation.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> from geonnax import SIREN
        >>> key = jr.key(0)
        >>> inner = SIREN.init(2, 32, 1, depth=4, key=key)
        >>> wrapped = ConditionedINR.init(
        ...     inner, conditioner_cls=AffineModulation, cond_dim=4, key=key
        ... )
        >>> y = wrapped(jnp.zeros((10, 2)), jnp.zeros((10, 4)))
        >>> y.shape
        (10, 1)
    """

    inner: eqx.Module
    conditioners: list[AbstractConditioner]
    cond_dim: int = eqx.field(static=True)
    mode: ConditionedMode = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        inner: eqx.Module,
        *,
        conditioner_cls: type[AbstractConditioner],
        cond_dim: int,
        key: PRNGKeyArray,
        mode: ConditionedMode = "feature",
        **conditioner_kwargs: object,
    ) -> ConditionedINR:
        """Build a :class:`ConditionedINR` around ``inner``.

        Args:
            inner: Inner network. Must have ``layers: Sequence`` for
                ``mode="feature"``; any callable works for ``mode="input"``.
            conditioner_cls: One of :class:`ConcatConditioner`,
                :class:`AffineModulation`, or :class:`HyperLinear`.
            cond_dim: Context dimension passed to each conditioner.
            key: PRNG key, split internally for each conditioner.
            mode: ``"feature"`` (per-layer modulation, default) or
                ``"input"`` (single input-side concatenation).
            **conditioner_kwargs: Extra kwargs forwarded to each
                ``conditioner_cls.init``.

        Returns:
            Initialised :class:`ConditionedINR`.

        Raises:
            ValueError: If ``mode == "feature"`` and ``inner`` lacks a
                ``layers`` attribute, or if any layer is missing the
                ``out_features`` shape needed to size the conditioners.
        """
        if mode not in ("feature", "input"):
            raise ValueError(f"mode must be 'feature' or 'input'; got {mode!r}.")

        if mode == "input":
            keys = jr.split(key, 1)
            head = _build_conditioner(
                conditioner_cls,
                num_features=_inner_in_features(inner),
                cond_dim=cond_dim,
                key=keys[0],
                **conditioner_kwargs,
            )
            return cls(
                inner=inner,
                conditioners=[head],
                cond_dim=cond_dim,
                mode="input",
            )

        layers = getattr(inner, "layers", None)
        if layers is None:
            raise ValueError(
                "mode='feature' requires `inner` to expose a `layers` sequence "
                "(e.g. geonnax.SIREN). For inners without `layers`, use mode='input'."
            )
        if len(layers) < 2:
            raise ValueError(
                "mode='feature' needs at least two inner layers; got "
                f"{len(layers)}. Use mode='input' for shallow inners."
            )

        # One conditioner per non-readout layer (skip the last).
        n_cond = len(layers) - 1
        keys = jr.split(key, n_cond)
        conditioners: list[AbstractConditioner] = []
        for i, k in enumerate(keys):
            num_features = _layer_out_features(layers[i], i)
            conditioners.append(
                _build_conditioner(
                    conditioner_cls,
                    num_features=num_features,
                    cond_dim=cond_dim,
                    key=k,
                    **conditioner_kwargs,
                )
            )
        return cls(
            inner=inner,
            conditioners=conditioners,
            cond_dim=cond_dim,
            mode="feature",
        )

    def __call__(
        self,
        x: Float[Array, "*batch D_in"],
        z: Float[Array, "*batch K"] | Float[Array, " K"],
    ) -> Float[Array, "*batch D_out"]:
        if z.shape[-1] != self.cond_dim:
            raise ValueError(
                f"z.shape[-1]={z.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        if self.mode == "input":
            head = self.conditioners[0]
            x = head(x, z)
            return self.inner(x)  # ty: ignore[call-non-callable]

        # mode == "feature": run each non-readout inner layer, then condition.
        layers = self.inner.layers  # ty: ignore[unresolved-attribute]
        h = x
        for i, layer in enumerate(layers[:-1]):
            h = layer(h)
            h = self.conditioners[i](h, z)
        return layers[-1](h)


def _build_conditioner(
    cls: type[AbstractConditioner],
    *,
    num_features: int,
    cond_dim: int,
    key: PRNGKeyArray,
    **kwargs: object,
) -> AbstractConditioner:
    """Construct a conditioner, threading the key only for variants that need one."""
    if cls is HyperLinear:
        # HyperLinear needs (target_in, target_out, cond_dim).
        return HyperLinear.init(
            target_in=num_features,
            target_out=num_features,
            cond_dim=cond_dim,
            key=key,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
    return cls.init(  # ty: ignore[unresolved-attribute]
        num_features=num_features,
        cond_dim=cond_dim,
        key=key,
        **kwargs,
    )


def _layer_out_features(layer: object, index: int) -> int:
    """Best-effort extraction of a layer's output width."""
    for attr in ("out_features", "output_dim", "hidden_features"):
        v = getattr(layer, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError(
        f"Could not infer out_features for layer #{index} of type "
        f"{type(layer).__name__}. Expose an int `out_features` attribute."
    )


def _inner_in_features(inner: object) -> int:
    for attr in ("in_features", "input_dim"):
        v = getattr(inner, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError(
        "Could not infer in_features for the inner network. Expose an int "
        "`in_features` attribute or use mode='feature'."
    )


# ---------------------------------------------------------------------------
# Constructor sugar: NIF-style HyperSIREN
# ---------------------------------------------------------------------------


class GeneratedSiren(eqx.Module):
    """Wrapper around a SIREN whose layers consume generated weights.

    Built by :func:`HyperSIREN`. The ``parameter_net`` runs once on ``mu``
    per forward call to produce the latent ``z``; ``z`` then drives every
    per-layer :class:`HyperLinear`.
    """

    parameter_net: eqx.Module
    siren: SIREN
    hyper_layers: list[HyperLinear]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __call__(self, x: Array, mu: Array) -> Array:
        z = self.parameter_net(mu)  # ty: ignore[call-non-callable]
        squeeze = x.ndim == 1
        if squeeze:
            x = einx.id("c -> 1 c", x)
        h = x
        for siren_layer, hyper in zip(
            self.siren.layers, self.hyper_layers, strict=True
        ):
            pre = hyper(h, z)
            if siren_layer.layer_type == "last":
                h = pre
            else:
                h = jnp.sin(siren_layer.omega * pre)
        return h[0] if squeeze else h


def HyperSIREN(
    in_features: int,
    hidden_features: int,
    out_features: int,
    *,
    depth: int,
    cond_dim: int,
    parameter_net: eqx.Module,
    key: PRNGKeyArray,
    first_omega: float = 30.0,
    hidden_omega: float = 30.0,
    c: float = 6.0,
    init_scale: float = 0.1,
) -> GeneratedSiren:
    """NIF-style ShapeNet/ParameterNet composite (Pan, Brunton, Kutz — JMLR 2023).

    Builds a SIREN shape-net of the requested topology, then constructs a
    parallel list of :class:`HyperLinear` generators — one per SIREN layer
    — whose ``init_scale`` is calibrated per Sitzmann regime so the
    *expected magnitude* of each generated ``W`` matches the half-width
    of Sitzmann's :func:`geonnax.siren.siren_W_limit` at init.
    Without this calibration the ShapeNet's pre-activation variance is
    wrong and training is unstable.

    The user-supplied ``parameter_net`` runs once on ``mu`` per forward
    call to produce the latent ``z``; ``z`` then drives every per-layer
    :class:`HyperLinear`. ``parameter_net`` must be callable with signature
    ``(P,) -> (cond_dim,)``.

    Args:
        in_features: Coordinate dimension of the SIREN.
        hidden_features: Hidden width.
        out_features: Output dimension.
        depth: SIREN depth (must be ≥ 2).
        cond_dim: Latent dimension produced by ``parameter_net``.
        parameter_net: User-supplied callable ``(P,) -> (cond_dim,)``.
        key: PRNG key, split internally for the SIREN init and the hyper
            generators.
        first_omega: First-layer ``omega``.
        hidden_omega: Hidden-layer ``omega``.
        c: SIREN Theorem-1 constant.
        init_scale: Multiplicative factor applied on top of the per-regime
            calibration; default ``0.1`` matches NIF.

    Returns:
        A composite that takes ``(x, mu)`` and runs the NIF forward.

    Raises:
        ValueError: If ``depth < 2`` or any positive-only argument is
            non-positive.
    """
    if depth < 2:
        raise ValueError(f"depth must be >= 2 (first + last); got depth={depth}.")
    for name, v in {
        "in_features": in_features,
        "hidden_features": hidden_features,
        "out_features": out_features,
        "cond_dim": cond_dim,
        "first_omega": first_omega,
        "hidden_omega": hidden_omega,
        "c": c,
        "init_scale": init_scale,
    }.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0; got {v}.")

    siren_key, *hyper_keys = jr.split(key, 1 + depth)
    siren = SIREN.init(
        in_features,
        hidden_features,
        out_features,
        depth=depth,
        key=siren_key,
        first_omega=first_omega,
        hidden_omega=hidden_omega,
        c=c,
    )
    hyper_layers: list[HyperLinear] = []
    for i, layer in enumerate(siren.layers):
        layer_in = layer.in_features
        layer_out = layer.out_features
        # Calibrate so generated W magnitude ≈ Sitzmann's per-regime half-width.
        regime_limit = siren_W_limit(layer.layer_type, layer_in, layer.omega, c)
        per_layer_scale = init_scale * regime_limit / math.sqrt(max(cond_dim, 1))
        hyper_layers.append(
            HyperLinear.init(
                target_in=layer_in,
                target_out=layer_out,
                cond_dim=cond_dim,
                key=hyper_keys[i],
                init_scale=per_layer_scale,
            )
        )
    return GeneratedSiren(
        parameter_net=parameter_net,
        siren=siren,
        hyper_layers=hyper_layers,
        in_features=in_features,
        out_features=out_features,
        depth=depth,
    )


__all__ = [
    "AbstractConditioner",
    "AffineModulation",
    "ConcatConditioner",
    "ConditionedINR",
    "FiLM",
    "GeneratedSiren",
    "HyperLinear",
    "HyperSIREN",
]
