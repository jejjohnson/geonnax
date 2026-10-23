r"""The Bayesian Neural Field network (Saad et al., 2024) as a deterministic core.

`NeuralFieldMLP` is the spatiotemporal network of Google's BayesNF without its
priors: per-input learned scales, gated feature blocks (rescaled inputs,
dyadic Fourier features, seasonal features of a time column, pairwise
interactions and an optional pre-encoded block such as spherical-harmonic
features), a learned mix of ``elu`` and ``tanh``, NTK-scaled dense layers with
learned gains, and a gained linear readout. It acts on one point
``(D,) -> (K,)``; batch with `jax.vmap`.

Its named fields are the contract pyrox-nn's ``BayesianNeuralField`` swaps for
NumPyro sites (Logistic(0, 1) priors in the original model): renaming one is a
breaking change.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from geonnax.basis import fourier_features, interaction_features, seasonal_features
from geonnax.geo import _promote_to_floating


# softplus(_UNIT_GAIN) == 1, so a fresh model applies unit gains everywhere.
_UNIT_GAIN = math.log(math.e - 1.0)


def _num_seasonal_features(
    periods: tuple[float, ...], harmonics: tuple[int, ...]
) -> int:
    # seasonal_features returns 2 columns per (period, harmonic) pair.
    return 2 * sum(harmonics) if periods else 0


class NeuralFieldMLP(eqx.Module):
    r"""The BayesNF spatiotemporal network as a deterministic `equinox.Module`.

    For one input $x \in \mathbb{R}^D$ with fixed scales $s$ and learned
    log-scale adjustments $a$, the rescaled input is
    $\tilde x = x \oslash (s \odot e^{a})$. Feature blocks $F_k$ are built from
    it: $\tilde x$ itself, dyadic Fourier features of each column with a
    positive degree (`geonnax.basis.fourier_features`, ``rescale=True``),
    seasonal features of the **raw** time column ``x[time_col]``
    (`geonnax.basis.seasonal_features`, ``rescale=True``), pairwise
    interaction products of $\tilde x$, and an optional ``extra`` block
    passed at call time (for example `SphericalHarmonicEncoder` features).
    Each block is gated by $\mathrm{softplus}(g_k)$ and the blocks are
    concatenated into $h_0$. With the activation
    $\sigma_\alpha(z) = \alpha\,\mathrm{elu}(z) + (1-\alpha)\tanh(z)$,
    $\alpha = \mathrm{sigmoid}(\beta)$:

    $$
    h_{\ell+1} = \sigma_\alpha\!\Big(\mathrm{softplus}(c_\ell)\,
    \big(W_\ell^\top h_\ell / \sqrt{|h_\ell|} + b_\ell\big)\Big),
    \qquad
    y = \mathrm{softplus}(c_{\text{out}})\,
    \big(W_{\text{out}}^\top h_L / \sqrt{|h_L|} + b_{\text{out}}\big).
    $$

    The $1/\sqrt{\text{fan-in}}$ factor is the NTK scaling of BayesNF, which
    makes the prior predictive fan-in independent. With ``extra=None`` the
    forward pass equals pyrox-nn's ``BayesianNeuralField`` given the same
    parameter values.

    Initialisation (deterministic training from scratch): dense weights
    $\mathcal{N}(0, 1)$ with the $1/\sqrt{\text{fan-in}}$ in the forward pass
    (the NTK parameterisation of Jacot et al., 2018; Lee et al., 2018), which
    keeps activations unit-scale across depth; zero biases and scale
    adjustments; every gain at $\mathrm{softplus}^{-1}(1)$; $\beta = 0$
    ($\alpha = 1/2$). This deliberately departs from BayesNF, whose MAP and
    ensemble fits start from Logistic(0, 1) prior draws (variance
    $\pi^2/3$); pyrox-nn restores that when it wraps the core.

    Named parameter fields (swapped by pyrox-nn for sample sites):
    ``log_scale_adjustment``, ``feature_gain``, ``logit_activation_weight``,
    ``layers_W``, ``layers_b``, ``layers_gain``, ``output_W``, ``output_b``,
    ``output_gain``.

    Attributes:
        log_scale_adjustment: Learned per-input log scale $a$, shape ``(D,)``.
        feature_gain: Gate logits $g_k$, one per feature block, shape ``(B,)``.
        logit_activation_weight: Activation-mix logit $\beta$, scalar.
        layers_W: Hidden weights, a tuple of ``depth`` arrays ``(I, width)``.
        layers_b: Hidden biases, a tuple of ``depth`` arrays ``(width,)``.
        layers_gain: Hidden-layer gain logits $c_\ell$, shape ``(depth,)``.
        output_W: Readout weights, shape ``(width, K)``.
        output_b: Readout bias, shape ``(K,)``.
        output_gain: Readout gain logit $c_{\text{out}}$, scalar.
        input_scales: Fixed per-input scales $s$ (often the training-data
            inter-quartile range).
        fourier_degrees: Dyadic Fourier degree per input column; ``0`` skips
            a column; empty for none.
        interactions: Index pairs for interaction features; empty for none.
        seasonality_periods: Periods of the seasonal features.
        num_seasonal_harmonics: Harmonics per period.
        time_col: Input column holding time for the seasonal features.
        extra_features: Width of the optional ``extra`` block; ``0`` for none.
        width: Hidden width.
        depth: Number of hidden layers.
        out_features: Output dimension $K$.

    References:
        Jacot, A., Gabriel, F., Hongler, C. (2018). Neural tangent kernel.
        *NeurIPS*. Lee, J., et al. (2018). Deep neural networks as Gaussian
        processes. *ICLR*.
        Saad, F. A., et al. (2024). Scalable spatiotemporal prediction with
        Bayesian neural fields. *Nature Communications* 15, 7942.
    """

    log_scale_adjustment: Float[Array, " D"]
    feature_gain: Float[Array, " B"]
    logit_activation_weight: Float[Array, ""]
    layers_W: tuple[Float[Array, "I O"], ...]
    layers_b: tuple[Float[Array, " O"], ...]
    layers_gain: Float[Array, " L"]
    output_W: Float[Array, "O K"]
    output_b: Float[Array, " K"]
    output_gain: Float[Array, ""]
    input_scales: tuple[float, ...] = eqx.field(static=True)
    fourier_degrees: tuple[int, ...] = eqx.field(static=True)
    interactions: tuple[tuple[int, int], ...] = eqx.field(static=True)
    seasonality_periods: tuple[float, ...] = eqx.field(static=True)
    num_seasonal_harmonics: tuple[int, ...] = eqx.field(static=True)
    time_col: int = eqx.field(static=True)
    extra_features: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        input_scales: tuple[float, ...],
        *,
        width: int,
        depth: int,
        out_features: int = 1,
        fourier_degrees: tuple[int, ...] = (),
        interactions: tuple[tuple[int, int], ...] = (),
        seasonality_periods: tuple[float, ...] = (),
        num_seasonal_harmonics: tuple[int, ...] = (),
        time_col: int = 0,
        extra_features: int = 0,
        key: PRNGKeyArray,
    ) -> NeuralFieldMLP:
        """Build a `NeuralFieldMLP` with the deterministic default initialisation.

        Args:
            input_scales: Positive per-input scales; their number sets ``D``.
            width: Hidden width (``>= 1``).
            depth: Number of hidden layers (``>= 0``).
            out_features: Output dimension (``>= 1``).
            fourier_degrees: Per-input dyadic Fourier degrees (``>= 0``);
                empty, or one per input.
            interactions: Pairs ``(i, j)`` of input indices.
            seasonality_periods: Positive periods for seasonal features.
            num_seasonal_harmonics: Harmonics per period, same length.
            time_col: Index of the time column in ``x``.
            extra_features: Width of the ``extra`` block given at call time.
            key: PRNG key for the dense weights.

        Returns:
            A new `NeuralFieldMLP`.

        Raises:
            ValueError: On an empty or non-positive ``input_scales``, bad
                sizes, mismatched lengths, out-of-range indices, or a
                negative degree.
        """
        input_scales = tuple(float(s) for s in input_scales)
        d = len(input_scales)
        if d == 0 or any(not math.isfinite(s) or s <= 0.0 for s in input_scales):
            raise ValueError(
                f"input_scales must be non-empty and positive; got {input_scales}."
            )
        if width < 1 or depth < 0 or out_features < 1 or extra_features < 0:
            raise ValueError(
                "need width >= 1, depth >= 0, out_features >= 1, extra_features >= 0; "
                f"got {width}, {depth}, {out_features}, {extra_features}."
            )
        fourier_degrees = tuple(int(k) for k in fourier_degrees)
        if fourier_degrees and len(fourier_degrees) != d:
            raise ValueError(
                f"fourier_degrees needs one entry per input ({d}); "
                f"got {len(fourier_degrees)}."
            )
        if any(k < 0 for k in fourier_degrees):
            raise ValueError(f"fourier_degrees must be >= 0; got {fourier_degrees}.")
        interactions = tuple((int(i), int(j)) for i, j in interactions)
        if any(not (0 <= i < d and 0 <= j < d) for i, j in interactions):
            raise ValueError(
                f"interactions index outside [0, {d}); got {interactions}."
            )
        seasonality_periods = tuple(float(p) for p in seasonality_periods)
        num_seasonal_harmonics = tuple(int(h) for h in num_seasonal_harmonics)
        if len(seasonality_periods) != len(num_seasonal_harmonics):
            raise ValueError(
                "seasonality_periods and num_seasonal_harmonics need the same length; "
                f"got {len(seasonality_periods)} and {len(num_seasonal_harmonics)}."
            )
        if any(p <= 0.0 for p in seasonality_periods) or any(
            h < 0 for h in num_seasonal_harmonics
        ):
            raise ValueError(
                "periods must be > 0 and harmonics >= 0; got "
                f"{seasonality_periods} and {num_seasonal_harmonics}."
            )
        if not 0 <= time_col < d:
            raise ValueError(f"time_col must be in [0, {d}); got {time_col}.")

        block_widths = [d]
        block_widths += [2 * k for k in fourier_degrees if k > 0]
        n_seasonal = _num_seasonal_features(seasonality_periods, num_seasonal_harmonics)
        if n_seasonal > 0:
            block_widths.append(n_seasonal)
        if interactions:
            block_widths.append(len(interactions))
        if extra_features > 0:
            block_widths.append(extra_features)

        keys = jax.random.split(key, depth + 1)
        fan_in = sum(block_widths)
        layers_W, layers_b = [], []
        for k in keys[:depth]:
            layers_W.append(jax.random.normal(k, (fan_in, width)))
            layers_b.append(jnp.zeros((width,)))
            fan_in = width
        return cls(
            log_scale_adjustment=jnp.zeros((d,)),
            feature_gain=jnp.full((len(block_widths),), _UNIT_GAIN),
            logit_activation_weight=jnp.zeros(()),
            layers_W=tuple(layers_W),
            layers_b=tuple(layers_b),
            layers_gain=jnp.full((depth,), _UNIT_GAIN),
            output_W=jax.random.normal(keys[depth], (fan_in, out_features)),
            output_b=jnp.zeros((out_features,)),
            output_gain=jnp.asarray(_UNIT_GAIN),
            input_scales=input_scales,
            fourier_degrees=fourier_degrees,
            interactions=interactions,
            seasonality_periods=seasonality_periods,
            num_seasonal_harmonics=num_seasonal_harmonics,
            time_col=time_col,
            extra_features=extra_features,
            width=width,
            depth=depth,
            out_features=out_features,
        )

    def features(
        self,
        x: Float[Array, " D"],
        extra: Float[Array, " E"] | None = None,
    ) -> Float[Array, " F"]:
        """Gated, concatenated feature blocks $h_0$ for one input.

        Args:
            x: One input point, shape ``(D,)``.
            extra: Pre-encoded features, shape ``(extra_features,)``;
                required exactly when ``extra_features > 0``.

        Returns:
            $h_0$, shape ``(F,)``.

        Raises:
            ValueError: If ``x`` or ``extra`` has the wrong shape.
        """
        d = len(self.input_scales)
        if x.shape != (d,):
            raise ValueError(f"x must have shape ({d},); got {x.shape}. Use jax.vmap.")
        if (extra is None) != (self.extra_features == 0):
            raise ValueError(
                f"extra must be given exactly when extra_features > 0 "
                f"(extra_features={self.extra_features})."
            )
        if extra is not None and extra.shape != (self.extra_features,):
            raise ValueError(
                f"extra must have shape ({self.extra_features},); got {extra.shape}."
            )

        x = _promote_to_floating(x)
        scales = jnp.asarray(self.input_scales, dtype=x.dtype)
        scaled = x / (scales * jnp.exp(self.log_scale_adjustment))
        row = einx.id("d -> 1 d", scaled)

        blocks = [scaled]
        for col, k in enumerate(self.fourier_degrees):
            if k > 0:
                blocks.append(fourier_features(row[:, col], k, rescale=True)[0])
        if _num_seasonal_features(
            self.seasonality_periods, self.num_seasonal_harmonics
        ):
            t = einx.id("d -> 1 d", x)[:, self.time_col]
            blocks.append(
                seasonal_features(
                    t,
                    self.seasonality_periods,
                    self.num_seasonal_harmonics,
                    rescale=True,
                )[0]
            )
        if self.interactions:
            pairs = jnp.asarray(self.interactions, dtype=jnp.int32)
            blocks.append(interaction_features(row, pairs)[0])
        if extra is not None:
            blocks.append(extra)

        gains = jax.nn.softplus(self.feature_gain)
        return jnp.concatenate([g * b for g, b in zip(gains, blocks, strict=True)])

    def __call__(
        self,
        x: Float[Array, " D"],
        extra: Float[Array, " E"] | None = None,
    ) -> Float[Array, " K"]:
        """Evaluate the field at one point.

        Args:
            x: One input point, shape ``(D,)``.
            extra: Pre-encoded features, shape ``(extra_features,)``, when the
                model was built with ``extra_features > 0``.

        Returns:
            Field value, shape ``(out_features,)``.

        Examples:
            >>> import jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax import NeuralFieldMLP
            >>> field = NeuralFieldMLP.init(
            ...     (365.0, 180.0, 90.0), width=32, depth=2,
            ...     fourier_degrees=(0, 3, 3), seasonality_periods=(365.25,),
            ...     num_seasonal_harmonics=(2,), key=jr.PRNGKey(0),
            ... )
            >>> field(jnp.array([10.0, 3.5, 40.0])).shape
            (1,)
            >>> jax.vmap(field)(jnp.ones((8, 3))).shape
            (8, 1)
        """
        h = self.features(x, extra)
        alpha = jax.nn.sigmoid(self.logit_activation_weight)
        for W, b, c in zip(self.layers_W, self.layers_b, self.layers_gain, strict=True):
            z = jax.nn.softplus(c) * (
                einx.dot("i, i o -> o", h / jnp.sqrt(h.shape[0]), W) + b
            )
            h = alpha * jax.nn.elu(z) + (1.0 - alpha) * jnp.tanh(z)
        out = einx.dot("i, i k -> k", h / jnp.sqrt(h.shape[0]), self.output_W)
        return jax.nn.softplus(self.output_gain) * (out + self.output_b)


__all__ = ["NeuralFieldMLP"]
