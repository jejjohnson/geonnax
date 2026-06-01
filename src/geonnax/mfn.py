"""Multiplicative Filter Networks (Fathony et al., ICLR 2021).

Deterministic ``equinox.Module`` cores. The Bayesian variants
``BayesianFourierNet`` / ``BayesianGaborNet`` live in the consuming
probabilistic library as Tier-D wrappers that swap each filter's
``Omega``/``phi`` (and ``mu``/``log_gamma``) and each readout linear
for ``pyrox_sample`` sites.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array as JaxArray
from jaxtyping import Array, Float, PRNGKeyArray


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` if any keyword value is non-positive."""
    for name, v in values.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}.")


class FourierFilter(eqx.Module):
    r"""Single Fourier filter: :math:`g(x) = \sin(\Omega x + \varphi)`.

    One multiplicative filter primitive for use inside a
    :class:`FourierNet`.

    Init follows Fathony et al. (2021) §4.1: frequencies are drawn as
    :math:`\Omega_{ij} \sim \mathcal{N}(0,\,\sigma_f^2/D)` where
    :math:`D` is ``in_features`` and :math:`\sigma_f` is
    ``freq_scale``; phases are drawn as
    :math:`\varphi_i \sim \mathrm{Uniform}(-\pi, \pi)`.

    Attributes:
        Omega: Frequency matrix of shape ``(out_features, in_features)``.
        phi: Phase vector of shape ``(out_features,)``.
        in_features: Input dimension.
        out_features: Output (filter) dimension.
    """

    Omega: Float[Array, "out in"]
    phi: Float[Array, " out"]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        freq_scale: float = 256.0,
    ) -> FourierFilter:
        """Construct with Fathony-et-al. §4.1 initialization."""
        _require_positive(
            in_features=in_features,
            out_features=out_features,
            freq_scale=freq_scale,
        )
        k_omega, k_phi = jax.random.split(key)
        omega_std = freq_scale / math.sqrt(in_features)
        Omega = jax.random.normal(k_omega, (out_features, in_features)) * omega_std
        phi = jax.random.uniform(k_phi, (out_features,), minval=-jnp.pi, maxval=jnp.pi)
        return cls(
            Omega=Omega,
            phi=phi,
            in_features=in_features,
            out_features=out_features,
        )

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N H"]:
        proj = einx.dot("n d, h d -> n h", jnp.atleast_2d(x), self.Omega)
        return jnp.sin(proj + self.phi)


class GaborFilter(eqx.Module):
    r"""Single Gabor filter:
    :math:`g(x) = \sin(\Omega x + \varphi) \odot \exp(-\tfrac{\gamma}{2}\|x - \mu\|^2)`.

    Init follows Fathony et al. (2021) §4.2: per-filter
    :math:`\gamma_i \sim \mathrm{Gamma}(\alpha, \beta)`,
    :math:`\mu_i \sim \mathrm{Uniform}(\text{domain})`,
    :math:`\Omega_{i,:} \sim \mathcal{N}(0, \gamma_i\,I_D)` (the
    load-bearing tied initialization).

    :math:`\gamma` is stored in log space so positivity is preserved
    without optimizer constraints.

    Attributes:
        Omega: Frequency matrix ``(out_features, in_features)``.
        phi: Phase vector ``(out_features,)``.
        mu: Envelope centres ``(out_features, in_features)``.
        log_gamma: Log-bandwidth ``(out_features,)``.
        in_features: Input dimension.
        out_features: Output (filter) dimension.
        domain: ``(low, high)`` used for :math:`\mu` initialization (static).
    """

    Omega: Float[Array, "out in"]
    phi: Float[Array, " out"]
    mu: Float[Array, "out in"]
    log_gamma: Float[Array, " out"]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    domain: tuple[float, float] = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        domain: tuple[float, float] = (-1.0, 1.0),
        gamma_alpha: float = 6.0,
        gamma_beta: float = 1.0,
    ) -> GaborFilter:
        """Construct with Fathony-et-al. §4.2 initialization."""
        _require_positive(
            in_features=in_features,
            out_features=out_features,
            gamma_alpha=gamma_alpha,
            gamma_beta=gamma_beta,
        )
        if domain[0] >= domain[1]:
            raise ValueError(f"domain must satisfy low < high; got domain={domain}.")
        k_gamma, k_mu, k_omega, k_phi = jax.random.split(key, 4)
        # jax.random.gamma samples Gamma(alpha, 1); divide by beta for rate=beta.
        gamma = jax.random.gamma(k_gamma, gamma_alpha, (out_features,)) / gamma_beta
        log_gamma = jnp.log(gamma)
        mu = jax.random.uniform(
            k_mu, (out_features, in_features), minval=domain[0], maxval=domain[1]
        )
        # Tied init: Omega_i ~ N(0, gamma_i * I_D).
        Omega = jax.random.normal(k_omega, (out_features, in_features)) * jnp.sqrt(
            gamma[:, None]
        )
        phi = jax.random.uniform(k_phi, (out_features,), minval=-jnp.pi, maxval=jnp.pi)
        return cls(
            Omega=Omega,
            phi=phi,
            mu=mu,
            log_gamma=log_gamma,
            in_features=in_features,
            out_features=out_features,
            domain=domain,
        )

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N H"]:
        x2d = jnp.atleast_2d(x)
        gamma = jnp.exp(self.log_gamma)
        x_norm_sq = jnp.sum(x2d**2, axis=-1, keepdims=True)
        mu_norm_sq = jnp.sum(self.mu**2, axis=-1)[None, :]
        cross = einx.dot("n d, h d -> n h", x2d, self.mu)
        sq_dist = jnp.maximum(x_norm_sq + mu_norm_sq - 2.0 * cross, 0.0)
        envelope = jnp.exp(-0.5 * gamma[None, :] * sq_dist)
        sinusoidal = jnp.sin(einx.dot("n d, h d -> n h", x2d, self.Omega) + self.phi)
        return sinusoidal * envelope


def mfn_forward(
    x: Float[Array, "N D"],
    filters: Sequence[Callable[[JaxArray], JaxArray]],
    linears: Sequence[Callable[[JaxArray], JaxArray]],
) -> Float[Array, "N O"]:
    """Pure-JAX MFN forward pass given user-supplied filter and linear callables.

    Implements the Fathony et al. (2021) multiplicative chaining:

    .. math::

        z_1 = g_1(x), \\quad
        z_{i+1} = g_{i+1}(x) \\odot (W_i z_i + b_i), \\quad
        y = W_L z_L + b_L.

    Exists as an escape hatch so users can plug custom filter families
    into the MFN topology without subclassing :class:`FourierNet` or
    :class:`GaborNet`.

    ``filters`` and ``linears`` must have the same length :math:`L`.
    ``linears`` are treated as single-sample callables and are
    :func:`jax.vmap`-ed over the batch dimension internally.
    :class:`FourierFilter` / :class:`GaborFilter` instances handle
    batched input natively.
    """
    if len(filters) == 0 or len(linears) == 0:
        raise ValueError(
            f"filters and linears must be non-empty; got lengths "
            f"{len(filters)} and {len(linears)}."
        )
    if len(filters) != len(linears):
        raise ValueError(
            f"filters and linears must have equal length; got "
            f"{len(filters)} and {len(linears)}."
        )
    x = jnp.atleast_2d(x)
    z = filters[0](x)
    for f, lin in zip(filters[1:], linears[:-1], strict=True):
        z = f(x) * jax.vmap(lin)(z)
    return jax.vmap(linears[-1])(z)


class FourierNet(eqx.Module):
    r"""Multiplicative Fourier Filter Network (Fathony et al., ICLR 2021).

    Chains :class:`FourierFilter` primitives multiplicatively:

    .. math::

        z_1 = g_1(x), \quad
        z_{i+1} = g_{i+1}(x) \odot (W_i z_i + b_i), \quad
        y = W_L z_L + b_L.

    Each :math:`g_i` is a :class:`FourierFilter` of width
    ``hidden_features``; the last linear is the readout projecting to
    ``out_features``.

    Note:
        Single-point input ``(D,)`` is automatically promoted to
        ``(1, D)`` and the result is squeezed back to ``(O,)``.

    Attributes:
        filters: Length-``depth`` list of :class:`FourierFilter`.
        linears: Length-``depth`` list of :class:`equinox.nn.Linear`.
        in_features: Input dimension.
        hidden_features: Filter / hidden width.
        out_features: Output dimension.
        depth: Number of filter layers :math:`L`.
    """

    filters: list[FourierFilter]
    linears: list[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: PRNGKeyArray,
        freq_scale: float = 256.0,
    ) -> FourierNet:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}.")
        keys = jax.random.split(key, 2 * depth)
        filter_keys = keys[:depth]
        linear_keys = keys[depth:]
        filters = [
            FourierFilter.init(
                in_features, hidden_features, key=filter_keys[i], freq_scale=freq_scale
            )
            for i in range(depth)
        ]
        linears = [
            eqx.nn.Linear(
                hidden_features,
                hidden_features if i < depth - 1 else out_features,
                key=linear_keys[i],
            )
            for i in range(depth)
        ]
        return cls(
            filters=filters,
            linears=linears,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
        )

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        squeeze = x.ndim == 1
        out = mfn_forward(x, self.filters, self.linears)
        return out[0] if squeeze else out


class GaborNet(eqx.Module):
    r"""Multiplicative Gabor Filter Network (Fathony et al., ICLR 2021).

    Same MFN topology as :class:`FourierNet` but each :math:`g_i` is a
    :class:`GaborFilter` — a sinusoidal oscillation modulated by a
    Gaussian envelope:

    .. math::

        g_i(x) = \sin(\Omega_i x + \varphi_i)
                  \odot \exp\!\bigl(-\tfrac{\gamma_i}{2}\|x - \mu_i\|^2\bigr).

    Attributes:
        filters: Length-``depth`` list of :class:`GaborFilter`.
        linears: Length-``depth`` list of :class:`equinox.nn.Linear`.
        in_features: Input dimension.
        hidden_features: Filter / hidden width.
        out_features: Output dimension.
        depth: Number of filter layers :math:`L`.
        domain: ``(low, high)`` used for :math:`\mu` initialization.
        gamma_alpha: Shape parameter of the :math:`\gamma` Gamma prior.
        gamma_beta: Rate parameter of the :math:`\gamma` Gamma prior.
    """

    filters: list[GaborFilter]
    linears: list[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    domain: tuple[float, float] = eqx.field(static=True)
    gamma_alpha: float = eqx.field(static=True)
    gamma_beta: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: PRNGKeyArray,
        domain: tuple[float, float] = (-1.0, 1.0),
        gamma_alpha: float = 6.0,
        gamma_beta: float = 1.0,
    ) -> GaborNet:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}.")
        keys = jax.random.split(key, 2 * depth)
        filter_keys = keys[:depth]
        linear_keys = keys[depth:]
        filters = [
            GaborFilter.init(
                in_features,
                hidden_features,
                key=filter_keys[i],
                domain=domain,
                gamma_alpha=gamma_alpha,
                gamma_beta=gamma_beta,
            )
            for i in range(depth)
        ]
        linears = [
            eqx.nn.Linear(
                hidden_features,
                hidden_features if i < depth - 1 else out_features,
                key=linear_keys[i],
            )
            for i in range(depth)
        ]
        return cls(
            filters=filters,
            linears=linears,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            domain=domain,
            gamma_alpha=gamma_alpha,
            gamma_beta=gamma_beta,
        )

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        squeeze = x.ndim == 1
        out = mfn_forward(x, self.filters, self.linears)
        return out[0] if squeeze else out


__all__ = [
    "FourierFilter",
    "FourierNet",
    "GaborFilter",
    "GaborNet",
    "mfn_forward",
]
