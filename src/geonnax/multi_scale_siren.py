r"""Multi-scale (frequency-banded) sinusoidal representation networks.

A plain SIREN carries a single first-layer frequency $\omega_0$, which fixes
the band the network is biased toward (Rahaman et al., 2019; Wang et al.,
2021). `MultiScaleSIREN` runs $K$ independent `SIREN` sub-networks at
logarithmically spaced first-layer frequencies and sums their readouts, so a
signal with mixed low- and high-frequency content is covered by construction
rather than by tuning one $\omega_0$.

The Bayesian variant — priors per band — lives downstream; this module is the
deterministic core.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from geonnax.siren import SIREN


def log_spaced_omegas(
    omega_min: float,
    omega_max: float,
    num_scales: int,
) -> tuple[float, ...]:
    r"""Return ``num_scales`` log-spaced frequencies spanning the closed band.

    Implements
    $\omega_0^{(k)} = \omega_{\min}\,(\omega_{\max} / \omega_{\min})^{(k-1)/(K-1)}$
    for $k = 1 \ldots K$, matching `numpy.logspace` endpoints. A single scale
    degenerates to ``(omega_min,)``.

    Args:
        omega_min: Lowest first-layer frequency; must be > 0.
        omega_max: Highest first-layer frequency; must be >= ``omega_min``.
        num_scales: Number of bands $K$; must be >= 1.

    Returns:
        Tuple of ``num_scales`` Python floats, ascending.

    Raises:
        ValueError: If the frequencies are non-positive, out of order, or
            ``num_scales < 1``.

    Examples:
        >>> from geonnax.multi_scale_siren import log_spaced_omegas
        >>> [round(w, 3) for w in log_spaced_omegas(1.0, 100.0, 5)]
        [1.0, 3.162, 10.0, 31.623, 100.0]
        >>> log_spaced_omegas(30.0, 60.0, 1)  # single band -> the lower edge
        (30.0,)
    """
    if num_scales < 1:
        raise ValueError(f"num_scales must be >= 1, got {num_scales}.")
    if omega_min <= 0:
        raise ValueError(f"omega_min must be > 0, got {omega_min}.")
    if omega_max < omega_min:
        raise ValueError(
            f"omega_max must be >= omega_min, got {omega_max} < {omega_min}."
        )
    if num_scales == 1:
        return (float(omega_min),)
    ratio = omega_max / omega_min
    return tuple(
        float(omega_min * ratio ** (k / (num_scales - 1))) for k in range(num_scales)
    )


class MultiScaleSIREN(eqx.Module):
    r"""Parallel frequency-banded SIRENs summed at the readout.

    Topology (Pan et al., 2023, §3.2):

    $$
    u(x) = \sum_{k=1}^{K} \mathrm{SIREN}_k(x),
    \qquad
    \omega_0^{(k)} = \omega_{\min}
    \left(\frac{\omega_{\max}}{\omega_{\min}}\right)^{(k-1)/(K-1)} .
    $$

    Each $\mathrm{SIREN}_k$ is a complete `geonnax.SIREN` built at
    ``first_omega = omega_0_values[k]``; sub-networks share only the input
    $x$ and the readout shape — there is no weight sharing across bands.
    ``hidden_omega`` is shared by every band because the first layer carries
    the frequency selectivity while hidden layers carry composition (Wang et
    al., 2021, §3.3). ``depth`` and ``hidden_features`` apply uniformly.

    The readout is a plain **sum**, not a learned mixture; compose with an
    `equinox.nn.Linear` downstream if you want a learned per-band weighting.

    Choosing between this and `geonnax.mfn`: an MFN is one deep network whose
    output is a product of filters — a single path reaching exponentially many
    effective frequencies through depth, which suits signals needing many
    scales resolved at *every* coordinate. `MultiScaleSIREN` is $K$ parallel
    paths each pinned to its own band, which suits signals that decompose
    spectrally (PDE solutions with clean low/high separation, multi-band
    audio, spatiotemporal fields whose axes differ in frequency content).

    Attributes:
        sub_networks: The $K$ per-band `geonnax.SIREN` sub-networks.
        omega_0_values: First-layer frequency of each sub-network, ascending.
        in_features: Input dimension.
        hidden_features: Hidden width of every sub-network.
        out_features: Output dimension.
        depth: Layer count per sub-network, readout included.
        num_scales: Number of bands $K$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.multi_scale_siren import MultiScaleSIREN
        >>> net = MultiScaleSIREN.init(
        ...     2, 32, 1, depth=4, num_scales=5, key=jr.PRNGKey(0)
        ... )
        >>> net(jnp.ones(2)).shape  # (2,) -> (1,)
        (1,)
        >>> [round(w, 2) for w in net.omega_0_values]
        [1.0, 2.78, 7.75, 21.56, 60.0]
    """

    sub_networks: tuple[SIREN, ...]
    omega_0_values: tuple[float, ...] = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    num_scales: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        num_scales: int,
        key: PRNGKeyArray,
        omega_min: float = 1.0,
        omega_max: float = 60.0,
        omega_0_values: tuple[float, ...] | None = None,
        hidden_omega: float = 30.0,
        c: float = 6.0,
    ) -> MultiScaleSIREN:
        r"""Build ``num_scales`` SIRENs at log-spaced (or explicit) frequencies.

        Args:
            in_features: Input dimension.
            hidden_features: Hidden width, shared by every band.
            out_features: Output dimension.
            depth: Layers per sub-network including the readout; must be >= 2.
            num_scales: Number of bands $K$; must be >= 1.
            key: PRNG key, split one way per sub-network.
            omega_min: Lowest first-layer frequency (ignored when
                ``omega_0_values`` is given).
            omega_max: Highest first-layer frequency (ignored when
                ``omega_0_values`` is given).
            omega_0_values: Explicit per-band first-layer frequencies. Must
                have length ``num_scales``; overrides the log-spaced default.
            hidden_omega: Hidden-layer frequency, shared across all bands.
            c: Constant from Sitzmann et al. (2020), Theorem 1.

        Returns:
            The initialised `MultiScaleSIREN`.

        Raises:
            ValueError: If ``num_scales < 1``, or ``omega_0_values`` is given
                with a length other than ``num_scales``.

        Examples:
            >>> import jax.random as jr
            >>> from geonnax.multi_scale_siren import MultiScaleSIREN
            >>> net = MultiScaleSIREN.init(
            ...     1, 16, 1, depth=3, num_scales=3,
            ...     omega_0_values=(2.0, 10.0, 50.0), key=jr.PRNGKey(0),
            ... )
            >>> [sub.first_omega for sub in net.sub_networks]
            [2.0, 10.0, 50.0]
        """
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}.")
        if omega_0_values is None:
            omegas = log_spaced_omegas(omega_min, omega_max, num_scales)
        else:
            if len(omega_0_values) != num_scales:
                raise ValueError(
                    f"omega_0_values must have length num_scales={num_scales}, "
                    f"got {len(omega_0_values)}."
                )
            omegas = tuple(float(w) for w in omega_0_values)
        keys = jax.random.split(key, num_scales)
        # SIREN.init validates depth, the feature sizes, and every omega.
        sub_networks = tuple(
            SIREN.init(
                in_features,
                hidden_features,
                out_features,
                depth=depth,
                key=k,
                first_omega=omega,
                hidden_omega=hidden_omega,
                c=c,
            )
            for omega, k in zip(omegas, keys, strict=True)
        )
        return cls(
            sub_networks=sub_networks,
            omega_0_values=omegas,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            num_scales=num_scales,
        )

    def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, " D_out"]:
        r"""Sum the per-band readouts: $u(x) = \sum_k \mathrm{SIREN}_k(x)$.

        Args:
            x: Input vector of shape ``(in_features,)``.

        Returns:
            Output vector of shape ``(out_features,)``.

        Examples:
            >>> import jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax.multi_scale_siren import MultiScaleSIREN
            >>> net = MultiScaleSIREN.init(
            ...     2, 16, 3, depth=3, num_scales=4, key=jr.PRNGKey(0)
            ... )
            >>> net(jnp.ones(2)).shape  # single example
            (3,)
            >>> jax.vmap(net)(jnp.ones((10, 2))).shape  # batched
            (10, 3)
        """
        # Each band is an independent SIREN; the readout is an unweighted sum.
        out = self.sub_networks[0](x)
        for sub in self.sub_networks[1:]:
            out = out + sub(x)
        return out


__all__ = ["MultiScaleSIREN", "log_spaced_omegas"]
