"""Sinusoidal Representation Networks (Sitzmann et al., NeurIPS 2020).

Pure ``equinox.Module`` cores. The Bayesian SIREN wrapper that swaps
each layer's ``W``/``b`` for ``pyrox_sample`` sites is the
representative Tier-D conversion built atop these cores.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


SirenLayerType = Literal["first", "hidden", "last"]


class SirenLayerSpec(NamedTuple):
    """Static per-layer metadata shared by deterministic and Bayesian SIRENs."""

    layer_type: SirenLayerType
    in_features: int
    out_features: int
    omega: float
    c: float


def siren_W_limit(
    layer_type: SirenLayerType,
    in_features: int,
    omega: float,
    c: float = 6.0,
) -> float:
    """Return the half-width ``a`` of the ``U(-a, a)`` weight init for a SIREN layer.

    Implements Sitzmann et al. (2020) Theorem 1's three-regime prescription:

    - ``"first"``:  ``a = 1 / in_features``
    - ``"hidden"``: ``a = sqrt(c / in_features) / omega``
    - ``"last"``:   ``a = sqrt(c / in_features)``

    Args:
        layer_type: One of ``"first"``, ``"hidden"``, ``"last"``.
        in_features: Input dimension of the layer.
        omega: Frequency multiplier (only affects hidden-layer limit).
        c: Constant from Theorem 1; defaults to 6.0.

    Returns:
        Half-width ``a`` such that ``W ~ U(-a, a)``.

    Raises:
        ValueError: If ``layer_type`` is not one of the three valid values.
    """
    if layer_type == "first":
        return 1.0 / in_features
    if layer_type == "hidden":
        return math.sqrt(c / in_features) / omega
    if layer_type == "last":
        return math.sqrt(c / in_features)
    raise ValueError(
        f"layer_type must be 'first', 'hidden', or 'last', got {layer_type!r}"
    )


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` if any keyword value is non-positive."""
    for name, v in values.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}.")


def build_siren_specs(
    in_features: int,
    hidden_features: int,
    out_features: int,
    depth: int,
    first_omega: float,
    hidden_omega: float,
    c: float,
) -> tuple[SirenLayerSpec, ...]:
    """Produce per-layer specs for a depth-``depth`` SIREN, first + hidden… + last."""
    specs: list[SirenLayerSpec] = []
    for i in range(depth):
        if i == 0:
            specs.append(
                SirenLayerSpec("first", in_features, hidden_features, first_omega, c)
            )
        elif i == depth - 1:
            specs.append(
                SirenLayerSpec("last", hidden_features, out_features, hidden_omega, c)
            )
        else:
            specs.append(
                SirenLayerSpec(
                    "hidden", hidden_features, hidden_features, hidden_omega, c
                )
            )
    return tuple(specs)


class SirenDense(eqx.Module):
    r"""Sine-activated dense layer: ``y = sin(ω · (W x + b))`` or ``y = W x + b``.

    Single primitive of a SIREN network with three init regimes
    (Sitzmann et al. 2020, Theorem 1):

    +----------+-------------------------------------------+-------------+
    | Regime   | ``W`` init                                | Activation  |
    +==========+===========================================+=============+
    | first    | ``U(-1/d_in, 1/d_in)``                    | ``sin(ω··)``|
    +----------+-------------------------------------------+-------------+
    | hidden   | ``U(-√(c/d_in)/ω, √(c/d_in)/ω)``         | ``sin(ω··)``|
    +----------+-------------------------------------------+-------------+
    | last     | ``U(-√(c/d_in), √(c/d_in))``              | none        |
    +----------+-------------------------------------------+-------------+

    Bias ``b`` is initialised ``U(-1/√d_in, 1/√d_in)`` for every regime.

    Attributes:
        W: Weight matrix of shape ``(in_features, out_features)``.
        b: Bias vector of shape ``(out_features,)``.
        omega: Frequency multiplier applied inside the sine.
        in_features: Input dimension.
        out_features: Output dimension.
        layer_type: One of ``"first"``, ``"hidden"``, ``"last"``.
        c: Constant from Theorem 1 (default 6.0).
    """

    W: Float[Array, "in_features out_features"]
    b: Float[Array, " out_features"]
    omega: float = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    layer_type: SirenLayerType = eqx.field(static=True)
    c: float = eqx.field(static=True, default=6.0)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: Array,
        omega: float = 30.0,
        layer_type: SirenLayerType = "hidden",
        c: float = 6.0,
    ) -> SirenDense:
        """Construct a ``SirenDense`` with Sitzmann-regime weight initialisation."""
        _require_positive(
            in_features=in_features,
            out_features=out_features,
            omega=omega,
            c=c,
        )
        k_w, k_b = jax.random.split(key)
        # siren_W_limit validates layer_type.
        w_limit = siren_W_limit(layer_type, in_features, omega, c)
        W = jax.random.uniform(
            k_w, (in_features, out_features), minval=-w_limit, maxval=w_limit
        )
        b_limit = 1.0 / math.sqrt(in_features)
        b = jax.random.uniform(k_b, (out_features,), minval=-b_limit, maxval=b_limit)
        return cls(
            W=W,
            b=b,
            omega=omega,
            in_features=in_features,
            out_features=out_features,
            layer_type=layer_type,
            c=c,
        )

    def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, " D_out"]:
        pre = einx.dot("i, i o -> o", x, self.W) + self.b
        if self.layer_type == "last":
            return pre
        return jnp.sin(self.omega * pre)


class SIREN(eqx.Module):
    r"""Multi-layer sinusoidal representation network (Sitzmann et al., NeurIPS 2020).

    Topology:

    .. math::

        z_1 &= \sin(\omega_0 (W_0 x + b_0)), \\
        z_{i+1} &= \sin(\omega (W_i z_i + b_i)), \quad i = 1 \ldots L-1, \\
        y &= W_L z_L + b_L.

    Each layer uses the corresponding Sitzmann Theorem 1 init regime
    (:class:`SirenDense`):  ``"first"`` for layer 0, ``"hidden"`` for
    intermediate layers, and ``"last"`` for the readout.

    ``depth`` counts *all* layers including the readout; ``depth=2`` gives
    one first-layer + one last-layer (no hidden layers); ``depth=5`` gives
    first + 3 hidden + last.  Must be ≥ 2.
    """

    layers: list[SirenDense]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    first_omega: float = eqx.field(static=True)
    hidden_omega: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: Array,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
        c: float = 6.0,
    ) -> SIREN:
        """Construct a SIREN with the correct per-layer init regimes."""
        if depth < 2:
            raise ValueError(f"depth must be >= 2 (first + last); got depth={depth}")
        _require_positive(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
            c=c,
        )
        specs = build_siren_specs(
            in_features,
            hidden_features,
            out_features,
            depth,
            first_omega,
            hidden_omega,
            c,
        )
        keys = jax.random.split(key, depth)
        layers = [
            SirenDense.init(
                spec.in_features,
                spec.out_features,
                key=k,
                omega=spec.omega,
                layer_type=spec.layer_type,
                c=spec.c,
            )
            for spec, k in zip(specs, keys, strict=True)
        ]
        return cls(
            layers=layers,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
        )

    def __call__(self, x: Float[Array, " D_in"]) -> Float[Array, " D_out"]:
        z = x
        for layer in self.layers:
            z = layer(z)
        return z


__all__ = [
    "SIREN",
    "SirenDense",
    "SirenLayerSpec",
    "SirenLayerType",
    "build_siren_specs",
    "siren_W_limit",
]
