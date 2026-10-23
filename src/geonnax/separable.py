r"""Separable implicit neural representations (SPINN; Cho et al., NeurIPS 2023).

`SeparableINR` gives each input axis its own small network and combines them
with a rank-$r$ sum of products, so a field on a tensor-product grid of
$N_1 \times \dots \times N_A$ points costs $\sum_a N_a$ network evaluations
instead of $\prod_a N_a$. That makes full-grid physics residuals (for example
in neural-field 4D-Var over $(t, x, y)$) cheap to evaluate every iteration.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from geonnax.siren import SIREN


_AXES = "abcdefghij"


class SeparableINR(eqx.Module):
    r"""A rank-$r$ separable neural field over $A$ input axes.

    Each axis $a$ has a network $\phi_a : \mathbb{R} \to \mathbb{R}^{r \times K}$,
    and the field is

    $$
    f(x)_k = \sum_{j=1}^{r} \prod_{a=1}^{A} \phi_a(x_a)_{jk}.
    $$

    `__call__` evaluates one point $x \in \mathbb{R}^A$; `on_grid` evaluates
    the tensor-product grid of $A$ one-dimensional coordinate arrays at the
    cost of the 1-D networks plus one contraction. A derivative along axis $a$
    only differentiates $\phi_a$, so forward-mode `jax.jvp` per axis is cheap.

    Axis networks are `SIREN`s (``net="siren"``, the default) or tanh MLPs
    (``net="mlp"``, the SPINN paper's activation, with Equinox's default
    init), each mapping ``(1,)`` to ``(rank * out_features,)``, reshaped to
    ``(rank, out_features)``.

    Initialisation: each axis network keeps its own init (SIREN's $U(-a, a)$
    regimes via `siren_W_limit`, Equinox's default for the MLP), and the
    product is not rescaled, as in SPINN. With SIREN axes each readout is
    $O(1)$, so the field starts at $O(\sqrt{r})$ whatever $A$ is. With tanh
    MLP axes each readout is small, so the field (and its gradients) shrink
    geometrically with $A$; prefer ``net="siren"`` beyond two axes.

    Attributes:
        axis_nets: One network per axis, a tuple of length ``num_axes``.
        rank: Separation rank $r$.
        out_features: Output dimension $K$.

    References:
        Cho, J., Nam, S., Yang, H., Yun, S.-B., Hong, Y., Park, E. (2023).
        Separable physics-informed neural networks. *NeurIPS*.
    """

    axis_nets: tuple[SIREN | eqx.nn.MLP, ...]
    rank: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        num_axes: int,
        rank: int,
        hidden_features: int,
        depth: int,
        out_features: int = 1,
        *,
        net: Literal["siren", "mlp"] = "siren",
        first_omega: float = 30.0,
        key: PRNGKeyArray,
    ) -> SeparableINR:
        """Build one network per axis.

        Args:
            num_axes: Number of input axes $A$ (``1 <= A <= 10``).
            rank: Separation rank $r$ (``>= 1``).
            hidden_features: Hidden width of each axis network (``>= 1``).
            depth: Layers per axis network: SIREN layers including the
                readout (``>= 2``), or MLP hidden layers (``>= 1``).
            out_features: Output dimension $K$ (``>= 1``).
            net: ``"siren"`` or ``"mlp"`` (tanh).
            first_omega: SIREN first-layer frequency; ignored for ``"mlp"``.
            key: PRNG key, split once per axis.

        Returns:
            A new `SeparableINR`.

        Raises:
            ValueError: On a size out of range or an unknown ``net``.
        """
        if not 1 <= num_axes <= len(_AXES):
            raise ValueError(f"num_axes must be in [1, {len(_AXES)}]; got {num_axes}.")
        if rank < 1 or hidden_features < 1 or out_features < 1:
            raise ValueError(
                "rank, hidden_features and out_features must be >= 1; got "
                f"{rank}, {hidden_features}, {out_features}."
            )
        if net not in ("siren", "mlp"):
            raise ValueError(f"net must be 'siren' or 'mlp'; got {net!r}.")
        if net == "siren" and depth < 2:
            raise ValueError(f"a SIREN axis network needs depth >= 2; got {depth}.")
        if net == "mlp" and depth < 1:
            raise ValueError(f"an MLP axis network needs depth >= 1; got {depth}.")

        width = rank * out_features
        nets = []
        for k in jax.random.split(key, num_axes):
            if net == "siren":
                nets.append(
                    SIREN.init(
                        1,
                        hidden_features,
                        width,
                        depth=depth,
                        first_omega=first_omega,
                        key=k,
                    )
                )
            else:
                nets.append(
                    eqx.nn.MLP(
                        in_size=1,
                        out_size=width,
                        width_size=hidden_features,
                        depth=depth,
                        activation=jnp.tanh,
                        key=k,
                    )
                )
        return cls(axis_nets=tuple(nets), rank=rank, out_features=out_features)

    @property
    def num_axes(self) -> int:
        """Number of input axes $A$."""
        return len(self.axis_nets)

    def _axis_features(self, a: int, t: Float[Array, ""]) -> Float[Array, "R K"]:
        out = self.axis_nets[a](einx.id(" -> 1", t))  # (1,) -> (R·K,)
        return einx.id("(r k) -> r k", out, r=self.rank)

    def __call__(self, x: Float[Array, " A"]) -> Float[Array, " K"]:
        """Evaluate the field at one point.

        Args:
            x: One point, shape ``(num_axes,)``.

        Returns:
            Field value, shape ``(out_features,)``.

        Raises:
            ValueError: If ``x`` is not of shape ``(num_axes,)``.

        Examples:
            >>> import jax, jax.numpy as jnp, jax.random as jr
            >>> from geonnax import SeparableINR
            >>> f = SeparableINR.init(3, rank=8, hidden_features=16, depth=3,
            ...                       key=jr.PRNGKey(0))
            >>> f(jnp.array([0.1, -0.2, 0.3])).shape
            (1,)
            >>> jax.vmap(f)(jnp.zeros((5, 3))).shape
            (5, 1)
        """
        if x.shape != (self.num_axes,):
            raise ValueError(
                f"x must have shape ({self.num_axes},); got {x.shape}. Use jax.vmap "
                "for scattered points or on_grid for a tensor-product grid."
            )
        prod = self._axis_features(0, x[0])
        for a in range(1, self.num_axes):
            prod = prod * self._axis_features(a, x[a])
        return einx.sum("[r] k", prod)

    def on_grid(self, axes: tuple[Float[Array, " N"], ...]) -> Float[Array, "K *N"]:
        """Evaluate the field on the tensor-product grid of 1-D coordinates.

        Args:
            axes: ``num_axes`` one-dimensional coordinate arrays; axis ``a``
                has shape ``(N_a,)``.

        Returns:
            Field values, shape ``(out_features, N_1, ..., N_A)``.

        Raises:
            ValueError: If the number or rank of the coordinate arrays is wrong.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax import SeparableINR
            >>> f = SeparableINR.init(2, rank=4, hidden_features=8, depth=2,
            ...                       out_features=3, key=jr.PRNGKey(0))
            >>> f.on_grid((jnp.linspace(0, 1, 5), jnp.linspace(0, 1, 7))).shape
            (3, 5, 7)
        """
        if len(axes) != self.num_axes or any(c.ndim != 1 for c in axes):
            raise ValueError(
                f"axes must be {self.num_axes} one-dimensional arrays; got shapes "
                f"{[c.shape for c in axes]}."
            )
        feats = [
            jax.vmap(lambda t, a=a: self._axis_features(a, t))(c)  # (N_a, R, K)
            for a, c in enumerate(axes)
        ]
        names = _AXES[: self.num_axes]
        # Σ_r ∏_a φ_a[n_a, r, k] -> (K, N_1, ..., N_A). einx.dot contracts an
        # axis over exactly two operands, so the A-ary contraction uses einsum.
        spec = ",".join(f"{n}rk" for n in names) + f"->k{names}"
        return jnp.einsum(spec, *feats)


__all__ = ["SeparableINR"]
