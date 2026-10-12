r"""Local Neural Operator (LNO; Li et al., CMAME 2024): a local core marched in time.

An LNO is a *usage pattern*, not a new layer: a fully convolutional core
with a bounded (here: effectively bounded) receptive field, applied
autoregressively as an explicit time-stepper $u_t \mapsto u_{t+\Delta t}$,
with the domain boundary handled by padding the field before every step and
cropping after. Because the core is local and shift-equivariant, one trained
LNO runs on domains of any size. The default core is a shallow `CNO` on its
input grid (``resolution=None``); `LNO.from_core` wraps any other
``(C, *spatial) -> (C, *spatial)`` module, such as a shallow `UNet`.

Modules act on one example of shape ``(channels, *spatial)`` with
``num_spatial_dims`` ∈ {1, 2, 3}; ``jax.vmap`` over a batch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.cno import CNO


Boundary = Literal["periodic", "reflect", "zero"]

_PAD_MODES = {"periodic": "wrap", "reflect": "reflect", "zero": "constant"}


def cno_receptive_radius(depth: int, n_res_blocks: int, kernel_size: int) -> int:
    r"""Effective receptive radius, in grid points, of a `CNO` run on its input grid.

    Counts the convolution stencils on the deepest path of a ``CNO`` with
    ``resolution=None``: the lift, the D block of every encoder level, the
    ``2·n_res_blocks`` bottleneck convolutions, the I and U blocks of every
    decoder level and the projection block. A stencil of half-width
    $h = (k - 1)/2$ on level $\ell$ (grid spacing $2^\ell$ fine points)
    reaches $h\,2^\ell$ points, so with $L$ = ``depth`` and $n$ =
    ``n_res_blocks``

    $$
    r = h\Big[\underbrace{1}_{\text{lift}}
        + \underbrace{(2^L - 1)}_{D}
        + \underbrace{2n\,2^L}_{R^n \text{ at } L}
        + \underbrace{4(2^L - 1)}_{I,\,U}
        + \underbrace{1}_{\text{proj}}\Big]
      = h\big[(5 + 2n)\,2^L - 3\big].
    $$

    The skip paths are shorter, so this is the stencil reach of the whole
    network.

    **It is not a hard bound.** The CNO resamples with ideal Fourier filters
    (`band_limited_resize`) in every D / U block and every
    `AntiAliasedActivation`, and their sinc kernels are global with tails
    decaying like $1/\text{distance}$, so $\partial G(u)(x_a)/\partial u(x_b)$
    is small but nonzero for every $\lVert x_a - x_b\rVert > r$. Measured on
    the Jacobian row of a centre output at random initialisation
    (`tests/test_lno.py`, 1-D, 18 draws over 6 configurations), 95.4–99.9 %
    of its squared norm lies within $r$ (the tests require 90 %); most of it
    lies within 3–4 points, since the short skip paths carry the bulk of the
    sensitivity, while the 1 % tail can reach tens of points. Treat $r$ as
    the halo a boundary treatment needs and as the scale to compare with a
    domain of dependence, not as an exact cut-off.

    Pseudocode:

    ```text
    h = (kernel_size - 1) // 2
    return h * ((5 + 2 * n_res_blocks) * 2**depth - 3)
    ```

    Args:
        depth: CNO encoder / decoder levels (at least 1).
        n_res_blocks: R blocks per level (at least 0).
        kernel_size: Odd convolution stencil width.

    Returns:
        The effective receptive radius $r$ in fine-grid points.

    Raises:
        ValueError: If ``depth < 1``, ``n_res_blocks < 0`` or ``kernel_size``
            is not a positive odd integer.

    Examples:
        >>> from geonnax.lno import cno_receptive_radius
        >>> cno_receptive_radius(depth=1, n_res_blocks=1, kernel_size=3)
        11
        >>> cno_receptive_radius(depth=3, n_res_blocks=2, kernel_size=3)  # CNO defaults
        69
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}.")
    if n_res_blocks < 0:
        raise ValueError(f"n_res_blocks must be >= 0, got {n_res_blocks}.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )
    h = (kernel_size - 1) // 2
    return h * ((5 + 2 * n_res_blocks) * 2**depth - 3)


class LNO(eqx.Module):
    r"""Local Neural Operator: pad → local core → crop, marched with `jax.lax.scan`.

    Li et al. (*Local neural operator for solving transient partial
    differential equations on varied domains*, CMAME 2024,
    arXiv:2203.08145, eqs. 2–3) define the LNO as a time-marching operator
    $G: u_t \mapsto u_{t+\Delta t}$ that is

    - **local-related**: $\partial u_{t+\Delta t}(x_a) / \partial u_t(x_b)
      = 0$ whenever $\lVert x_a - x_b\rVert > r$;
    - **shift-invariant**: $G$ commutes with translations (fully
      convolutional, no positional encoding).

    One step pads the field by the halo $p$ with the boundary treatment $B$,
    applies the core $N$ and crops back,

    $$u_{t+\Delta t} = \mathrm{crop}_p\big(N(\mathrm{pad}_{p,B}(u_t))\big),$$

    and `rollout` iterates it. As in the paper the halo equals the receptive
    range ("corrosion width"), so every kept output sees only the field and
    its boundary extension, not the core's own zero padding. ``boundary`` is
    ``"periodic"`` (wrap-around; circular padding in the paper),
    ``"reflect"`` (mirror about the edge sample, no repeat) or ``"zero"``
    (constant zero; the paper's constant padding for far-field boundaries).
    Solid walls need a solver near the boundary and are out of scope.

    **Receptive range ↔ time step.** Ye et al. (*On the locality of local
    neural operator in learning fluid dynamics*, CMAME 2024,
    arXiv:2312.09820) find that the receptive range must be compatible with
    the task: too small is "fatal", usually causing numerical oscillation,
    and too large keeps the LNO from its best accuracy. Physically $r$ should
    cover the domain of dependence of one step, a CFL-style
    $r \gtrsim c\,\Delta t / \Delta x$ grid points for a signal speed $c$
    (our reading; that paper states the finding empirically). `init`
    computes $r$ with `cno_receptive_radius` and raises if it exceeds a
    caller-supplied ``receptive_range``; it is an *effective* radius (the
    CNO's Fourier resampling has small global tails), see that function.

    ```text
    def step(u):                                   # (C, *n)
        p_lo = halo; p_hi = halo + extra           # extra: round up to a
        v = pad(u, (p_lo, p_hi), boundary)         #   multiple of `multiple`
        return crop(core(v), p_lo, n)              # back to (C, *n)

    def rollout(u0, T):
        _, traj = scan(lambda u, _: (step(u), step(u)), u0, length=T)
        return traj                                # (T, C, *n): u_1 .. u_T
    ```

    Attributes:
        core: The local operator ``(C, *spatial) -> (C, *spatial)``.
        num_spatial_dims: Number of spatial axes (static).
        boundary: ``"periodic"``, ``"reflect"`` or ``"zero"`` (static).
        halo: Points padded on each side before the core (static).
        multiple: Every padded extent is rounded up to a multiple of this,
            ``2**depth`` for a CNO core (static).
        receptive_radius: The core's effective receptive radius in grid
            points, or ``None`` if unknown (static).

    Examples:
        A 1-D LNO on a periodic domain, marched for 5 steps, then the *same*
        parameters on a domain of another size:

        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.lno import LNO
        >>> lno = LNO.init(2, 1, key=jr.PRNGKey(0), hidden_channels=4,
        ...                depth=1, n_res_blocks=1)
        >>> lno.receptive_radius, lno.halo
        (11, 11)
        >>> lno.step(jnp.ones((2, 50))).shape
        (2, 50)
        >>> lno.rollout(jnp.ones((2, 50)), 5).shape     # (T, C, n)
        (5, 2, 50)
        >>> lno.rollout(jnp.ones((2, 77)), 5).shape     # any domain size
        (5, 2, 77)
        >>> jax.vmap(lno)(jnp.ones((3, 2, 50))).shape   # a batch of fields
        (3, 2, 50)

        Too large a receptive field for the target range is an error:

        >>> try:
        ...     LNO.init(2, 1, key=jr.PRNGKey(0), depth=2, receptive_range=10)
        ... except ValueError as err:
        ...     print(str(err).split(";")[0])
        the CNO core's effective receptive radius 25 exceeds receptive_range=10
    """

    core: Callable[[Array], Array]
    num_spatial_dims: int = eqx.field(static=True)
    boundary: Boundary = eqx.field(static=True)
    halo: int = eqx.field(static=True)
    multiple: int = eqx.field(static=True)
    receptive_radius: int | None = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        boundary: Boundary = "periodic",
        receptive_range: int | None = None,
        halo: int | None = None,
        hidden_channels: int = 32,
        depth: int = 1,
        n_res_blocks: int = 1,
        kernel_size: int = 3,
        activation: Callable[[Array], Array] = jax.nn.leaky_relu,
        activation_up: int = 2,
        weight_standardize: bool = False,
    ) -> LNO:
        """Construct an LNO around a shallow `CNO` core on the input grid.

        Args:
            channels: State channels (the core maps ``channels`` to
                ``channels`` so the step can be iterated).
            num_spatial_dims: Number of spatial axes (1, 2 or 3).
            key: PRNG key for the core.
            boundary: Boundary treatment, ``"periodic"``, ``"reflect"`` or
                ``"zero"``.
            receptive_range: Target receptive range $r$ in grid points (the
                domain of dependence of one step); raise if the core's
                effective radius exceeds it. ``None`` skips the check.
            halo: Points padded on each side; defaults to the core's
                effective receptive radius.
            hidden_channels: CNO width of the finest level.
            depth: CNO levels; shallow by default to keep the receptive field
                small (it grows like ``2**depth``).
            n_res_blocks: CNO R blocks per level.
            kernel_size: Odd CNO stencil width.
            activation: CNO nonlinearity.
            activation_up: Factor of the finer grid the activation runs on.
            weight_standardize: Standardize the CNO conv weights at call time.

        Returns:
            The LNO.

        Raises:
            ValueError: If ``boundary`` is unknown, ``halo`` or
                ``receptive_range`` is negative, the core's effective
                receptive radius exceeds ``receptive_range``, or the `CNO`
                rejects its arguments.
        """
        radius = cno_receptive_radius(depth, n_res_blocks, kernel_size)
        if receptive_range is not None:
            if receptive_range < 0:
                raise ValueError(
                    f"receptive_range must be >= 0, got {receptive_range}."
                )
            if radius > receptive_range:
                raise ValueError(
                    f"the CNO core's effective receptive radius {radius} exceeds "
                    f"receptive_range={receptive_range}; reduce depth, "
                    "n_res_blocks or kernel_size."
                )
        core = CNO.init(
            channels,
            channels,
            num_spatial_dims,
            key=key,
            hidden_channels=hidden_channels,
            depth=depth,
            n_res_blocks=n_res_blocks,
            kernel_size=kernel_size,
            activation=activation,
            activation_up=activation_up,
            weight_standardize=weight_standardize,
        )
        return cls.from_core(
            core,
            num_spatial_dims,
            halo=radius if halo is None else halo,
            boundary=boundary,
            multiple=2**depth,
            receptive_radius=radius,
        )

    @classmethod
    def from_core(
        cls,
        core: Callable[[Array], Array],
        num_spatial_dims: int,
        *,
        halo: int,
        boundary: Boundary = "periodic",
        multiple: int = 1,
        receptive_radius: int | None = None,
    ) -> LNO:
        """Wrap an existing local core (keyless; the core holds the weights).

        Use this for a core other than the default CNO, e.g. a shallow
        `UNet` with ``multiple=2**len(dim_mults)``. The halo is then the
        caller's estimate of the core's receptive radius. Note that a `UNet`
        is not local-related: its GroupNorm statistics and bottleneck
        attention mix the whole padded domain, so the halo fixes only the
        boundary extension, not the dependence range.

        Args:
            core: A module ``(C, *spatial) -> (C, *spatial)`` whose spatial
                extents must be multiples of ``multiple``.
            num_spatial_dims: Number of spatial axes (1, 2 or 3).
            halo: Points padded on each side before the core.
            boundary: ``"periodic"``, ``"reflect"`` or ``"zero"``.
            multiple: Round every padded extent up to a multiple of this.
            receptive_radius: The core's receptive radius, if known (stored
                for reference).

        Returns:
            The LNO.

        Raises:
            ValueError: If ``num_spatial_dims`` is not 1, 2 or 3,
                ``boundary`` is unknown, ``halo < 0`` or ``multiple < 1``.

        Examples:
            >>> import jax.numpy as jnp, jax.random as jr
            >>> from geonnax.lno import LNO
            >>> from geonnax.unet import UNet
            >>> unet = UNet.init(8, key=jr.PRNGKey(0), channels=1,
            ...                  num_spatial_dims=2, dim_mults=(1, 2))
            >>> lno = LNO.from_core(unet, 2, halo=4, boundary="reflect",
            ...                     multiple=4)
            >>> lno.rollout(jnp.ones((1, 30, 30)), 3).shape
            (3, 1, 30, 30)
        """
        if num_spatial_dims not in (1, 2, 3):
            raise ValueError(
                f"num_spatial_dims must be 1, 2 or 3, got {num_spatial_dims}."
            )
        if boundary not in _PAD_MODES:
            raise ValueError(
                f"boundary must be 'periodic', 'reflect' or 'zero', got {boundary!r}."
            )
        if halo < 0:
            raise ValueError(f"halo must be >= 0, got {halo}.")
        if multiple < 1:
            raise ValueError(f"multiple must be >= 1, got {multiple}.")
        return cls(
            core=core,
            num_spatial_dims=num_spatial_dims,
            boundary=boundary,
            halo=halo,
            multiple=multiple,
            receptive_radius=receptive_radius,
        )

    def pad(self, u: Float[Array, "C *spatial"]) -> Float[Array, "C *padded"]:
        """Extend every spatial axis by ``halo`` per side with the boundary rule.

        The high side gets the extra points that round the extent up to a
        multiple of ``multiple``. These lie beyond the kept region and are
        filled by the same rule.
        """
        widths = []
        for n in u.shape[1:]:
            extra = -(n + 2 * self.halo) % self.multiple
            widths.append((self.halo, self.halo + extra))
        return jnp.pad(u, [(0, 0), *widths], mode=_PAD_MODES[self.boundary])

    def unpad(
        self, v: Float[Array, "C *padded"], spatial: tuple[int, ...]
    ) -> Float[Array, "C *spatial"]:
        """Crop the ``halo``-offset window of extents ``spatial``."""
        return v[(slice(None), *(slice(self.halo, self.halo + n) for n in spatial))]

    def step(self, u: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        """Advance one $\\Delta t$: pad, apply the core, crop."""
        if u.ndim != 1 + self.num_spatial_dims:
            raise ValueError(
                f"expected (C, *spatial) with {self.num_spatial_dims} spatial "
                f"axes, got shape {u.shape}."
            )
        return self.unpad(self.core(self.pad(u)), tuple(u.shape[1:]))

    def __call__(self, u: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        """Alias of `step`, the one-step operator."""
        return self.step(u)

    def rollout(
        self, u0: Float[Array, "C *spatial"], n_steps: int
    ) -> Float[Array, "T C *spatial"]:
        """March ``n_steps`` steps from ``u0`` with `jax.lax.scan`.

        Args:
            u0: Initial state ``(C, *spatial)``.
            n_steps: Number of steps $T \\geq 1$ (a Python int).

        Returns:
            The trajectory $u_1, \\dots, u_T$, shape ``(T, C, *spatial)``;
            ``u0`` itself is not included.

        Raises:
            ValueError: If ``n_steps < 1``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")

        def body(u: Array, _: None) -> tuple[Array, Array]:
            v = self.step(u)
            return v, v

        _, traj = jax.lax.scan(body, u0, None, length=n_steps)
        return traj


LocalNeuralOperator = LNO
"""Verbose alias for `LNO`."""


__all__ = ["LNO", "LocalNeuralOperator", "cno_receptive_radius"]
