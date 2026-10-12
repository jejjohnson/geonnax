r"""Convolutional Neural Operator (CNO; Raonić et al., NeurIPS 2023) and its blocks.

A CNO is a U-Net whose every operation maps band-limited functions to
band-limited functions, so the discrete network is the faithful
representation of a continuous operator (a representation-equivalent neural
operator, ReNO) rather than a resolution-bound CNN. It keeps the U-Net's
*local* convolution stencils, which suit sharp or discontinuous features that
the mode truncation of `FNO` smooths, and makes the rest band-limit
preserving with `band_limited_resize` (sampling-rate changes) and
`AntiAliasedActivation` (the nonlinearity evaluated on a finer grid).

Modules act on one example of shape ``(channels, *spatial)`` with
``num_spatial_dims`` ∈ {1, 2, 3} spatial axes on the periodic unit box;
``jax.vmap`` over a batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import einx
import equinox as eqx
import jax
from jaxtyping import Array, Float

from geonnax.layers import AntiAliasedActivation, StandardizedConv, band_limited_resize
from geonnax.layers._utils import pointwise


Resample = Literal["same", "down", "up"]


def _check_example(x: Array, num_spatial_dims: int) -> None:
    """Raise unless ``x`` is one ``(C, *spatial)`` example with that many axes."""
    if x.ndim != 1 + num_spatial_dims:
        raise ValueError(
            f"expected (C, *spatial) with {num_spatial_dims} spatial axes, "
            f"got shape {x.shape}."
        )


def _same_conv(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    key: Array,
    weight_standardize: bool,
) -> StandardizedConv:
    """A ``kernel_size`` stencil with zero "same" padding (CNO SM A.4)."""
    return StandardizedConv.init(
        num_spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        key=key,
        padding=kernel_size // 2,
        standardize=weight_standardize,
    )


class CNOBlock(eqx.Module):
    r"""Convolution then band-limited activation: the CNO I, D and U blocks.

    With $K$ a ``kernel_size`` convolution on the current $N$-point grid,
    $\sigma$ the pointwise ``activation`` and $R_M$ the band-limited resize
    onto $M$ points (`band_limited_resize`), one block computes

    $$v \mapsto R_{M}\,\sigma\big(R_{rN}\,K v\big),$$

    with $r$ = ``activation_up`` and $M = N$ (``resample="same"``, the
    invariant block $I = \Sigma \circ K$), $M = N/2$ (``"down"``, the D block)
    or $M = 2N$ (``"up"``, the U block), per spatial axis (Raonić et al.,
    NeurIPS 2023, eqs. 2.4–2.8 and SM A.4). For ``"same"`` this is
    `AntiAliasedActivation` applied to $K v$.

    ```text
    h = conv(x)                                    # (C_out, *N), local stencil
    fine = r * N
    out = N | N // 2 | 2 * N                       # same | down | up
    return resize(sigma(resize(h, fine)), out)     # up -> sigma -> low-pass
    ```

    Attributes:
        conv: The ``kernel_size`` convolution $K$.
        activation: The anti-aliased activation (its ``activation`` and
            ``up`` set $\sigma$ and $r$).
        resample: ``"same"``, ``"down"`` or ``"up"`` (static).

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.cno import CNOBlock
        >>> down = CNOBlock.init(4, 8, 2, key=jr.PRNGKey(0), resample="down")
        >>> down(jnp.ones((4, 32, 32))).shape        # D block: halves the grid
        (8, 16, 16)
        >>> up = CNOBlock.init(8, 4, 2, key=jr.PRNGKey(1), resample="up")
        >>> up(jnp.ones((8, 16, 16))).shape          # U block: doubles it
        (4, 32, 32)
    """

    conv: StandardizedConv
    activation: AntiAliasedActivation
    resample: Resample = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        resample: Resample = "same",
        kernel_size: int = 3,
        activation: Callable[[Array], Array] = jax.nn.leaky_relu,
        activation_up: int = 2,
        weight_standardize: bool = False,
    ) -> CNOBlock:
        """Construct an I (``"same"``), D (``"down"``) or U (``"up"``) block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            num_spatial_dims: Number of spatial axes (1, 2 or 3).
            key: PRNG key for the convolution.
            resample: Grid change after the activation: ``"same"``,
                ``"down"`` (halve) or ``"up"`` (double).
            kernel_size: Odd convolution stencil width.
            activation: Pointwise nonlinearity (leaky ReLU, as in CNO).
            activation_up: Factor of the finer grid it is evaluated on.
            weight_standardize: Standardize the conv weights at call time.

        Returns:
            The block.

        Raises:
            ValueError: If a channel count or ``activation_up`` is below 1,
                ``num_spatial_dims`` is not 1, 2 or 3, ``kernel_size`` is not
                a positive odd integer, or ``resample`` is unknown.
        """
        _validate_common(num_spatial_dims, kernel_size, activation_up)
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"in_channels and out_channels must be >= 1, got {in_channels} "
                f"and {out_channels}."
            )
        if resample not in ("same", "down", "up"):
            raise ValueError(
                f"resample must be 'same', 'down' or 'up', got {resample!r}."
            )
        conv = _same_conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            key=key,
            weight_standardize=weight_standardize,
        )
        act = AntiAliasedActivation.init(
            num_spatial_dims, activation=activation, up=activation_up
        )
        return cls(conv=conv, activation=act, resample=resample)

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *out_spatial"]:
        _check_example(x, self.activation.num_spatial_dims)
        h = self.conv(x)
        if self.resample == "same":
            return self.activation(h)
        spatial = tuple(h.shape[1:])
        if self.resample == "down":
            if any(n % 2 for n in spatial):
                raise ValueError(
                    f"a 'down' CNOBlock needs even spatial extents, got {spatial}."
                )
            out = tuple(n // 2 for n in spatial)
        else:
            out = tuple(2 * n for n in spatial)
        fine = tuple(self.activation.up * n for n in spatial)
        v = self.activation.activation(band_limited_resize(h, fine))
        return band_limited_resize(v, out)


class CNOResBlock(eqx.Module):
    r"""The CNO residual (R) block, $v \mapsto v + K_2\,\Sigma\,K_1 v$.

    Two same-width ``kernel_size`` convolutions around an
    `AntiAliasedActivation` $\Sigma$, plus the identity (Raonić et al.,
    NeurIPS 2023, eq. 2.7). The grid and channel count are preserved.

    ```text
    return x + conv2(antialiased_activation(conv1(x)))
    ```

    Attributes:
        conv1: First convolution $K_1$.
        activation: The anti-aliased activation $\Sigma$.
        conv2: Second convolution $K_2$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.cno import CNOResBlock
        >>> blk = CNOResBlock.init(8, 1, key=jr.PRNGKey(0))
        >>> blk(jnp.ones((8, 64))).shape
        (8, 64)
    """

    conv1: StandardizedConv
    activation: AntiAliasedActivation
    conv2: StandardizedConv

    @classmethod
    def init(
        cls,
        channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        kernel_size: int = 3,
        activation: Callable[[Array], Array] = jax.nn.leaky_relu,
        activation_up: int = 2,
        weight_standardize: bool = False,
    ) -> CNOResBlock:
        """Construct an R block at width ``channels``.

        Args:
            channels: Input and output channels.
            num_spatial_dims: Number of spatial axes (1, 2 or 3).
            key: PRNG key for the two convolutions.
            kernel_size: Odd convolution stencil width.
            activation: Pointwise nonlinearity (leaky ReLU, as in CNO).
            activation_up: Factor of the finer grid it is evaluated on.
            weight_standardize: Standardize the conv weights at call time.

        Returns:
            The block.

        Raises:
            ValueError: If ``channels`` or ``activation_up`` is below 1,
                ``num_spatial_dims`` is not 1, 2 or 3, or ``kernel_size`` is
                not a positive odd integer.
        """
        _validate_common(num_spatial_dims, kernel_size, activation_up)
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}.")
        k1, k2 = jax.random.split(key)
        conv1, conv2 = (
            _same_conv(
                num_spatial_dims,
                channels,
                channels,
                kernel_size,
                key=k,
                weight_standardize=weight_standardize,
            )
            for k in (k1, k2)
        )
        act = AntiAliasedActivation.init(
            num_spatial_dims, activation=activation, up=activation_up
        )
        return cls(conv1=conv1, activation=act, conv2=conv2)

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        _check_example(x, self.activation.num_spatial_dims)
        return x + self.conv2(self.activation(self.conv1(x)))


def _validate_common(num_spatial_dims: int, kernel_size: int, up: int) -> None:
    """Shared `init` checks of the CNO blocks."""
    if num_spatial_dims not in (1, 2, 3):
        raise ValueError(f"num_spatial_dims must be 1, 2 or 3, got {num_spatial_dims}.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}."
        )
    if up < 1:
        raise ValueError(f"activation_up must be an integer >= 1, got {up}.")


class CNO(eqx.Module):
    r"""Convolutional Neural Operator: lift → U-Net of band-limited blocks → project.

    The operator U-Net of Raonić et al. (*Convolutional Neural Operators for
    robust and accurate learning of PDEs*, NeurIPS 2023, arXiv:2302.01178,
    Fig. 1 and SM C.1.4). With $c_\ell = $ ``hidden_channels``$\cdot 2^\ell$
    channels on the $N / 2^\ell$-point grid of level $\ell = 0, \dots, L$
    ($L$ = ``depth``), $I$, $D$, $U$ the `CNOBlock` kinds and $R$ the
    `CNOResBlock`,

    $$
    \begin{aligned}
    v_0 &= I(u), \qquad
      s_\ell = R^{n}(v_\ell), \qquad v_{\ell+1} = D(v_\ell), \\
    x_L &= R^{n}(v_L), \qquad
      x_\ell = \big[\,U\,I(x_{\ell+1}),\; s_\ell\,\big], \\
    G(u) &= W\, I(x_0),
    \end{aligned}
    $$

    with $R^n$ the ``n_res_blocks`` R blocks in sequence, $[\cdot,\cdot]$
    channel concatenation (the U-Net patching) and $W$ a pointwise
    projection. Every block maps band-limited functions to band-limited
    functions, so the network is a ReNO.

    ```text
    if resolution: u = resize(u, resolution)      # onto the computational grid
    v = lift(u); skips = []
    for l in 0..L-1:
        skips.append(res_blocks[l](v))            # R blocks on the skip
        v = down_blocks[l](v)                     # D: conv, Σ, halve grid
    x = res_blocks[L](v)                          # bottleneck R blocks
    for l in L-1..0:
        x = up_blocks[l](inv_blocks[l](x))        # I, then U: double grid
        x = concat([x, skips[l]], channel axis)
    y = proj(proj_block(x))                       # I, then 1x1 conv
    if resolution: y = resize(y, input grid)      # back to the caller's grid
    ```

    **Resolution.** The convolution stencils are tied to the grid spacing, so
    a CNO is defined on one computational grid (SM C.4). Set ``resolution``
    (the training grid) and every input is first band-limited-resized onto
    it and the output resized back, which realises the continuous operator
    on any input grid: this is how the paper evaluates CNO across
    resolutions. With ``resolution=None`` the network runs on whatever grid
    it is given, as a local, fixed-spacing operator (the mode a
    time-marching local neural operator wants), and is *not* resolution
    invariant. Either way CNO reduces aliasing rather than eliminating it,
    and gives no free zero-shot super-resolution beyond its computational
    grid. All resampling is periodic (an ideal Fourier filter, where the
    reference implementation uses windowed sinc filters); the convolutions
    zero-pad.

    **Activation.** The paper's leaky ReLU at ``activation_up=2`` is the
    default. Its kink decays slowly in frequency, so the 2× grid cuts its
    aliasing error only about 5× (1-D), against about 65× (1-D) and 25× (2-D)
    for GELU (`tests/test_antialias.py`); pass ``activation=jax.nn.gelu`` or
    a larger ``activation_up`` (cost grows as ``activation_up**d``) when
    aliasing matters more than fidelity to the paper. The paper's batch norm
    after each convolution has no single-example equivalent and is omitted;
    ``weight_standardize`` is available instead.

    Attributes:
        lift: Lifting block $I$, ``in_channels -> hidden_channels``.
        down_blocks: The ``depth`` D blocks.
        res_blocks: ``depth + 1`` lists of ``n_res_blocks`` R blocks (levels
            ``0..depth-1`` on the skips, the last at the bottleneck).
        inv_blocks: The ``depth`` decoder I blocks, indexed by level.
        up_blocks: The ``depth`` U blocks, indexed by level.
        proj_block: Projection I block, ``2 * hidden_channels ->
            hidden_channels``.
        proj: Pointwise output map ``hidden_channels -> out_channels``.
        num_spatial_dims, in_channels, out_channels, hidden_channels: Static.
        depth, kernel_size: Static.
        resolution: The computational grid, or ``None`` (static).

    Examples:
        A 2-D operator on a 32 × 32 computational grid, evaluated at two
        resolutions with the *same* parameters, and on a batch:

        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.cno import CNO
        >>> op = CNO.init(3, 1, 2, key=jr.PRNGKey(0), hidden_channels=8,
        ...               depth=2, n_res_blocks=1, resolution=32)
        >>> op(jnp.ones((3, 32, 32))).shape
        (1, 32, 32)
        >>> op(jnp.ones((3, 64, 64))).shape          # resized to 32², and back
        (1, 64, 64)
        >>> jax.vmap(op)(jnp.ones((4, 3, 32, 32))).shape   # batched
        (4, 1, 32, 32)

        Without ``resolution`` it runs on the given grid (each extent
        divisible by ``2**depth``), here a 1-D profile:

        >>> op = CNO.init(1, 2, 1, key=jr.PRNGKey(0), hidden_channels=4, depth=3)
        >>> op(jnp.ones((1, 40))).shape
        (2, 40)
    """

    lift: CNOBlock
    down_blocks: list[CNOBlock]
    res_blocks: list[list[CNOResBlock]]
    inv_blocks: list[CNOBlock]
    up_blocks: list[CNOBlock]
    proj_block: CNOBlock
    proj: eqx.nn.Conv
    num_spatial_dims: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    resolution: tuple[int, ...] | None = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        hidden_channels: int = 32,
        depth: int = 3,
        n_res_blocks: int = 2,
        kernel_size: int = 3,
        activation: Callable[[Array], Array] = jax.nn.leaky_relu,
        activation_up: int = 2,
        weight_standardize: bool = False,
        resolution: int | Sequence[int] | None = None,
    ) -> CNO:
        """Construct a CNO.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            num_spatial_dims: Number of spatial axes (1, 2 or 3).
            key: PRNG key.
            hidden_channels: Width of the finest level; level ``l``
                has ``hidden_channels * 2**l`` channels.
            depth: Number of encoder / decoder levels (at least 1).
            n_res_blocks: R blocks per level and at the bottleneck (the
                paper's N_res,i and N_res,b, tied); ``0`` leaves the
                skips as plain identities.
            kernel_size: Odd convolution stencil width (3 in the paper).
            activation: Pointwise nonlinearity (leaky ReLU, as in the paper).
            activation_up: Factor of the finer grid every activation is
                evaluated on (2 in the paper).
            weight_standardize: Standardize the conv weights at call time.
            resolution: The computational grid: an int (every axis) or one
                extent per axis, each divisible by ``2**depth``. Inputs on
                another grid are resized onto it and the output back.
                ``None`` runs on the input grid.

        Returns:
            The operator.

        Raises:
            ValueError: If a channel count or ``activation_up`` is below 1,
                ``num_spatial_dims`` is not 1, 2 or 3, ``depth < 1``,
                ``n_res_blocks < 0``, ``kernel_size`` is not a positive odd
                integer, or ``resolution`` has the wrong length or an extent
                not divisible by ``2**depth``.
        """
        _validate_common(num_spatial_dims, kernel_size, activation_up)
        if min(in_channels, out_channels, hidden_channels) < 1:
            raise ValueError(
                "in_channels, out_channels and hidden_channels must be >= 1, got "
                f"{in_channels}, {out_channels} and {hidden_channels}."
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        if n_res_blocks < 0:
            raise ValueError(f"n_res_blocks must be >= 0, got {n_res_blocks}.")
        grid: tuple[int, ...] | None = None
        if resolution is not None:
            grid = (
                (int(resolution),) * num_spatial_dims
                if isinstance(resolution, int)
                else tuple(int(n) for n in resolution)
            )
            if len(grid) != num_spatial_dims:
                raise ValueError(
                    f"resolution must have {num_spatial_dims} extents, got {grid}."
                )
            _check_divisible(grid, depth, "resolution")

        widths = [hidden_channels * 2**level for level in range(depth + 1)]
        n_keys = 3 + 3 * depth + (depth + 1) * n_res_blocks
        keys = iter(jax.random.split(key, n_keys))
        d = num_spatial_dims

        def block(c_in: int, c_out: int, resample: Resample = "same") -> CNOBlock:
            return CNOBlock.init(
                c_in,
                c_out,
                d,
                key=next(keys),
                resample=resample,
                kernel_size=kernel_size,
                activation=activation,
                activation_up=activation_up,
                weight_standardize=weight_standardize,
            )

        def res_block(channels: int) -> CNOResBlock:
            return CNOResBlock.init(
                channels,
                d,
                key=next(keys),
                kernel_size=kernel_size,
                activation=activation,
                activation_up=activation_up,
                weight_standardize=weight_standardize,
            )

        lift = block(in_channels, widths[0])
        down_blocks = [block(widths[lv], widths[lv + 1], "down") for lv in range(depth)]
        res_blocks = [
            [res_block(widths[lv]) for _ in range(n_res_blocks)]
            for lv in range(depth + 1)
        ]
        # Decoder level ``lv`` reads the bottleneck (width c_L) at the deepest
        # level, else the concatenation [U(...), skip] of width 2 c_{lv+1}.
        dec_in = [
            widths[depth] if lv == depth - 1 else 2 * widths[lv + 1]
            for lv in range(depth)
        ]
        inv_blocks = [block(dec_in[lv], dec_in[lv]) for lv in range(depth)]
        up_blocks = [block(dec_in[lv], widths[lv], "up") for lv in range(depth)]
        proj_block = block(2 * widths[0], widths[0])
        proj = pointwise(d, widths[0], out_channels, key=next(keys))
        return cls(
            lift=lift,
            down_blocks=down_blocks,
            res_blocks=res_blocks,
            inv_blocks=inv_blocks,
            up_blocks=up_blocks,
            proj_block=proj_block,
            proj=proj,
            num_spatial_dims=d,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            depth=depth,
            kernel_size=kernel_size,
            resolution=grid,
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        _check_example(x, self.num_spatial_dims)
        spatial = tuple(x.shape[1:])
        grid = spatial if self.resolution is None else self.resolution
        if self.resolution is None:
            _check_divisible(grid, self.depth, "input spatial extents")
        u = band_limited_resize(x, grid)  # identity when grid == spatial

        v = self.lift(u)
        skips = []
        for res, down in zip(self.res_blocks[:-1], self.down_blocks, strict=True):
            skips.append(_apply_all(res, v))
            v = down(v)
        h = _apply_all(self.res_blocks[-1], v)
        for inv, up, skip in zip(
            reversed(self.inv_blocks),
            reversed(self.up_blocks),
            reversed(skips),
            strict=True,
        ):
            h = einx.id("a ..., b ... -> (a + b) ...", up(inv(h)), skip)
        y = self.proj(self.proj_block(h))
        return band_limited_resize(y, spatial)


def _apply_all(blocks: list[CNOResBlock], x: Array) -> Array:
    """Apply ``blocks`` in sequence."""
    for block in blocks:
        x = block(x)
    return x


def _check_divisible(grid: tuple[int, ...], depth: int, what: str) -> None:
    """Raise unless every extent of ``grid`` is a positive multiple of ``2**depth``."""
    factor = 2**depth
    if any(n < factor or n % factor for n in grid):
        raise ValueError(
            f"{what} must be divisible by 2**depth = {factor}, got {grid}."
        )


ConvolutionalNeuralOperator = CNO
"""Verbose alias for `CNO`."""


__all__ = ["CNO", "CNOBlock", "CNOResBlock", "ConvolutionalNeuralOperator"]
