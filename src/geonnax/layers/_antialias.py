r"""Band-limited resampling and an aliasing-reduced pointwise activation.

The two primitives that make a convolutional network act between spaces of
band-limited functions, as in the Convolutional Neural Operator (CNO; Raonić et
al., NeurIPS 2023, arXiv:2302.01178, SM A.4):

- `band_limited_resize` changes the sampling rate of a periodic field by
  truncating or zero-padding its Fourier coefficients (an ideal low-pass);
- `AntiAliasedActivation` evaluates a pointwise nonlinearity on a finer grid
  and low-passes the result back, the up → σ → down construction of
  Alias-Free GAN (Karras et al., NeurIPS 2021, arXiv:2106.12423).

Unlike `Downsample` / `Upsample` (lossless space-to-depth channel
rearrangements), these change the number of samples of the *same* function.
Both assume a periodic domain $[0, 1)^d$ sampled at $x_j = j / N$.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _band_slices(n_in: int, n_out: int, *, last: bool) -> list[tuple[slice, slice]]:
    """``(input, output)`` slice pairs of the modes one axis keeps when resampled.

    An axis whose extent is unchanged keeps every bin. Otherwise only the
    wavenumbers with ``|k| < min(n_in, n_out) / 2`` are kept, so the Nyquist
    bin of an even grid (``k = ±n/2``, which has no conjugate partner) is
    dropped and the kept spectrum stays Hermitian. The rfft-halved last axis
    holds only ``k >= 0``; every other axis holds ``0..m-1`` and ``-(m-1)..-1``.
    """
    if n_in == n_out:
        return [(slice(None), slice(None))]
    m = (min(n_in, n_out) + 1) // 2  # number of k >= 0 with |k| < min / 2
    pairs = [(slice(0, m), slice(0, m))]
    if not last and m > 1:
        pairs.append((slice(n_in - m + 1, n_in), slice(n_out - m + 1, n_out)))
    return pairs


def band_limited_resize(
    x: Float[Array, "C *spatial"], out_spatial: tuple[int, ...]
) -> Float[Array, "C *out_spatial"]:
    r"""Resample a periodic field onto a new grid by Fourier truncation / zero-padding.

    Treat the samples $x_j$ on the grid $j / N$ of $[0, 1)^d$ as the
    trigonometric polynomial with the same discrete Fourier coefficients
    $\hat x_k = \frac{1}{\prod_a N_a} \sum_j x_j e^{-2\pi i\, k\cdot j / N}$
    and evaluate it on the grid $j / M$:

    $$y_j = \sum_{k \in K} \hat x_k\, e^{2\pi i\, k\cdot j / M},
    \qquad K = \prod_a K_a,$$

    with $K_a$ every bin of axis $a$ when $M_a = N_a$, and otherwise
    $K_a = \{k : |k| < \min(N_a, M_a) / 2\}$. Downsampling crops the high
    modes (an ideal low-pass), upsampling zero-pads them (band-limited
    interpolation). The Nyquist bin of an even grid is dropped on a resized
    axis because it has no conjugate partner on the other grid.

    ```text
    x̂ = rfftn(x, norm="forward")                 # continuous Fourier coefficients
    ŷ = zeros(C, *M[:-1], M[-1] // 2 + 1)
    for (in, out) in product(per-axis kept-mode slice pairs):
        ŷ[out] = x̂[in]                           # truncate or zero-pad
    return irfftn(ŷ, s=M, norm="forward")
    ```

    Consequences: a field whose spectrum lies in $|k_a| < \min(N_a, M_a)/2$
    is resampled exactly (the samples of the analytic field on the new grid),
    and resizing up then back down is the identity on any field without
    Nyquist content. Down then up is a projection, not an identity.

    The domain is assumed periodic: a non-periodic field has a jump at the
    boundary and resampling it rings (Gibbs). The resampling is the ideal
    (sinc) filter in the Fourier domain; CNO's reference implementation uses
    windowed-sinc filters instead (Raonić et al., NeurIPS 2023,
    arXiv:2302.01178, SM A.4 and C.1.4).

    Args:
        x: One channel-first example, shape ``(C, *spatial)``.
        out_spatial: Target grid extents, one per spatial axis of ``x``.

    Returns:
        The resampled field, shape ``(C, *out_spatial)``, in the floating
        dtype of the transform (``x``'s dtype for a floating ``x``).

    Raises:
        ValueError: If ``x`` has no spatial axis, ``out_spatial`` has a
            different length than ``x.shape[1:]``, or any target extent is
            below 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax.layers import band_limited_resize
        >>> s = jnp.arange(16) / 16
        >>> x = jnp.stack([jnp.sin(2 * jnp.pi * 3 * s)])         # (C, N) = (1, 16)
        >>> y = band_limited_resize(x, (40,))                    # (1, 40)
        >>> t = jnp.arange(40) / 40
        >>> bool(jnp.allclose(y[0], jnp.sin(2 * jnp.pi * 3 * t), atol=1e-5))
        True
        >>> band_limited_resize(jnp.ones((2, 32, 24)), (16, 48)).shape
        (2, 16, 48)
    """
    in_spatial = tuple(x.shape[1:])
    out_spatial = tuple(int(s) for s in out_spatial)
    if not in_spatial:
        raise ValueError(f"x must have shape (C, *spatial), got {x.shape}.")
    if len(out_spatial) != len(in_spatial):
        raise ValueError(
            f"out_spatial must have {len(in_spatial)} extents (one per spatial "
            f"axis of x), got {out_spatial}."
        )
    if any(s < 1 for s in out_spatial):
        raise ValueError(f"all out_spatial extents must be >= 1, got {out_spatial}.")
    if out_spatial == in_spatial:
        return x
    d = len(in_spatial)
    axes = tuple(range(1, d + 1))
    x_ft = jnp.fft.rfftn(x, axes=axes, norm="forward")
    per_axis = [
        _band_slices(n, m, last=a == d - 1)
        for a, (n, m) in enumerate(zip(in_spatial, out_spatial, strict=True))
    ]
    y_ft = jnp.zeros(
        (x.shape[0], *out_spatial[:-1], out_spatial[-1] // 2 + 1), dtype=x_ft.dtype
    )
    for combo in itertools.product(*per_axis):
        src = (slice(None), *(pair[0] for pair in combo))
        dst = (slice(None), *(pair[1] for pair in combo))
        y_ft = y_ft.at[dst].set(x_ft[src])
    return jnp.fft.irfftn(y_ft, s=out_spatial, axes=axes, norm="forward")


class AntiAliasedActivation(eqx.Module):
    r"""A pointwise nonlinearity between band-limited function spaces.

    The CNO activation operator (Raonić et al., NeurIPS 2023, arXiv:2302.01178,
    SM A.4), after the alias-free activation of Karras et al.
    (*Alias-Free Generative Adversarial Networks*, NeurIPS 2021,
    arXiv:2106.12423): with $U_r$ the band-limited resize by an integer factor
    $r$ = ``up`` along every spatial axis and $D_r = U_r^{-1}$ its ideal
    low-pass downsample (`band_limited_resize`),

    $$\Sigma(x) = D_r\, \sigma\big(U_r\, x\big).$$

    ```text
    fine = tuple(up * n for n in spatial)
    v = band_limited_resize(x, fine)         # band-limited interpolation
    v = activation(v)                        # pointwise, creates new harmonics
    return band_limited_resize(v, spatial)   # ideal low-pass + decimate
    ```

    A pointwise $\sigma$ on the original grid folds every harmonic above the
    Nyquist frequency back into the retained band. On the $r$-times finer grid
    only harmonics above $(r - \tfrac12) N$ fold into $|k| < N/2$, and those
    decay fast for a smooth $\sigma$, so the output approximates the
    band-limited projection of the continuous $\sigma(x)$ far better (see
    `tests/test_antialias.py`). This *reduces* aliasing rather than
    eliminating it, and assumes a periodic domain. A network built from it
    is still effectively tied to its training resolution (interpolate to that
    grid, evaluate, interpolate back): the construction buys robustness to
    the grid, not free zero-shot super-resolution. CNO uses ``up=2`` (SM A.4)
    and windowed-sinc filters; this layer uses the ideal Fourier filter.

    Attributes:
        activation: The pointwise nonlinearity (static).
        up: Integer upsampling factor of the grid the nonlinearity is
            evaluated on (static).
        num_spatial_dims: Number of trailing spatial axes (static).

    Examples:
        >>> import jax, jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import AntiAliasedActivation
        >>> act = AntiAliasedActivation.init(2)                  # GELU, up=2
        >>> act(jr.normal(jr.PRNGKey(0), (4, 32, 32))).shape
        (4, 32, 32)
        >>> act = AntiAliasedActivation.init(1, activation=jax.nn.relu, up=4)
        >>> jax.vmap(act)(jnp.ones((8, 3, 64))).shape            # batch of 8
        (8, 3, 64)
    """

    activation: Callable[[Array], Array] = eqx.field(static=True)
    up: int = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        num_spatial_dims: int,
        *,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        up: int = 2,
    ) -> AntiAliasedActivation:
        """Construct an anti-aliased activation (no parameters, so no key).

        Args:
            num_spatial_dims: Number of trailing spatial axes of the input.
            activation: Pointwise nonlinearity; GELU by default (CNO uses
                leaky ReLU, Raonić et al., NeurIPS 2023).
            up: Integer factor of the finer grid the nonlinearity is
                evaluated on; ``1`` reduces to the plain pointwise ``activation``.

        Returns:
            The activation module.

        Raises:
            ValueError: If ``num_spatial_dims`` or ``up`` is below 1.
        """
        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims must be >= 1, got {num_spatial_dims}.")
        if up < 1:
            raise ValueError(f"up must be an integer >= 1, got {up}.")
        return cls(
            activation=activation, up=int(up), num_spatial_dims=int(num_spatial_dims)
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        if x.ndim != 1 + self.num_spatial_dims:
            raise ValueError(
                f"expected (C, *spatial) with {self.num_spatial_dims} spatial "
                f"axes, got shape {x.shape}."
            )
        spatial = tuple(x.shape[1:])
        fine = tuple(self.up * n for n in spatial)
        v = self.activation(band_limited_resize(x, fine))
        return band_limited_resize(v, spatial)


__all__ = ["AntiAliasedActivation", "band_limited_resize"]
