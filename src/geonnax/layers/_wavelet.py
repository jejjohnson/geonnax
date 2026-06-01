"""Discrete wavelet transform layers (the core of the Wavelet Neural Operator).

A multi-level, dimension-flexible discrete wavelet transform (DWT) and its
inverse, plus a `WaveletConv` that mixes channels in the wavelet domain
— the wavelet analogue of `SpectralConv`.

Analysis and synthesis are implemented as exact adjoints (circular boundary):
one DWT level correlates the signal with the decomposition filters and keeps
the even samples, while the inverse zero-upsamples and convolves with the same
filters. For an *orthonormal* wavelet the analysis operator is orthogonal, so
``idwt(dwt(x)) == x`` exactly, at any resolution with even spatial extents —
no boundary-dependent phase tuning. Each spatial axis must be divisible by
``2 ** level``.

Filter taps are hard-coded (no external dependency); the high-pass filter is
derived from the low-pass by the quadrature-mirror relation
``dec_hi[k] = (-1)**k * dec_lo[L - 1 - k]``.

References:
    Mallat (1989), "A theory for multiresolution signal decomposition".
    Tripura & Chakraborty (2023), "Wavelet Neural Operator".
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


Wavelet = Literal[
    "haar",
    "db1",
    "db2",
    "db3",
    "db4",
    "db5",
    "sym4",
    "sym5",
    "sym6",
    "coif1",
    "coif2",
]

# Orthonormal decomposition low-pass taps (sum == sqrt(2), unit L2 norm). The
# high-pass partner is derived by the quadrature-mirror relation, and synthesis
# reuses these same filters as the analysis adjoint, so only the low-pass taps
# need to be stored.
_DEC_LO: dict[str, tuple[float, ...]] = {
    "haar": (0.7071067811865476, 0.7071067811865476),
    "db2": (
        -0.1294095225512604,
        0.2241438680420134,
        0.8365163037378079,
        0.4829629131445341,
    ),
    "db3": (
        0.0352262918857095,
        -0.0854412738820267,
        -0.1350110200102546,
        0.4598775021184914,
        0.8068915093110924,
        0.3326705529500826,
    ),
    "db4": (
        -0.0105974017849973,
        0.0328830116668852,
        0.0308413818355607,
        -0.1870348117188811,
        -0.0279837694168599,
        0.6308807679295904,
        0.7148465705525415,
        0.2303778133088552,
    ),
    "db5": (
        0.003335725285001549,
        -0.012580751999015526,
        -0.006241490213011705,
        0.07757149384006515,
        -0.03224486958502952,
        -0.24229488706619015,
        0.13842814590110342,
        0.7243085284385744,
        0.6038292697974729,
        0.160102397974125,
    ),
    "sym4": (
        -0.0757657147892733,
        -0.0296355276459985,
        0.4976186676320155,
        0.8037387518059161,
        0.2978577956052774,
        -0.0992195435768472,
        -0.0126039672620378,
        0.0322231006040427,
    ),
    "sym5": (
        0.027333068345077982,
        0.029519490925774643,
        -0.039134249302383094,
        0.1993975339773936,
        0.7234076904024206,
        0.6339789634582119,
        0.01660210576452232,
        -0.17532808990845047,
        -0.021101834024758855,
        0.019538882735286728,
    ),
    "sym6": (
        0.015404109327027373,
        0.0034907120842174702,
        -0.11799011114819057,
        -0.048311742585633,
        0.4910559419267466,
        0.787641141030194,
        0.3379294217276218,
        -0.07263752278646252,
        -0.021060292512300564,
        0.04472490177066578,
        0.0017677118642428036,
        -0.007800708325034148,
    ),
    "coif1": (
        -0.01565572813546454,
        -0.0727326195128539,
        0.38486484686420286,
        0.8525720202122554,
        0.3378976624578092,
        -0.0727326195128539,
    ),
    "coif2": (
        -0.0007205494453645122,
        -0.0018232088707029932,
        0.0056114348193944995,
        0.023680171946334084,
        -0.0594344186464569,
        -0.0764885990783064,
        0.41700518442169254,
        0.8127236354455423,
        0.3861100668211622,
        -0.06737255472196302,
        -0.04146493678175915,
        0.016387336463522112,
    ),
}
_DEC_LO["db1"] = _DEC_LO["haar"]


def _filter_bank(wavelet: Wavelet) -> tuple[Array, Array]:
    """Return the orthonormal ``(dec_lo, dec_hi)`` analysis filters."""
    if wavelet not in _DEC_LO:
        raise ValueError(f"unknown wavelet {wavelet!r}; choose from {sorted(_DEC_LO)}.")
    dec_lo = jnp.asarray(_DEC_LO[wavelet])
    n = dec_lo.shape[0]
    # Quadrature-mirror high-pass: dec_hi[k] = (-1)^k dec_lo[L-1-k].
    signs = jnp.array([(-1.0) ** k for k in range(n)])
    dec_hi = signs * dec_lo[::-1]
    return dec_lo, dec_hi


def _axis_slice(ndim: int, axis: int, sl: slice) -> tuple[slice, ...]:
    """Index tuple selecting ``sl`` along ``axis`` and everything elsewhere."""
    idx: list[slice] = [slice(None)] * ndim
    idx[axis] = sl
    return tuple(idx)


def _analysis_1d(x: Array, h: Array, axis: int) -> Array:
    """Circular-correlate with ``h`` along ``axis``, then keep even samples."""
    # full[n] = sum_k h[k] * x[(n + k) mod N]  (circular correlation)
    terms = jnp.stack([h[k] * jnp.roll(x, -k, axis=axis) for k in range(h.shape[0])])
    full = jnp.sum(terms, axis=0)
    return full[_axis_slice(x.ndim, axis, slice(0, None, 2))]


def _synthesis_1d(c: Array, h: Array, axis: int) -> Array:
    """Adjoint of `_analysis_1d`: zero-upsample, then convolve with ``h``."""
    shape = list(c.shape)
    shape[axis] = 2 * c.shape[axis]
    up = jnp.zeros(shape, c.dtype)
    up = up.at[_axis_slice(c.ndim, axis, slice(0, None, 2))].set(c)
    # sum_k h[k] * roll(up, +k)  (adjoint of the correlation above)
    terms = jnp.stack([h[k] * jnp.roll(up, k, axis=axis) for k in range(h.shape[0])])
    return jnp.sum(terms, axis=0)


def _dwt_level(
    x: Array, dec_lo: Array, dec_hi: Array, axes: tuple[int, ...]
) -> dict[tuple[str, ...], Array]:
    """One DWT level over ``axes``, returning the ``2**len(axes)`` subbands.

    Keys are tuples of ``"a"`` (approximation/low-pass) and ``"d"``
    (detail/high-pass) per axis, e.g. ``("a", "d")`` is the LH subband in 2D.
    """
    subbands: dict[tuple[str, ...], Array] = {(): x}
    for ax in axes:
        nxt: dict[tuple[str, ...], Array] = {}
        for key, arr in subbands.items():
            nxt[(*key, "a")] = _analysis_1d(arr, dec_lo, ax)
            nxt[(*key, "d")] = _analysis_1d(arr, dec_hi, ax)
        subbands = nxt
    return subbands


def _idwt_level(
    subbands: dict[tuple[str, ...], Array],
    dec_lo: Array,
    dec_hi: Array,
    axes: tuple[int, ...],
) -> Array:
    """Inverse of `_dwt_level`."""
    cur = dict(subbands)
    for ax in reversed(axes):
        merged: dict[tuple[str, ...], Array] = {}
        for key in cur:
            prefix = key[:-1]
            if prefix in merged:
                continue
            lo = _synthesis_1d(cur[(*prefix, "a")], dec_lo, ax)
            hi = _synthesis_1d(cur[(*prefix, "d")], dec_hi, ax)
            merged[prefix] = lo + hi
        cur = merged
    return cur[()]


# A multi-level decomposition: the coarsest approximation plus one detail dict
# per level, ordered finest first (level 1, 2, ..., J).
Coefficients = tuple[Array, list[dict[tuple[str, ...], Array]]]


def dwt(
    x: Float[Array, "C *spatial"],
    wavelet: Wavelet,
    level: int,
    axes: tuple[int, ...],
) -> Coefficients:
    """Multi-level separable DWT over ``axes``.

    Args:
        x: Input of shape ``(channels, *spatial)``.
        wavelet: Wavelet name (e.g. ``"haar"``, ``"db4"``).
        level: Number of decomposition levels; each spatial extent must be
            divisible by ``2 ** level``.
        axes: Spatial axes to transform (e.g. ``(1, 2)`` for a 2D field).

    Returns:
        ``(approx, details)`` where ``approx`` is the coarsest low-pass band and
        ``details[j]`` holds the ``2**len(axes) - 1`` detail subbands of level
        ``j + 1`` (finest first).
    """
    dec_lo, dec_hi = _filter_bank(wavelet)
    approx = x
    all_a = ("a",) * len(axes)
    details: list[dict[tuple[str, ...], Array]] = []
    for _ in range(level):
        subbands = _dwt_level(approx, dec_lo, dec_hi, axes)
        approx = subbands.pop(all_a)
        details.append(subbands)
    return approx, details


def idwt(
    coeffs: Coefficients, wavelet: Wavelet, axes: tuple[int, ...]
) -> Float[Array, "C *spatial"]:
    """Invert `dwt` (exact for orthonormal wavelets)."""
    dec_lo, dec_hi = _filter_bank(wavelet)
    approx, details = coeffs
    all_a = ("a",) * len(axes)
    for subbands in reversed(details):
        full = {all_a: approx, **subbands}
        approx = _idwt_level(full, dec_lo, dec_hi, axes)
    return approx


def _flatten(coeffs: Coefficients) -> tuple[list[Array], list[list[tuple]]]:
    """Flatten coefficients to a list of subband arrays + a key layout."""
    approx, details = coeffs
    arrays = [approx]
    layout = []
    for subbands in details:
        keys = sorted(subbands)
        layout.append(keys)
        arrays.extend(subbands[k] for k in keys)
    return arrays, layout


def _unflatten(arrays: list[Array], layout: list[list[tuple]]) -> Coefficients:
    """Inverse of `_flatten`."""
    approx = arrays[0]
    details = []
    i = 1
    for keys in layout:
        details.append({k: arrays[i + j] for j, k in enumerate(keys)})
        i += len(keys)
    return approx, details


def _num_subbands(num_spatial_dims: int, level: int) -> int:
    """Total subband count: one approximation plus ``2**d - 1`` details/level."""
    return 1 + level * (2**num_spatial_dims - 1)


class WaveletConv(eqx.Module):
    """Wavelet-domain channel mixing — the wavelet analogue of ``SpectralConv``.

    Decomposes the input with a multi-level DWT, applies a learnable
    ``(out_channels, in_channels)`` linear channel mix to *each* wavelet subband
    (the approximation and every detail band), then reconstructs with the
    inverse DWT. Like a spectral convolution it is resolution-flexible (the
    weights are per-subband channel maps, independent of grid size); each
    spatial extent must be divisible by ``2 ** level``.

    Attributes:
        weights: One ``(out_channels, in_channels)`` matrix per subband, in
            flatten order (approximation first).
        bias: Optional spatial-domain bias of shape ``(out_channels, *ones)``.
        in_channels, out_channels: Channel counts (static).
        num_spatial_dims, wavelet, level: Transform configuration (static).
    """

    weights: list[Float[Array, "C_out C_in"]]
    bias: Float[Array, "C_out *ones"] | None
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)
    wavelet: Wavelet = eqx.field(static=True)
    level: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        *,
        key: Array,
        wavelet: Wavelet = "db4",
        level: int = 1,
        bias: bool = True,
        init_scale: float | None = None,
    ) -> WaveletConv:
        """Construct a wavelet convolution.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            num_spatial_dims: Number of spatial axes (1, 2, or 3).
            key: PRNG key.
            wavelet: Wavelet name — ``"haar"``, Daubechies ``"db2"``..``"db5"``,
                Symlets ``"sym4"``..``"sym6"``, or Coiflets ``"coif1"``/``"coif2"``
                (Coiflets have vanishing moments on the scaling function too, so
                they absorb smooth large-scale trends — handy for fields like SSH
                or SST).
            level: Number of DWT levels.
            bias: Whether to add a learnable spatial-domain bias.
            init_scale: Std of the channel-mix weights; defaults to the
                fan-based ``sqrt(2 / (in + out))``.

        Raises:
            ValueError: If ``level < 1`` or ``num_spatial_dims < 1``.
        """
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}.")
        if num_spatial_dims < 1:
            raise ValueError(f"num_spatial_dims must be >= 1, got {num_spatial_dims}.")
        scale = (
            math.sqrt(2.0 / (in_channels + out_channels))
            if init_scale is None
            else init_scale
        )
        n = _num_subbands(num_spatial_dims, level)
        keys = jax.random.split(key, n)
        weights = [
            scale * jax.random.normal(keys[i], (out_channels, in_channels))
            for i in range(n)
        ]
        bias_arr = jnp.zeros((out_channels, *(1,) * num_spatial_dims)) if bias else None
        return cls(
            weights=weights,
            bias=bias_arr,
            in_channels=in_channels,
            out_channels=out_channels,
            num_spatial_dims=num_spatial_dims,
            wavelet=wavelet,
            level=level,
        )

    def __call__(
        self, x: Float[Array, "C_in *spatial"]
    ) -> Float[Array, "C_out *spatial"]:
        axes = tuple(range(1, self.num_spatial_dims + 1))
        coeffs = dwt(x, self.wavelet, self.level, axes)
        arrays, layout = _flatten(coeffs)
        # Per-subband channel mix: (out, in) x (in, *sub_spatial) -> (out, ...).
        mixed = [
            jnp.einsum("oi,i...->o...", w, a)
            for a, w in zip(arrays, self.weights, strict=True)
        ]
        out = idwt(_unflatten(mixed, layout), self.wavelet, axes)
        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = ["Wavelet", "WaveletConv", "dwt", "idwt"]
