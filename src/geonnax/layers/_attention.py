"""Self-attention layers over flattened spatial grids.

Both layers are pre-norm and return the attention output only — the caller is
responsible for adding the residual.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers._conv import StandardizedConv
from geonnax.layers._utils import group_count


def _window_partition(x: Array, window: int) -> tuple[Array, tuple[int, ...]]:
    """Split ``(C, *spatial)`` into non-overlapping windows.

    Spatial extents must already be multiples of ``window``. Returns a
    ``(C, n_windows, window**d)`` array (windows batched, intra-window positions
    flattened) plus the per-axis window counts needed to invert it.
    """
    c, *spatial = x.shape
    d = len(spatial)
    counts = tuple(s // window for s in spatial)
    # (C, n_1, w, n_2, w, ..., n_d, w)
    interleaved = [c]
    for n in counts:
        interleaved += [n, window]
    x = x.reshape(interleaved)
    # -> (C, n_1, ..., n_d, w_1, ..., w_d)
    n_axes = [1 + 2 * i for i in range(d)]
    w_axes = [2 + 2 * i for i in range(d)]
    x = jnp.transpose(x, [0, *n_axes, *w_axes])
    return x.reshape(c, math.prod(counts), window**d), counts


def _window_merge(x: Array, counts: tuple[int, ...], window: int) -> Array:
    """Inverse of `_window_partition`, back to ``(C, *spatial)``."""
    c = x.shape[0]
    d = len(counts)
    x = x.reshape([c, *counts, *([window] * d)])
    # interleave back to (C, n_1, w_1, n_2, w_2, ...)
    perm = [0]
    for i in range(d):
        perm += [1 + i, 1 + d + i]
    x = jnp.transpose(x, perm)
    spatial = [n * window for n in counts]
    return x.reshape(c, *spatial)


class Attention(eqx.Module):
    """Pre-norm multi-head self-attention with cosine (QK-normalised) similarity.

    Flattens the spatial grid into a sequence, normalises queries/keys along
    the head dimension (Henry et al., 2020, "Query-Key Normalization") for
    stable, scale-invariant logits, then projects back to the channel grid.

    Attributes:
        norm: Pre-norm GroupNorm.
        to_qkv: ``1x1`` conv producing fused queries/keys/values.
        to_out: ``1x1`` conv projecting back to ``dim`` channels.
        heads: Number of attention heads.
        dim_head: Channels per head.
        scale: Fixed logit temperature applied to the cosine similarity.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import Attention
        >>> attn = Attention.init(16, 2, key=jr.PRNGKey(0), heads=2, dim_head=8)
        >>> attn(jnp.ones((16, 8, 8))).shape   # shape-preserving
        (16, 8, 8)
    """

    norm: eqx.nn.GroupNorm
    to_qkv: StandardizedConv
    to_out: StandardizedConv
    heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dim: int,
        num_spatial_dims: int,
        *,
        key: Array,
        heads: int = 4,
        dim_head: int = 32,
        groups: int = 8,
        scale: float = 10.0,
    ) -> Attention:
        """Construct a self-attention block."""
        k1, k2 = jax.random.split(key)
        inner = heads * dim_head
        norm = eqx.nn.GroupNorm(groups=group_count(dim, groups), channels=dim)
        to_qkv = StandardizedConv.init(
            num_spatial_dims, dim, inner * 3, 1, key=k1, use_bias=False
        )
        to_out = StandardizedConv.init(num_spatial_dims, inner, dim, 1, key=k2)
        return cls(
            norm=norm,
            to_qkv=to_qkv,
            to_out=to_out,
            heads=heads,
            dim_head=dim_head,
            scale=scale,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        spatial = x.shape[1:]
        h = self.norm(x)
        qkv = self.to_qkv(h).reshape(3, self.heads, self.dim_head, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (heads, dim_head, n)
        q = q / (jnp.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        k = k / (jnp.linalg.norm(k, axis=1, keepdims=True) + 1e-8)
        sim = jnp.einsum("h d i, h d j -> h i j", q, k) * self.scale
        attn = jax.nn.softmax(sim, axis=-1)
        out = jnp.einsum("h i j, h d j -> h d i", attn, v)
        out = out.reshape(self.heads * self.dim_head, *spatial)
        return self.to_out(out)


class LinearAttention(eqx.Module):
    """Pre-norm linear attention — ``O(n)`` in the number of spatial positions.

    A memory-frugal alternative to `Attention` for higher-resolution
    stages, using the softmax-feature factorisation of Shen et al. (2018,
    "Efficient Attention").

    Attributes:
        norm: Pre-norm GroupNorm.
        to_qkv: ``1x1`` conv producing fused queries/keys/values.
        to_out: ``1x1`` conv projecting back to ``dim`` channels.
        heads: Number of attention heads.
        dim_head: Channels per head.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import LinearAttention
        >>> attn = LinearAttention.init(16, 2, key=jr.PRNGKey(0), heads=2,
        ...                             dim_head=8)
        >>> attn(jnp.ones((16, 8, 8))).shape
        (16, 8, 8)
    """

    norm: eqx.nn.GroupNorm
    to_qkv: StandardizedConv
    to_out: StandardizedConv
    heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dim: int,
        num_spatial_dims: int,
        *,
        key: Array,
        heads: int = 4,
        dim_head: int = 32,
        groups: int = 8,
    ) -> LinearAttention:
        """Construct a linear-attention block."""
        k1, k2 = jax.random.split(key)
        inner = heads * dim_head
        norm = eqx.nn.GroupNorm(groups=group_count(dim, groups), channels=dim)
        to_qkv = StandardizedConv.init(
            num_spatial_dims, dim, inner * 3, 1, key=k1, use_bias=False
        )
        to_out = StandardizedConv.init(num_spatial_dims, inner, dim, 1, key=k2)
        return cls(
            norm=norm, to_qkv=to_qkv, to_out=to_out, heads=heads, dim_head=dim_head
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        spatial = x.shape[1:]
        h = self.norm(x)
        qkv = self.to_qkv(h).reshape(3, self.heads, self.dim_head, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (heads, dim_head, n)
        q = jax.nn.softmax(q, axis=1)
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("h d n, h e n -> h d e", k, v)
        out = jnp.einsum("h d e, h d n -> h e n", context, q)
        out = out.reshape(self.heads * self.dim_head, *spatial)
        return self.to_out(out)


class WindowedAttention(eqx.Module):
    """Swin-style windowed multi-head self-attention with cosine similarity.

    Like `Attention`, but attention is computed *within* non-overlapping
    ``window``-sized blocks of the spatial grid rather than globally, cutting
    cost from ``O(N**2)`` to ``O(N * window**d)`` per call (Liu et al., 2021,
    "Swin Transformer"). Spatial extents need not be multiples of ``window`` —
    the grid is cyclically padded to the next multiple and cropped back, which
    is natural for the periodic fields these layers target. Use it in place of
    `Attention` when the grid is too large for global attention.

    Attributes:
        norm: Pre-norm GroupNorm.
        to_qkv: ``1x1`` conv producing fused queries/keys/values.
        to_out: ``1x1`` conv projecting back to ``dim`` channels.
        heads: Number of attention heads.
        dim_head: Channels per head.
        window: Side length of the (square/cubic) attention window.
        scale: Fixed logit temperature applied to the cosine similarity.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from geonnax.layers import WindowedAttention
        >>> attn = WindowedAttention.init(16, 2, window=4, key=jr.PRNGKey(0),
        ...                               heads=2, dim_head=8)
        >>> attn(jnp.ones((16, 12, 12))).shape   # 12 not a multiple of 4
        (16, 12, 12)
    """

    norm: eqx.nn.GroupNorm
    to_qkv: StandardizedConv
    to_out: StandardizedConv
    heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    window: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dim: int,
        num_spatial_dims: int,
        *,
        window: int,
        key: Array,
        heads: int = 4,
        dim_head: int = 32,
        groups: int = 8,
        scale: float = 10.0,
    ) -> WindowedAttention:
        """Construct a windowed self-attention block with window side ``window``."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        k1, k2 = jax.random.split(key)
        inner = heads * dim_head
        norm = eqx.nn.GroupNorm(groups=group_count(dim, groups), channels=dim)
        to_qkv = StandardizedConv.init(
            num_spatial_dims, dim, inner * 3, 1, key=k1, use_bias=False
        )
        to_out = StandardizedConv.init(num_spatial_dims, inner, dim, 1, key=k2)
        return cls(
            norm=norm,
            to_qkv=to_qkv,
            to_out=to_out,
            heads=heads,
            dim_head=dim_head,
            window=window,
            scale=scale,
        )

    def __call__(self, x: Float[Array, "C *spatial"]) -> Float[Array, "C *spatial"]:
        spatial = x.shape[1:]
        qkv = self.to_qkv(self.norm(x))
        # Cyclically pad each spatial axis up to a multiple of the window.
        pads = [(0, 0)] + [(0, (-s) % self.window) for s in spatial]
        qkv = jnp.pad(qkv, pads, mode="wrap")
        windows, counts = _window_partition(qkv, self.window)  # (3*inner, B, M)
        b, m = windows.shape[1], windows.shape[2]
        windows = windows.reshape(3, self.heads, self.dim_head, b, m)
        q, k, v = windows[0], windows[1], windows[2]  # (heads, dim_head, B, M)
        q = q / (jnp.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        k = k / (jnp.linalg.norm(k, axis=1, keepdims=True) + 1e-8)
        sim = jnp.einsum("h d b i, h d b j -> h b i j", q, k) * self.scale
        attn = jax.nn.softmax(sim, axis=-1)
        out = jnp.einsum("h b i j, h d b j -> h d b i", attn, v)
        out = out.reshape(self.heads * self.dim_head, b, m)
        out = _window_merge(out, counts, self.window)  # (inner, *padded)
        out = out[(slice(None), *(slice(0, s) for s in spatial))]  # crop padding
        return self.to_out(out)


__all__ = ["Attention", "LinearAttention", "WindowedAttention"]
