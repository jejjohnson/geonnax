"""Self-attention layers over flattened spatial grids.

Both layers are pre-norm and return the attention output only — the caller is
responsible for adding the residual.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax.layers._conv import StandardizedConv
from geonnax.layers._utils import group_count


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


__all__ = ["Attention", "LinearAttention"]
