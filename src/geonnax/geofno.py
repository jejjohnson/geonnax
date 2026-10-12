"""Geo-FNO (Li et al., JMLR 2023): neural operators on point clouds via a learned chart.

Geo-FNO handles point clouds on irregular, per-sample geometries by learning a
chart $\\xi = \\varphi(x)$ from the physical domain onto the unit torus
$[0, 1)^d$, where a standard FNO runs on a uniform latent grid. This module
holds the chart, `LearnedDeformation`; it acts on the ``(N, d)`` points of one
example (``jax.vmap`` over a batch of padded clouds).

References:
    Z. Li, D. Z. Huang, B. Liu and A. Anandkumar (2023), "Fourier Neural
    Operator with Learned Deformations for PDEs on General Geometries",
    Journal of Machine Learning Research 24(388), 1-26, arXiv:2207.05209.
    The reference chart is ``IPHI`` in ``neuraloperator/Geo-FNO``
    (``elasticity/elas_geofno.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, get_args

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax._basis import fourier_basis


__all__ = ["DeformationKind", "LearnedDeformation"]

DeformationKind = Literal["identity", "mlp", "fourier"]
"""The residual of a `LearnedDeformation`: none, an MLP, or a Laplacian eigenbasis."""

_KINDS: tuple[str, ...] = get_args(DeformationKind)


class LearnedDeformation(eqx.Module):
    r"""A learned chart from the physical domain onto the latent torus (Geo-FNO).

    The chart is an affine box map plus a residual. With the physical bounding
    box $[\ell, h] \subset \mathbb{R}^d$ (``lower``, ``upper``) and the latent
    extent $b = 1/(1 + \mathrm{pad})$,

    $$
    s(x) = b\,\frac{x - \ell}{h - \ell} \in [0, b]^d,
    \qquad
    \xi = \varphi(x) = s(x) + \delta\big(s(x)\big).
    $$

    The band $(b, 1)$ is a guard band: the latent FFT is periodic on
    $[0, 1)^d$, and the band keeps opposite edges of a non-periodic domain from
    coupling through the torus. The residual $\delta$ depends on ``kind``:

    - ``"identity"``: $\delta \equiv 0$.
    - ``"mlp"``: $\delta(s) = \mathrm{MLP}\big([s, \sin(Bs), \cos(Bs), c]\big)$,
      with NeRF features at the frozen frequencies $B_j = \pi\,2^j$,
      $j = 0, \dots, F - 1$, an optional per-sample code $c$, and the final
      linear layer initialised to zero.
    - ``"fourier"``: $\delta(s) = \Phi(s - b/2)\,\beta$, where
      $\Phi \in \mathbb{R}^{N \times M}$ holds the tensor-product Dirichlet
      Laplacian eigenfunctions on $[-b/2, b/2]^d$
      (`geonnax.basis.fourier_basis`, $M = \prod_a$ ``n_basis[a]``) and
      $\beta$ = ``coeffs`` $\in \mathbb{R}^{M \times d}$ starts at zero. The
      Dirichlet condition gives $\delta = 0$ on the box boundary, so the
      edges of the latent box stay fixed and the chart cannot push boundary
      points into the guard band.

    The residual is additive and zero at init, so every kind starts as the
    exact box map: $\varphi = s$, $\det \nabla\varphi = \prod_a
    \mathrm{scale}_a$. Unlike the reference ``IPHI`` (a multiplicative
    residual $x + x\,\delta(x)$, close to the identity only near the origin),
    this holds for any bounding box, e.g. lon/lat. As in the reference the
    chart maps physical to latent coordinates only, is never inverted and is
    not constrained to be invertible; `jacobian_det` is the diagnostic
    ($\det \nabla\varphi \le 0$ somewhere means a fold-over).

    ``lower``, ``scale`` and ``frequencies`` are frozen leaves (wrapped in
    `jax.lax.stop_gradient` at use), so a gradient never moves the box or
    the feature frequencies. The field name ``coeffs`` is API: pyrox swaps it
    for a sample site with a spectral prior over the eigenvalues
    $\lambda$ returned by `fourier_basis` (jejjohnson/pyrox#298).

    Pseudocode of `__call__`:

    ```text
    s  = (x - lower) * scale                        (N, d) in [0, b]^d
    identity: δ = 0
    mlp:      f = [s, sin(s ⊗ B), cos(s ⊗ B), code]  (N, d + 2dF + P)
              δ = vmap(MLP)(f)                      (N, d)
    fourier:  Φ, λ = fourier_basis(s - b/2, n_basis, L=b/2, "dirichlet")
              δ = Φ @ coeffs                        (N, M) @ (M, d)
    ξ  = s + δ                                      (N, d)
    ```

    Points outside the bounding box are mapped by the same formulas (the box
    map and the bases extrapolate); nothing clips them.

    Attributes:
        lower: Lower corner $\ell$ of the physical box, ``(d,)``; frozen.
        scale: $b / (h - \ell)$, ``(d,)``; frozen.
        mlp: The residual MLP (``kind="mlp"``), else ``None``.
        frequencies: The NeRF frequencies $\pi 2^j$, ``(F,)``
            (``kind="mlp"``), else ``None``; frozen.
        coeffs: Eigenbasis coefficients $\beta$, ``(M, d)``
            (``kind="fourier"``), else ``None``. Named API.
        kind: ``"identity"``, ``"mlp"`` or ``"fourier"``.
        n_basis: Eigenfunctions per axis (``kind="fourier"``), else ``()``.
        code_dim: Length ``P`` of the per-sample code (``0``: no code).
        extent: The latent extent $b = 1/(1 + \mathrm{pad})$.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from geonnax.geofno import LearnedDeformation
        >>> x = jax.random.uniform(jax.random.PRNGKey(1), (300, 2)) * 10.0
        >>> phi = LearnedDeformation.init(
        ...     [0.0, 0.0], [10.0, 10.0], key=jax.random.PRNGKey(0), width=32, depth=2
        ... )
        >>> xi = phi(x)                          # (N, d) -> (N, d)
        >>> xi.shape
        (300, 2)
        >>> bool(jnp.all((xi >= 0.0) & (xi <= phi.extent)))  # inside [0, b]^d
        True
        >>> phi.jacobian_det(x[:3]).round(5)     # prod(scale) = (b / 10)^2 at init
        Array([0.00826, 0.00826, 0.00826], dtype=float32)
        >>> fphi = LearnedDeformation.init(
        ...     [0.0, 0.0], [10.0, 10.0], kind="fourier", n_basis=(4, 6)
        ... )
        >>> fphi.coeffs.shape                    # (M, d), M = 4 * 6
        (24, 2)
        >>> cphi = LearnedDeformation.init(
        ...     [0.0, 0.0], [10.0, 10.0], key=jax.random.PRNGKey(0), width=32, depth=2,
        ...     code_dim=5,
        ... )
        >>> cphi(x, jnp.ones(5)).shape           # per-sample code (P,)
        (300, 2)
        >>> batch = jnp.stack([x, x])            # (B, N, d)
        >>> jax.vmap(phi)(batch).shape
        (2, 300, 2)
    """

    lower: Float[Array, " d"]
    scale: Float[Array, " d"]
    mlp: eqx.nn.MLP | None
    frequencies: Float[Array, " F"] | None
    coeffs: Float[Array, "M d"] | None
    kind: DeformationKind = eqx.field(static=True)
    n_basis: tuple[int, ...] = eqx.field(static=True)
    code_dim: int = eqx.field(static=True)
    extent: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        key: Array | None = None,
        kind: DeformationKind = "mlp",
        pad: float = 0.1,
        width: int = 128,
        depth: int = 4,
        n_frequencies: int = 8,
        n_basis: int | tuple[int, ...] = 8,
        code_dim: int = 0,
    ) -> LearnedDeformation:
        """Build a chart that is exactly the box map onto $[0, b]^d$.

        Args:
            lower: Lower corner of the physical bounding box, length ``d``.
            upper: Upper corner, length ``d``, with ``upper > lower``.
            key: PRNG key for the MLP weights; required for ``kind="mlp"``,
                unused otherwise (the other kinds draw nothing).
            kind: ``"identity"``, ``"mlp"`` or ``"fourier"``.
            pad: Guard-band fraction; the latent extent is
                ``b = 1 / (1 + pad)``. ``pad >= 0``.
            width: Hidden width of the MLP (``kind="mlp"``).
            depth: Number of hidden layers of the MLP (``kind="mlp"``).
            n_frequencies: Number ``F >= 1`` of NeRF frequencies $\\pi 2^j$
                (``kind="mlp"``).
            n_basis: Dirichlet eigenfunctions per axis (``kind="fourier"``);
                an ``int`` is broadcast to all ``d`` axes.
            code_dim: Length ``P`` of the per-sample code fed to the MLP;
                ``0`` for none. Only ``kind="mlp"`` takes a code.

        Returns:
            A `LearnedDeformation` whose residual is zero.

        Raises:
            ValueError: If ``lower`` / ``upper`` are empty, differ in length
                or have ``upper <= lower`` on some axis; if ``pad < 0``; if
                ``kind`` is unknown; if ``key`` is missing for
                ``kind="mlp"``; if ``width < 1``, ``depth < 0``,
                ``n_frequencies < 1``, ``code_dim < 0`` or an ``n_basis``
                entry is ``< 1`` or has the wrong length; or if
                ``code_dim > 0`` with a kind other than ``"mlp"``.

        Examples:
            >>> from geonnax.geofno import LearnedDeformation
            >>> phi = LearnedDeformation.init(
            ...     [-6.0, 35.0], [10.0, 45.0], kind="identity"
            ... )
            >>> phi.scale.round(4)                # b / (upper - lower)
            Array([0.0568, 0.0909], dtype=float32)
        """
        lo = [float(v) for v in lower]
        hi = [float(v) for v in upper]
        if not lo or len(lo) != len(hi):
            raise ValueError(
                "lower and upper must be non-empty and of equal length, got "
                f"lower={lower!r}, upper={upper!r}."
            )
        bad = [a for a, (l_a, h_a) in enumerate(zip(lo, hi, strict=True)) if h_a <= l_a]
        if bad:
            raise ValueError(
                f"upper must exceed lower on every axis; got lower={lo}, upper={hi} "
                f"(axes {bad})."
            )
        if not pad >= 0.0:
            raise ValueError(f"pad must be >= 0, got {pad}.")
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}.")
        if code_dim < 0:
            raise ValueError(f"code_dim must be >= 0, got {code_dim}.")
        if code_dim > 0 and kind != "mlp":
            raise ValueError(
                f"code_dim={code_dim} needs kind='mlp'; kind={kind!r} takes no code."
            )
        d = len(lo)
        b = 1.0 / (1.0 + float(pad))
        lower_arr = jnp.asarray(lo, dtype=float)
        scale = b / (jnp.asarray(hi, dtype=float) - lower_arr)

        mlp = frequencies = coeffs = None
        n_basis_t: tuple[int, ...] = ()
        if kind == "mlp":
            if key is None:
                raise ValueError("kind='mlp' needs a PRNG key, got key=None.")
            if width < 1:
                raise ValueError(f"width must be >= 1, got {width}.")
            if depth < 0:
                raise ValueError(f"depth must be >= 0, got {depth}.")
            if n_frequencies < 1:
                raise ValueError(f"n_frequencies must be >= 1, got {n_frequencies}.")
            frequencies = jnp.pi * 2.0 ** jnp.arange(n_frequencies, dtype=float)
            mlp = eqx.nn.MLP(
                in_size=d * (1 + 2 * n_frequencies) + code_dim,
                out_size=d,
                width_size=width,
                depth=depth,
                activation=jnp.tanh,
                key=key,
            )
            # Zero the readout so δ ≡ 0 and φ is the box map at init.
            readout = mlp.layers[-1]
            assert readout.bias is not None
            mlp = eqx.tree_at(
                lambda m: (m.layers[-1].weight, m.layers[-1].bias),
                mlp,
                (jnp.zeros_like(readout.weight), jnp.zeros_like(readout.bias)),
            )
        elif kind == "fourier":
            n_basis_t = (n_basis,) * d if isinstance(n_basis, int) else tuple(n_basis)
            if len(n_basis_t) != d or any(m < 1 for m in n_basis_t):
                raise ValueError(
                    f"n_basis must be an int >= 1 or {d} ints >= 1, got {n_basis!r}."
                )
            coeffs = jnp.zeros((math.prod(n_basis_t), d), dtype=float)

        return cls(
            lower=lower_arr,
            scale=scale,
            mlp=mlp,
            frequencies=frequencies,
            coeffs=coeffs,
            kind=kind,
            n_basis=n_basis_t,
            code_dim=code_dim,
            extent=b,
        )

    def box(self, x: Float[Array, "N d"]) -> Float[Array, "N d"]:
        """Apply the frozen affine box map $s(x) = (x - \\ell)\\,\\mathrm{scale}$.

        Args:
            x: Physical points, ``(N, d)``.

        Returns:
            Box-normalised points, ``(N, d)``, in $[0, b]^d$ for ``x`` in the box.
        """
        lower = jax.lax.stop_gradient(self.lower)
        scale = jax.lax.stop_gradient(self.scale)
        return einx.multiply("n d, d", einx.subtract("n d, d", x, lower), scale)

    def __call__(
        self, x: Float[Array, "N d"], code: Float[Array, " P"] | None = None
    ) -> Float[Array, "N d"]:
        """Map physical points to latent coordinates, $\\xi = s(x) + \\delta(s(x))$.

        Args:
            x: Physical points of one example, ``(N, d)``.
            code: Per-sample code, ``(P,)`` with ``P = code_dim``; must be
                ``None`` exactly when ``code_dim == 0``.

        Returns:
            Latent coordinates ``(N, d)``.

        Raises:
            ValueError: If ``x`` is not ``(N, d)`` or ``code`` does not match
                ``code_dim``.
        """
        d = self.lower.shape[0]
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError(f"x must have shape (N, {d}), got {x.shape}.")
        if (code is None) != (self.code_dim == 0):
            raise ValueError(
                "code must be given exactly when code_dim > 0; "
                f"code_dim={self.code_dim}, "
                f"code={'None' if code is None else code.shape}."
            )
        if code is not None and code.shape != (self.code_dim,):
            raise ValueError(
                f"code must have shape ({self.code_dim},), got {code.shape}."
            )
        s = self.box(x)
        if self.kind == "identity":
            return s
        if self.kind == "fourier":
            assert self.coeffs is not None
            half = 0.5 * self.extent
            phi, _ = fourier_basis(s - half, self.n_basis, half, boundary="dirichlet")
            return s + einx.dot("n m, m d -> n d", phi, self.coeffs)
        assert self.mlp is not None and self.frequencies is not None
        freqs = jax.lax.stop_gradient(self.frequencies)
        bs = einx.multiply("n d, f -> n (d f)", s, freqs)
        if code is None:
            feats = einx.id(
                "n d, n e, n e -> n (d + e + e)", s, jnp.sin(bs), jnp.cos(bs)
            )
        else:
            feats = einx.id(
                "n d, n e, n e, p -> n (d + e + e + p)",
                s,
                jnp.sin(bs),
                jnp.cos(bs),
                code,
            )
        return s + jax.vmap(self.mlp)(feats)

    def jacobian_det(
        self, x: Float[Array, "N d"], code: Float[Array, " P"] | None = None
    ) -> Float[Array, " N"]:
        """Per-point Jacobian determinant $\\det \\nabla\\varphi(x_n)$.

        A diagnostic: at init it equals $\\prod_a \\mathrm{scale}_a$ at every
        point, and $\\det \\nabla\\varphi \\le 0$ anywhere means the chart
        folds over there (no claim is made that it never does). It is also
        the change-of-measure factor $d\\xi = |\\det \\nabla\\varphi|\\,dx$ for
        latent quadrature weights (#85).

        Computed as ``jax.vmap(jax.jacfwd(φ))`` over single points: ``d``
        forward-mode JVPs of a one-point forward pass per point, plus an
        ``O(d^3)`` determinant, i.e. about ``d`` times the cost of `__call__`
        and ``(N, d, d)`` memory for the Jacobians.

        Args:
            x: Physical points, ``(N, d)``.
            code: Per-sample code, as for `__call__`.

        Returns:
            Determinants, ``(N,)``.
        """

        def point(xn: Float[Array, " d"]) -> Float[Array, " d"]:
            return self(einx.id("d -> 1 d", xn), code)[0]

        return jnp.linalg.det(jax.vmap(jax.jacfwd(point))(x))
