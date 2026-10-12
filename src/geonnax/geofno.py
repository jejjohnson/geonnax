"""Geo-FNO (Li et al., JMLR 2023): neural operators on point clouds via a learned chart.

Geo-FNO handles point clouds on irregular, per-sample geometries by learning a
chart $\\xi = \\varphi(x)$ from the physical domain onto the unit torus
$[0, 1)^d$, where a standard FNO runs on a uniform latent grid. This module
holds the chart, `LearnedDeformation`, and the operator, `GeoFNO`; both act on
the ``(N, d)`` points of one example (``jax.vmap`` over a batch of padded
clouds).

References:
    Z. Li, D. Z. Huang, B. Liu and A. Anandkumar (2023), "Fourier Neural
    Operator with Learned Deformations for PDEs on General Geometries",
    Journal of Machine Learning Research 24(388), 1-26, arXiv:2207.05209.
    The reference chart is ``IPHI`` in ``neuraloperator/Geo-FNO``
    (``elasticity/elas_geofno.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, get_args

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax._basis import fourier_basis
from geonnax.fno import FNOBlock
from geonnax.layers._factorized import Factorization
from geonnax.layers._spectral import SpectralConv
from geonnax.layers._utils import pointwise


__all__ = ["DeformationKind", "GeoFNO", "LearnedDeformation"]

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


class GeoFNO(eqx.Module):
    r"""Geo-FNO: a Fourier neural operator on per-sample point clouds.

    Reads a field $u$ at $N$ points $x_n$ with quadrature weights $w_n$, maps
    the points onto the latent torus with the learned chart
    $\xi = \varphi(x)$ (`LearnedDeformation`), runs FNO blocks on a uniform
    latent grid $G$ (shape ``latent_shape``, nodes $j / s_a$ on $[0, 1)^d$),
    and evaluates the result at $M$ query points $x'_m$ (the input points by
    default):

    $$
    \begin{aligned}
    h_0 &= \sigma\big(\mathcal{K}_0[P u;\ \xi, w] + b_0(G)\big)
        && \text{points} \to G, \\
    h_\ell &= \sigma\big(\mathcal{K}_\ell h_{\ell-1} + W_\ell h_{\ell-1}
        + b_\ell(G)\big), \quad \ell = 1, \dots, L-2
        && \text{on } G, \\
    y &= Q_2\, \sigma\big(Q_1 (\mathcal{K}_{L-1} h_{L-2}\big|_{\xi'}
        + b_{L-1}(\xi'))\big), \quad \xi' = \varphi(x')
        && G \to \text{points}.
    \end{aligned}
    $$

    $P$, $Q_1$, $Q_2$ and $W_\ell$ are pointwise linear maps; $\mathcal{K}_0$
    is `SpectralConv.from_points` (the weighted non-uniform Fourier
    transform $\hat v_k = \sum_n w_n (P u)_n e^{-2\pi i k\cdot\xi_n}$,
    mixed and inverse-FFT-ed onto $G$), $\mathcal{K}_\ell$ and $W_\ell$ form
    a `geonnax.fno.FNOBlock`, and $\mathcal{K}_{L-1}$ is
    `SpectralConv.to_points` (the truncated Fourier series evaluated at
    $\xi'$). The positional bias $b_\ell$ is a 1×1 conv of the latent
    coordinates, added in every layer as in the reference
    (``b0``…``b4`` in ``elas_geofno.py``): it tells the network where it is
    on the torus once $\varphi$ moves points. With ``jacobian_weights=True``
    the weights become the latent measure
    $w^\xi_n = w_n |\det\nabla\varphi(x_n)| / \sum_m w_m
    |\det\nabla\varphi(x_m)|$ (off by default; jejjohnson/geonnax#85 decides).

    Pseudocode of `__call__` (``H = hidden_channels``):

    ```text
    ξ  = φ(x, code)                                         (N, d)
    w  = weights                                            (N,)
    if jacobian_weights: w = w·|det∇φ(x)| / Σ w·|det∇φ(x)|
    h  = P u                                                (H, N)
    h  = σ(encoder.from_points(h, ξ, w, G) + b_0(G))        (H, *G)
    for ℓ, block in enumerate(blocks, 1):                   L - 2 blocks
        h = σ(block(h) + b_ℓ(G))   # block = K_ℓ h + W_ℓ h, no activation
    ξ' = φ(x_out, code)                                     (M, d)
    h  = decoder.to_points(h, ξ') + b_out(ξ')               (H, M)
    y  = Q_2 σ(Q_1 h)                                       (C_out, M)
    ```

    **Ragged batches.** Clouds of different sizes are batched by padding:

    - pad every sample to $N_{\max}$ points with any finite in-box
      coordinates, ``u = 0`` and ``w = 0``;
    - ``jax.vmap(model)(u, x, w)``;
    - mask the loss with ``w > 0``.

    A zero-weight point contributes exact zeros to the encoder, so the
    output at the real points does not depend on the padding. The latent
    grid and the modes are static: ``latent_shape`` and ``n_modes`` cannot
    vary per sample.

    Deviations from the reference implementation (``neuraloperator/Geo-FNO``):

    - **quadrature weights**: the reference encoder sums
      $\sum_n u_n e^{-2\pi i k\cdot\xi_n}$ unweighted, which grows with $N$;
      here $w_n$ makes the coefficients a quadrature of the Fourier integral,
      so the operator converges as the cloud is refined;
    - **additive, zero-initialised chart**: $\varphi$ is exactly the box map
      at init for any bounding box (`LearnedDeformation`), where the
      reference's $x + x\,\delta(x)$ is near the identity only near the
      origin;
    - **latent guard band**: the box maps into $[0, b]^d$,
      $b = 1/(1 + \mathrm{pad})$, so opposite edges of a non-periodic domain
      do not couple through the periodic latent FFT.

    Cost per example: two dense non-uniform transforms,
    $O(N\,|\mathcal{K}|\,H)$ with $|\mathcal{K}|$ the retained modes, plus
    $L - 2$ FFT blocks, $O(H\, S \log S)$ with $S = \prod_a s_a$. Memory is
    dominated by the $N \times |\mathcal{K}|$ exponential table of the
    encoder; wrap the model in ``jax.checkpoint`` to trade it for compute.

    Reference:
        Z. Li, D. Z. Huang, B. Liu and A. Anandkumar (2023), "Fourier Neural
        Operator with Learned Deformations for PDEs on General Geometries",
        Journal of Machine Learning Research 24(388), 1-26, arXiv:2207.05209.

    Attributes:
        lifting: 1-D 1×1 conv over the points, ``in_channels -> H``.
        encoder: `SpectralConv` applied with `SpectralConv.from_points`.
        blocks: The ``n_layers - 2`` latent blocks (`geonnax.fno.FNOBlock`,
            built with ``use_activation=False``: the activation follows the
            positional bias).
        decoder: `SpectralConv` applied with `SpectralConv.to_points`.
        pos_bias: One ``d``-D 1×1 conv ``d -> H`` of the latent grid
            coordinates per latent layer (``n_layers - 1``), or ``None``.
        pos_bias_out: 1-D 1×1 conv ``d -> H`` of $\xi'$ at the query points,
            or ``None``.
        proj1, proj2: 1-D 1×1 convs ``H -> projection_channels -> out_channels``.
        deformation: The chart $\varphi$ (`LearnedDeformation`).
        activation: Pointwise nonlinearity $\sigma$.
        latent_shape: Latent grid extents ``(s_1, …, s_d)``.
        jacobian_weights: Whether to reweight by $|\det\nabla\varphi|$.
        in_channels, out_channels, hidden_channels: Channel counts.

    Examples:
        >>> import einx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from geonnax.geofno import GeoFNO
        >>> model = GeoFNO.init(
        ...     1, 2, (6, 6), (24, 24), key=jr.PRNGKey(0),
        ...     lower=[0.0, 0.0], upper=[10.0, 5.0], hidden_channels=8, n_layers=3,
        ...     deformation_kwargs={"width": 16, "depth": 1},
        ... )
        >>> box = jnp.array([10.0, 5.0])
        >>> x = einx.multiply("n d, d", jr.uniform(jr.PRNGKey(1), (300, 2)), box)
        >>> u = einx.id("n 1 -> 1 n", jnp.sin(x[:, :1]))   # (C_in, N)
        >>> w = jnp.full((300,), 1 / 300)                   # quadrature weights
        >>> model(u, x, w).shape                            # (C_out, N)
        (2, 300)
        >>> model(u, x, w, x_out=x[:50]).shape              # (C_out, M)
        (2, 50)

        A ragged batch: clouds of 300 and 200 real points, zero-padded to 300:

        >>> n_real = jnp.array([300, 200])
        >>> real = einx.less("n, b -> b n", jnp.arange(300), n_real)  # (B, N)
        >>> wb = einx.where("b n, b, -> b n", real, 1.0 / n_real, 0.0)
        >>> ub = einx.where("b n, c n, -> b c n", real, u, 0.0)
        >>> xb = jnp.stack([x, x])
        >>> yb = jax.vmap(model)(ub, xb, wb)                # mask the loss by w > 0
        >>> yb.shape
        (2, 2, 300)
    """

    lifting: eqx.nn.Conv
    encoder: SpectralConv
    blocks: list[FNOBlock]
    decoder: SpectralConv
    pos_bias: list[eqx.nn.Conv] | None
    pos_bias_out: eqx.nn.Conv | None
    proj1: eqx.nn.Conv
    proj2: eqx.nn.Conv
    deformation: LearnedDeformation
    activation: Callable[[Array], Array] = eqx.field(static=True)
    latent_shape: tuple[int, ...] = eqx.field(static=True)
    jacobian_weights: bool = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        latent_shape: tuple[int, ...],
        *,
        key: Array,
        lower: Sequence[float],
        upper: Sequence[float],
        hidden_channels: int = 32,
        n_layers: int = 4,
        projection_channels: int | None = None,
        deformation: DeformationKind = "mlp",
        deformation_kwargs: dict | None = None,
        positional_bias: bool = True,
        jacobian_weights: bool = False,
        factorization: Factorization = "dense",
        rank: float | int = 0.5,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        code_dim: int = 0,
    ) -> GeoFNO:
        """Construct a Geo-FNO whose chart starts as the box map.

        Args:
            in_channels: Number of input channels ``C_in``.
            out_channels: Number of output channels ``C_out``.
            n_modes: Retained Fourier modes per latent axis; its length is the
                dimension ``d``.
            latent_shape: Latent grid extents, one per axis, each greater than
                ``2 * n_modes``.
            key: PRNG key.
            lower: Lower corner of the physical bounding box, length ``d``.
            upper: Upper corner of the physical bounding box, length ``d``.
            hidden_channels: Latent channel width ``H``.
            n_layers: Spectral layers, ``>= 2``: the encoder, ``n_layers - 2``
                latent blocks (`geonnax.fno.FNOBlock`) and the decoder.
            projection_channels: Hidden width of the projection (defaults to
                ``hidden_channels``).
            deformation: Kind of chart residual, ``"identity"``, ``"mlp"`` or
                ``"fourier"`` (see `LearnedDeformation`).
            deformation_kwargs: Extra keyword arguments for
                `LearnedDeformation.init` (``pad``, ``width``, ``depth``,
                ``n_frequencies``, ``n_basis``).
            positional_bias: Add the 1×1 conv of the latent coordinates in
                every layer (as the reference does).
            jacobian_weights: Reweight the quadrature by
                $|\\det\\nabla\\varphi|$ (jejjohnson/geonnax#85).
            factorization: Spectral-weight parameterisation (``"dense"`` /
                ``"cp"`` / ``"tucker"`` / ``"tt"``) of every spectral conv.
            rank: Rank for the factorised forms; ignored when dense.
            activation: Pointwise nonlinearity.
            code_dim: Length ``P`` of the per-sample chart code (``0``: none;
                needs ``deformation="mlp"``).

        Returns:
            A `GeoFNO`.

        Raises:
            ValueError: If ``n_layers < 2``; if ``n_modes`` is empty or
                ``n_modes``, ``latent_shape``, ``lower`` and ``upper`` differ
                in length; if some ``latent_shape[a] <= 2 * n_modes[a]``; or
                for an invalid chart configuration (`LearnedDeformation.init`).
        """
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}.")
        n_modes = tuple(int(m) for m in n_modes)
        latent_shape = tuple(int(s) for s in latent_shape)
        d = len(n_modes)
        if d == 0 or not d == len(latent_shape) == len(lower) == len(upper):
            raise ValueError(
                "n_modes, latent_shape, lower and upper must be non-empty and of "
                f"equal length, got n_modes={n_modes}, latent_shape={latent_shape}, "
                f"lower={lower!r}, upper={upper!r}."
            )
        if any(2 * m >= s for m, s in zip(n_modes, latent_shape, strict=True)):
            raise ValueError(
                "each latent_shape extent must exceed 2 * n_modes = "
                f"{tuple(2 * m for m in n_modes)}, got latent_shape={latent_shape}."
            )
        projection_channels = projection_channels or hidden_channels

        (k_lift, k_enc, k_blocks, k_dec, k_pos, k_pos_out, k_p1, k_p2, k_def) = (
            jax.random.split(key, 9)
        )
        deform = LearnedDeformation.init(
            lower,
            upper,
            key=k_def,
            kind=deformation,
            code_dim=code_dim,
            **(deformation_kwargs or {}),
        )
        lifting = pointwise(1, in_channels, hidden_channels, key=k_lift)
        encoder, decoder = (
            SpectralConv.init(
                hidden_channels,
                hidden_channels,
                n_modes,
                key=k,
                factorization=factorization,
                rank=rank,
            )
            for k in (k_enc, k_dec)
        )
        blocks = [
            FNOBlock.init(
                hidden_channels,
                n_modes,
                key=k,
                factorization=factorization,
                rank=rank,
                activation=activation,
                use_activation=False,
            )
            for k in jax.random.split(k_blocks, n_layers - 2)
        ]
        pos_bias = pos_bias_out = None
        if positional_bias:
            pos_bias = [
                pointwise(d, d, hidden_channels, key=k)
                for k in jax.random.split(k_pos, n_layers - 1)
            ]
            pos_bias_out = pointwise(1, d, hidden_channels, key=k_pos_out)
        proj1 = pointwise(1, hidden_channels, projection_channels, key=k_p1)
        proj2 = pointwise(1, projection_channels, out_channels, key=k_p2)
        return cls(
            lifting=lifting,
            encoder=encoder,
            blocks=blocks,
            decoder=decoder,
            pos_bias=pos_bias,
            pos_bias_out=pos_bias_out,
            proj1=proj1,
            proj2=proj2,
            deformation=deform,
            activation=activation,
            latent_shape=latent_shape,
            jacobian_weights=jacobian_weights,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def _latent_coords(self, dtype: jnp.dtype) -> Float[Array, "d *G"]:
        """The latent grid nodes ``j / s_a`` on [0, 1)^d, channel-first."""
        axes = [jnp.arange(s, dtype=dtype) / s for s in self.latent_shape]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"))

    def _add_pos(
        self, h: Float[Array, "H *G"], layer: int, grid: Float[Array, "d *G"]
    ) -> Float[Array, "H *G"]:
        """Add the positional bias $b_\\ell(G)$ of latent layer ``layer``, if any."""
        if self.pos_bias is None:
            return h
        return h + self.pos_bias[layer](grid)

    def __call__(
        self,
        u: Float[Array, "C_in N"],
        x: Float[Array, "N d"],
        weights: Float[Array, " N"],
        *,
        x_out: Float[Array, "M d"] | None = None,
        code: Float[Array, " P"] | None = None,
    ) -> Float[Array, "C_out M"]:
        """Apply the operator to one point cloud.

        Args:
            u: Field values at the points, ``(C_in, N)``; ``0`` at padding.
            x: Physical point coordinates, ``(N, d)``; finite (in-box) at
                padding.
            weights: Quadrature weights, ``(N,)``, approximating cell volumes
                ($w_n \\ge 0$, $\\sum_n w_n \\approx 1$); ``0`` marks padding.
            x_out: Physical query points, ``(M, d)``; ``None`` evaluates at
                ``x``.
            code: Per-sample chart code, ``(P,)``, exactly when
                ``code_dim > 0``.

        Returns:
            The output field at the query points, ``(C_out, M)``.

        Raises:
            ValueError: If the shapes of ``u``, ``x``, ``weights`` or
                ``x_out`` disagree with each other or with the model, or
                ``code`` does not match ``code_dim``.
        """
        d = len(self.latent_shape)
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError(f"x must have shape (N, {d}), got {x.shape}.")
        n = x.shape[0]
        if u.shape != (self.in_channels, n):
            raise ValueError(
                f"u must have shape (in_channels, N) = ({self.in_channels}, {n}), "
                f"got {u.shape}."
            )
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}.")
        if x_out is not None and (x_out.ndim != 2 or x_out.shape[1] != d):
            raise ValueError(f"x_out must have shape (M, {d}), got {x_out.shape}.")

        xi = self.deformation(x, code)  # (N, d)
        w = weights
        if self.jacobian_weights:
            w = w * jnp.abs(self.deformation.jacobian_det(x, code))
            w = w / jnp.maximum(einx.sum("[n]", w), jnp.finfo(w.dtype).tiny)
        h = self.lifting(u)  # (H, N)
        h = self.encoder.from_points(h, xi, w, self.latent_shape)  # (H, *G)
        grid = self._latent_coords(h.dtype)  # (d, *G)
        h = self.activation(self._add_pos(h, 0, grid))
        for i, block in enumerate(self.blocks, start=1):
            h = self.activation(self._add_pos(block(h), i, grid))  # K h + W h + b
        xi_out = xi if x_out is None else self.deformation(x_out, code)  # (M, d)
        h = self.decoder.to_points(h, xi_out)  # (H, M)
        if self.pos_bias_out is not None:
            h = h + self.pos_bias_out(einx.id("m d -> d m", xi_out))
        return self.proj2(self.activation(self.proj1(h)))  # (C_out, M)
