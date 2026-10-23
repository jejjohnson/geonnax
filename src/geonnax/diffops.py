r"""Coordinate-derivative operators for continuous fields, flat and spherical.

A neural field is a function of its coordinates, so its spatial derivatives
come from automatic differentiation rather than finite differences. Each
operator here takes a **single-example** field and returns a new callable
that evaluates the derivative at one point; batch with `jax.vmap` and
compile with `jax.jit` as usual.

Euclidean fields of $x \in \mathbb{R}^D$:

- `grad`: $\nabla f$, shape ``(D,)``;
- `jacobian`: $\partial u_k / \partial x_d$, shape ``(K, D)``;
- `divergence`: $\nabla \cdot u$, a scalar;
- `curl2d`: $\partial_x u_y - \partial_y u_x$, a scalar;
- `laplacian`: $\Delta f$, a scalar.

Fields of a unit vector $n \in S^2$ (for example a SIREN behind
`Cartesian3DEncoder`, or `SphericalHarmonicEncoder` with
``input_mode="cartesian"``):

- `surface_grad`: $\nabla_S f$, a tangent vector in Cartesian components;
- `surface_grad_enu`: its east and north components at a lon/lat;
- `laplace_beltrami`: $\Delta_S f$, a scalar.

The spherical operators act on the **unit** sphere. For a sphere of radius
$R$, divide `surface_grad` by $R$ and `laplace_beltrami` by $R^2$.

These are operators, not losses: residual and physics costs built from them
belong to the caller (vardax, pipekit).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from geonnax._basis._rbf import _safe_sqrt
from geonnax.geo import _promote_to_floating, enu_basis, lonlat_to_cartesian3d


ScalarField = Callable[[Float[Array, " D"]], Float[Array, ""]]
VectorField = Callable[[Float[Array, " D"]], Float[Array, " K"]]


def _check_point(x: Float[Array, ...], name: str = "x") -> Float[Array, " D"]:
    """Validate one point and promote integer coordinates to floating."""
    if x.ndim != 1:
        raise ValueError(
            f"{name} must be a single point of shape (D,); got shape {x.shape}. "
            "Batch with jax.vmap."
        )
    return _promote_to_floating(x)


def grad(f: ScalarField) -> Callable[[Float[Array, " D"]], Float[Array, " D"]]:
    r"""Gradient $\nabla f$ of a scalar field.

    Args:
        f: Scalar field of one point, ``(D,) -> ()``.

    Returns:
        A callable ``(D,) -> (D,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> g = diffops.grad(lambda x: jnp.sum(x**2))
        >>> g(jnp.array([1.0, 2.0])).tolist()
        [2.0, 4.0]
    """

    def _grad(x: Float[Array, " D"]) -> Float[Array, " D"]:
        x = _check_point(x)
        return jax.grad(f)(x)

    return _grad


def jacobian(u: VectorField) -> Callable[[Float[Array, " D"]], Float[Array, "K D"]]:
    r"""Jacobian $J_{kd} = \partial u_k / \partial x_d$ of a vector field.

    Computed in forward mode (`jax.jacfwd`), which is cheapest when the
    number of coordinates $D$ is small, as it is for fields of space and time.

    Args:
        u: Vector field of one point, ``(D,) -> (K,)``.

    Returns:
        A callable ``(D,) -> (K, D)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> J = diffops.jacobian(lambda x: jnp.array([x[0] * x[1], x[1]]))
        >>> J(jnp.array([2.0, 3.0])).tolist()
        [[3.0, 2.0], [0.0, 1.0]]
    """

    def _jac(x: Float[Array, " D"]) -> Float[Array, "K D"]:
        x = _check_point(x)
        return jax.jacfwd(u)(x)

    return _jac


def divergence(u: VectorField) -> Callable[[Float[Array, " D"]], Float[Array, ""]]:
    r"""Divergence $\nabla\cdot u = \sum_d \partial u_d / \partial x_d$.

    Args:
        u: Vector field of one point, ``(D,) -> (D,)``.

    Returns:
        A callable ``(D,) -> ()``.

    Raises:
        ValueError: If ``u`` does not map ``(D,)`` to ``(D,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> div = diffops.divergence(lambda x: x * jnp.array([1.0, 2.0, 3.0]))
        >>> float(div(jnp.zeros(3)))
        6.0
    """

    def _div(x: Float[Array, " D"]) -> Float[Array, ""]:
        x = _check_point(x)
        J = jax.jacfwd(u)(x)
        if J.shape != (x.shape[0], x.shape[0]):
            raise ValueError(
                f"divergence needs u: (D,) -> (D,); got a Jacobian of shape {J.shape}."
            )
        return jnp.trace(J)

    return _div


def curl2d(u: VectorField) -> Callable[[Float[Array, " 2"]], Float[Array, ""]]:
    r"""Scalar curl (vorticity) of a planar field, $\partial_x u_y - \partial_y u_x$.

    Args:
        u: Planar vector field of one point, ``(2,) -> (2,)``, with inputs
            ordered ``(x, y)``.

    Returns:
        A callable ``(2,) -> ()``.

    Raises:
        ValueError: If ``u`` does not map ``(2,)`` to ``(2,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> rot = diffops.curl2d(lambda x: jnp.array([-x[1], x[0]]))  # solid body
        >>> float(rot(jnp.array([0.3, -0.2])))
        2.0
    """

    def _curl(x: Float[Array, " 2"]) -> Float[Array, ""]:
        x = _check_point(x)
        J = jax.jacfwd(u)(x)
        if J.shape != (2, 2):
            raise ValueError(
                f"curl2d needs u: (2,) -> (2,); got a Jacobian of shape {J.shape}."
            )
        return J[1, 0] - J[0, 1]

    return _curl


def laplacian(f: ScalarField) -> Callable[[Float[Array, " D"]], Float[Array, ""]]:
    r"""Laplacian $\Delta f = \sum_d \partial^2 f / \partial x_d^2$ of a scalar field.

    Evaluated as $\sum_d e_d^\top \nabla^2 f\, e_d$ with forward-over-reverse
    Hessian-vector products (`jax.jvp` of `jax.grad`) along the $D$ unit
    vectors, so the dense Hessian is never formed.

    Args:
        f: Scalar field of one point, ``(D,) -> ()``.

    Returns:
        A callable ``(D,) -> ()``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> lap = diffops.laplacian(lambda x: jnp.sum(x**2))
        >>> float(lap(jnp.array([0.5, -1.0, 2.0])))
        6.0
    """

    def _lap(x: Float[Array, " D"]) -> Float[Array, ""]:
        x = _check_point(x)
        g = jax.grad(f)

        def second(e: Float[Array, " D"]) -> Float[Array, ""]:
            return jnp.vdot(jax.jvp(g, (x,), (e,))[1], e)

        eye = jnp.eye(x.shape[0], dtype=x.dtype)
        return jnp.sum(jax.vmap(second)(eye))

    return _lap


def _unit(x: Float[Array, " 3"]) -> Float[Array, " 3"]:
    # x / |x|, guarded by the shared `_safe_sqrt`: the origin has no direction, so
    # it maps to
    # itself with finite gradients (masked padding rows stay NaN-free).
    sq = jnp.sum(x**2)
    return x / jnp.where(sq > 0.0, _safe_sqrt(sq), 1.0)


def _radial_extension(f: ScalarField) -> ScalarField:
    # g(x) = f(x / |x|) is constant along rays, so its Euclidean derivatives at
    # |x| = 1 are the surface derivatives of f.
    def g(x: Float[Array, " 3"]) -> Float[Array, ""]:
        return f(_unit(x))

    return g


def _check_sphere_point(n: Float[Array, ...]) -> Float[Array, " 3"]:
    """Validate a sphere point and project it onto the unit sphere."""
    n = _check_point(n, "n")
    if n.shape[0] != 3:
        raise ValueError(f"n must be a 3-D unit vector; got shape {n.shape}.")
    return _unit(n)


def surface_grad(f: ScalarField) -> Callable[[Float[Array, " 3"]], Float[Array, " 3"]]:
    r"""Surface gradient $\nabla_S f$ of a field on the unit sphere.

    With $g(x) = f(x / \lVert x \rVert)$, the radially constant extension of
    $f$, and $n$ the unit normal,

    $$
    \nabla_S f(n) = (I - n n^\top)\, \nabla g(n).
    $$

    Inputs are first projected to $n / \lVert n \rVert$, so any non-zero
    vector gives the surface gradient at its direction; the final projection
    removes round-off. The origin has no direction and returns a finite but
    meaningless value (with finite gradients, so masked padding is safe).

    Args:
        f: Field of a unit vector, ``(3,) -> ()``.

    Returns:
        A callable ``(3,) -> (3,)`` returning a tangent vector in
        Cartesian components, at the direction of the input.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> sg = diffops.surface_grad(lambda n: n[2])  # f = sin(latitude)
        >>> sg(jnp.array([1.0, 0.0, 0.0])).tolist()  # at the equator: due north
        [0.0, 0.0, 1.0]
    """
    g = _radial_extension(f)

    def _sgrad(n: Float[Array, " 3"]) -> Float[Array, " 3"]:
        n = _check_sphere_point(n)
        dg = jax.grad(g)(n)
        return dg - n * jnp.vdot(n, dg)

    return _sgrad


def surface_grad_enu(
    f: ScalarField,
    *,
    input_unit: Literal["degrees", "radians"] = "degrees",
) -> Callable[[Float[Array, " 2"]], Float[Array, " 2"]]:
    r"""East and north components of the surface gradient, at a lon/lat.

    Projects `surface_grad` onto the local east and north unit vectors of
    `geonnax.geo.enu_basis`:
    $(\hat e^\top \nabla_S f,\ \hat n^\top \nabla_S f)$. On a sphere of radius
    $R$ these are $R\,(\partial f/\partial x_{\text{east}},\ \partial
    f/\partial y_{\text{north}})$, the derivatives that give geostrophic
    velocity from sea-surface height.

    Args:
        f: Field of a unit vector, ``(3,) -> ()``.
        input_unit: Unit of the lon/lat passed to the returned callable.

    Returns:
        A callable ``(2,) -> (2,)`` taking ``(lon, lat)`` and returning
        ``(east, north)`` components on the unit sphere.

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> d = diffops.surface_grad_enu(lambda n: n[2])  # f = sin(latitude)
        >>> d(jnp.array([30.0, 0.0])).round(6).tolist()
        [0.0, 1.0]
    """
    sgrad = surface_grad(f)

    def _enu(lonlat: Float[Array, " 2"]) -> Float[Array, " 2"]:
        _check_point(lonlat, "lonlat")
        if lonlat.shape[0] != 2:
            raise ValueError(f"lonlat must have shape (2,); got {lonlat.shape}.")
        pts = einx.id("c -> 1 c", lonlat)
        n = lonlat_to_cartesian3d(pts, input_unit=input_unit)[0]
        frame = enu_basis(pts, input_unit=input_unit)[0]  # rows: east, north, up
        return einx.dot("r c, c -> r", frame[:2], sgrad(n))

    return _enu


def laplace_beltrami(
    f: ScalarField,
) -> Callable[[Float[Array, " 3"]], Float[Array, ""]]:
    r"""Laplace–Beltrami operator $\Delta_S f$ of a field on the unit sphere.

    In spherical coordinates $\Delta = \partial_r^2 + \tfrac{2}{r}\partial_r +
    r^{-2}\Delta_S$, so for the radially constant extension
    $g(x) = f(x/\lVert x\rVert)$ the radial terms vanish and

    $$
    \Delta_S f(n) = \Delta g(n), \qquad \lVert n \rVert = 1 .
    $$

    Spherical harmonics are its eigenfunctions:
    $\Delta_S Y_l^m = -l(l+1)\, Y_l^m$.

    Args:
        f: Field of a unit vector, ``(3,) -> ()``.

    Returns:
        A callable ``(3,) -> ()``, evaluated at the direction
        $n/\lVert n\rVert$ of the input (the origin returns a finite but
        meaningless value).

    Examples:
        >>> import jax.numpy as jnp
        >>> from geonnax import diffops
        >>> lb = diffops.laplace_beltrami(lambda n: n[2])  # Y_1^0 up to a constant
        >>> n = jnp.array([0.6, 0.0, 0.8])
        >>> round(float(lb(n)) / float(n[2]), 4)  # eigenvalue -l(l+1), l = 1
        -2.0
    """
    lap = laplacian(_radial_extension(f))

    def _lb(n: Float[Array, " 3"]) -> Float[Array, ""]:
        n = _check_sphere_point(n)
        return lap(n)

    return _lb


__all__ = [
    "curl2d",
    "divergence",
    "grad",
    "jacobian",
    "laplace_beltrami",
    "laplacian",
    "surface_grad",
    "surface_grad_enu",
]
