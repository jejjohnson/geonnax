r"""Point-cloud discretisation-invariance harness (a test helper, not shipped).

A point-set operator approximates a continuous one through quadrature, so its
output must converge as the cloud is refined, whatever the number and
placement of the points. `assert_point_refinement_converges` checks this on
jittered stratified clouds against a dense-grid reference; the operators that
read point clouds (`SpectralConv.from_points`, `GeoFNO`, `ManifoldNO`)
parametrise into it.

Stratified quadrature (one uniform point per cell, $w_n = 1/N$) of a smooth
periodic integrand has error $O(N^{-1/2 - 1/d})$, i.e. $\sim N^{-1}$ in 2-D,
against $O(N^{-1/2})$ for i.i.d. uniform points. An operator that forgets
the weights (the reference Geo-FNO sums $\sum_n u_n e^{-2\pi i k\cdot\xi_n}$
unweighted) instead scales $\propto N$ and fails at once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import einx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


def grid_nodes(grid: Sequence[int]) -> Float[Array, "S d"]:
    """The ``S = prod(grid)`` nodes ``j / s`` of a uniform grid on [0, 1)^d, (S, d)."""
    axes = jnp.meshgrid(*[jnp.arange(s) / s for s in grid], indexing="ij")
    return einx.id("a... d -> (a...) d", jnp.stack(axes, -1))


def stratified_cloud(
    s: int, d: int, *, key: Array
) -> tuple[Float[Array, "N d"], Float[Array, " N"]]:
    """One uniform point per cell of an ``s^d`` grid on [0, 1)^d.

    Returns:
        ``(xi, w)``: the ``N = s^d`` points, shape ``(N, d)`` (cell-major
        order), and the equal quadrature weights ``w = 1/N``, shape ``(N,)``.
    """
    corners = grid_nodes((s,) * d)
    xi = corners + jr.uniform(key, corners.shape) / s
    n = s**d
    return xi, jnp.full((n,), 1.0 / n)


def assert_point_refinement_converges(
    apply: Callable[[Array, Array, Array], Array],
    field: Callable[[Array], Array],
    reference: Array,
    sizes: Sequence[int],
    *,
    d: int,
    key: Array,
    max_rel_err: Sequence[float],
) -> None:
    """Assert an operator on refined stratified clouds approaches its grid output.

    For each ``s`` in ``sizes`` this draws ``xi, w = stratified_cloud(s, d)``,
    evaluates ``out = apply(field(xi), xi, w)`` and requires
    ``||out - reference|| / ||reference|| < max_rel_err[i]``.

    Args:
        apply: ``(u (C, N), xi (N, d), w (N,)) -> output`` on a fixed grid or at
            fixed query points (the same shape as ``reference``).
        field: ``xi (N, d) -> u (C, N)``; smooth, ideally band-limited.
        reference: The operator's output from the dense uniform grid.
        sizes: Cells per axis, increasing; ``N = s**d`` (``(45, 90)`` in 2-D
            gives ``N = 2025, 8100``).
        d: Dimension of the latent domain.
        key: PRNG key; one sub-key per size.
        max_rel_err: One tolerance per size; state where each came from.

    Raises:
        AssertionError: If any size exceeds its tolerance; the message lists
            the relative error at every size.
    """
    if len(sizes) != len(max_rel_err):
        raise ValueError(
            f"need one tolerance per size, got {len(sizes)} sizes and "
            f"{len(max_rel_err)} tolerances."
        )
    errors = []
    for s, k in zip(sizes, jr.split(key, len(sizes)), strict=True):
        xi, w = stratified_cloud(s, d, key=k)
        out = apply(field(xi), xi, w)
        rel = jnp.linalg.norm(out - reference) / jnp.linalg.norm(reference)
        errors.append(float(rel))
    report = ", ".join(
        f"N={s**d}: {e:.3g} (max {t})"
        for s, e, t in zip(sizes, errors, max_rel_err, strict=True)
    )
    assert all(e < t for e, t in zip(errors, max_rel_err, strict=True)), (
        f"point refinement does not converge: {report}"
    )
