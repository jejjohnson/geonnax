"""Self-tests for the point-cloud invariance harness in `_point_invariance`."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from _point_invariance import (
    assert_point_refinement_converges,
    grid_nodes,
    stratified_cloud,
)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_stratified_cloud_one_point_per_cell(d):
    s = 5
    xi, w = stratified_cloud(s, d, key=jr.PRNGKey(0))
    assert xi.shape == (s**d, d)
    assert w.shape == (s**d,)
    assert jnp.isclose(w.sum(), 1.0, atol=1e-6)
    # Point n lies in cell n of the row-major s^d grid.
    assert jnp.array_equal(jnp.floor(xi * s), jnp.round(grid_nodes((s,) * d) * s))


def test_harness_passes_exact_operator_and_rejects_bad_tolerances():
    # The weighted mean of a constant field is exact at every N.
    ref = jnp.ones(())

    def mean(u, xi, w):
        return jnp.dot(u[0], w)

    def field(xi):
        return jnp.ones((1, xi.shape[0]))

    kwargs = {"d": 2, "key": jr.PRNGKey(0)}
    assert_point_refinement_converges(
        mean, field, ref, (4, 8), max_rel_err=(1e-5, 1e-5), **kwargs
    )
    with pytest.raises(ValueError, match="one tolerance per size"):
        assert_point_refinement_converges(
            mean, field, ref, (4, 8), max_rel_err=(1e-5,), **kwargs
        )
