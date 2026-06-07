"""Tests for the localized / overcomplete bases: RBF, Gabor frame, time window."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import geonnax
from geonnax._basis import (
    gabor_frame,
    gabor_frame_grid,
    rbf_basis,
    wendland_c2,
    wendland_c4,
)
from geonnax.basis import gaussian_window_features


KEY = jr.PRNGKey(0)


# ---- Wendland kernels -------------------------------------------------------


@pytest.mark.parametrize("kernel", [wendland_c2, wendland_c4])
def test_wendland_peak_and_compact_support(kernel):
    assert float(kernel(jnp.array(0.0))) == pytest.approx(1.0)
    # Exactly zero at and beyond the support radius r = 1.
    assert float(kernel(jnp.array(1.0))) == 0.0
    assert float(kernel(jnp.array(1.5))) == 0.0
    # Non-negative and decreasing on (0, 1).
    r = jnp.linspace(0.0, 1.0, 50)
    vals = kernel(r)
    assert bool((vals >= 0).all())
    assert bool((jnp.diff(vals) <= 1e-6).all())


# ---- RBF basis --------------------------------------------------------------


def test_rbf_gaussian_matches_analytic():
    x = jnp.array([[0.5, 0.0]])
    out = rbf_basis(x, jnp.array([[0.0, 0.0]]), jnp.array([1.0]), kernel="gaussian")
    assert float(out[0, 0]) == pytest.approx(float(jnp.exp(-0.125)), abs=1e-6)


def test_rbf_wendland_is_locally_supported():
    centers = jnp.array([[0.0, 0.0], [5.0, 0.0]])
    out = rbf_basis(
        jnp.array([[0.0, 0.0]]), centers, jnp.array([1.0, 1.0]), kernel="wendland_c2"
    )
    assert float(out[0, 0]) == pytest.approx(1.0)  # on its own centre
    assert float(out[0, 1]) == 0.0  # 5 units away, width 1 -> exactly zero


def test_rbf_gram_is_positive_semidefinite():
    pts = jr.uniform(KEY, (40, 2))
    centers = jr.uniform(jr.PRNGKey(1), (12, 2))
    phi = rbf_basis(pts, centers, jnp.full(12, 0.5), kernel="wendland_c2")
    eigmin = float(jnp.linalg.eigvalsh(phi.T @ phi).min())
    assert eigmin > -1e-6


def test_rbf_grad_finite_at_center():
    # An eval point coincident with a centre gives r = 0; the softened sqrt must
    # keep gradients finite (bare sqrt has an infinite derivative there).
    def loss(centers):
        phi = rbf_basis(
            jnp.zeros((1, 2)), centers, jnp.ones(centers.shape[0]), kernel="wendland_c2"
        )
        return jnp.sum(phi**2)

    grad = jax.grad(loss)(jnp.zeros((3, 2)))
    assert bool(jnp.isfinite(grad).all())


@pytest.mark.parametrize("kernel", ["wendland_c2", "wendland_c4"])
def test_rbf_narrow_atom_peaks_at_center(kernel):
    # A point on its centre must peak at 1 even for a width comparable to the
    # safe-sqrt softening — the value must not be shifted by an additive epsilon.
    width = 1e-6
    out = rbf_basis(
        jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.array([width]), kernel=kernel
    )
    assert float(out[0, 0]) == pytest.approx(1.0)

    def loss(centers):
        return jnp.sum(
            rbf_basis(jnp.zeros((1, 2)), centers, jnp.array([width]), kernel=kernel)
            ** 2
        )

    assert bool(jnp.isfinite(jax.grad(loss)(jnp.zeros((1, 2)))).all())


def test_rbf_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="unknown kernel"):
        rbf_basis(jnp.zeros((2, 2)), jnp.zeros((1, 2)), jnp.ones(1), kernel="bogus")


# ---- Gabor frame ------------------------------------------------------------


def test_gabor_radial_matches_analytic():
    out = gabor_frame(
        jnp.array([[1.0, 0.0]]),
        jnp.array([[0.0, 0.0]]),
        jnp.array([2.0]),
        jnp.array([3.0]),
    )
    want = float(jnp.exp(-0.5 * (1.0 / 2.0) ** 2) * jnp.cos(3.0))
    assert float(out[0, 0]) == pytest.approx(want, abs=1e-6)


def test_gabor_directional_reduces_to_radial_when_aligned():
    x, c, L, k = (
        jnp.array([[1.0, 0.0]]),
        jnp.array([[0.0, 0.0]]),
        jnp.array([2.0]),
        jnp.array([3.0]),
    )
    radial = gabor_frame(x, c, L, k)
    directional = gabor_frame(x, c, L, k, orientations=jnp.array([[1.0, 0.0]]))
    assert float(directional[0, 0]) == pytest.approx(float(radial[0, 0]), abs=1e-6)


def test_gabor_grad_finite_at_center():
    def loss(centers):
        phi = gabor_frame(
            jnp.zeros((1, 2)),
            centers,
            jnp.ones(centers.shape[0]),
            jnp.ones(centers.shape[0]),
        )
        return jnp.sum(phi**2)

    assert bool(jnp.isfinite(jax.grad(loss)(jnp.zeros((3, 2)))).all())


# ---- Gabor frame grid -------------------------------------------------------


def test_gabor_frame_grid_is_overcomplete_with_geometry():
    grid = jnp.stack(
        jnp.meshgrid(jnp.linspace(0, 1, 12), jnp.linspace(0, 1, 12), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    phi, centers, scales, wavenumbers = gabor_frame_grid(
        grid, bounds, n_scales=3, base_scale=0.15, oversample=2.0
    )
    n, m = phi.shape
    assert m == centers.shape[0] == scales.shape[0] == wavenumbers.shape[0]
    assert m > n  # overcomplete frame
    # Dyadic scales and matching wavenumbers k = 2 pi / L.
    assert len(np.unique(np.asarray(scales))) == 3
    assert jnp.allclose(wavenumbers, 2.0 * jnp.pi / scales)


def test_gabor_frame_grid_centers_stay_within_bounds():
    # base_scale 0.6 with oversample 1 gives spacing 0.6, which does not divide
    # the unit box; no generated centre may fall outside [lo, hi].
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    _, centers, _, _ = gabor_frame_grid(
        jnp.zeros((4, 2)), bounds, n_scales=2, base_scale=0.6, oversample=1.0
    )
    c = np.asarray(centers)
    assert bool((c[:, 0] >= 0.0).all() and (c[:, 0] <= 1.0).all())
    assert bool((c[:, 1] >= 0.0).all() and (c[:, 1] <= 1.0).all())


@pytest.mark.slow
def test_gabor_frame_reconstructs_smooth_field():
    grid = jnp.stack(
        jnp.meshgrid(jnp.linspace(0, 1, 16), jnp.linspace(0, 1, 16), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    phi, *_ = gabor_frame_grid(
        grid, bounds, n_scales=3, base_scale=0.15, oversample=2.0
    )
    field = jnp.sin(2 * jnp.pi * grid[:, 0]) * jnp.cos(2 * jnp.pi * grid[:, 1])
    w, *_ = jnp.linalg.lstsq(phi, field, rcond=1e-6)
    rel = float(jnp.linalg.norm(phi @ w - field) / jnp.linalg.norm(field))
    assert rel < 0.05


def test_gabor_frame_grid_rejects_bad_args():
    grid, bounds = jnp.zeros((4, 2)), jnp.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="n_scales must be"):
        gabor_frame_grid(grid, bounds, n_scales=0, base_scale=0.1)
    with pytest.raises(ValueError, match="base_scale must be"):
        gabor_frame_grid(grid, bounds, n_scales=2, base_scale=0.0)
    with pytest.raises(ValueError, match="lo < hi"):
        gabor_frame_grid(
            grid, jnp.array([[1.0, 0.0], [0.0, 1.0]]), n_scales=2, base_scale=0.1
        )


# ---- Gaussian window --------------------------------------------------------


def test_gaussian_window_matches_analytic():
    t = jnp.linspace(0.0, 10.0, 11)
    out = gaussian_window_features(t, jnp.array([5.0]), jnp.array([2.0]))
    assert float(out[5, 0]) == pytest.approx(1.0)  # at the centre
    want = jnp.exp(-0.5 * ((t - 5.0) / 2.0) ** 2)
    assert jnp.allclose(out[:, 0], want, atol=1e-6)


def test_gaussian_window_jit_vmap():
    f = jax.jit(gaussian_window_features)
    t, c, w = jnp.linspace(0, 10, 8), jnp.array([2.0, 8.0]), jnp.array([1.0, 1.0])
    assert f(t, c, w).shape == (8, 2)
    batched = jax.vmap(lambda tt: gaussian_window_features(tt, c, w))(jnp.ones((3, 8)))
    assert batched.shape == (3, 8, 2)


# ---- exports ----------------------------------------------------------------


def test_exports():
    # All six are reachable from ``geonnax.basis`` (convenience re-exports).
    for name in [
        "rbf_basis",
        "wendland_c2",
        "wendland_c4",
        "gabor_frame",
        "gabor_frame_grid",
        "gaussian_window_features",
    ]:
        assert hasattr(geonnax.basis, name), name
    # The temporal window is native to basis.py; the geometric bases live in
    # (and are documented under) geonnax._basis.
    assert "gaussian_window_features" in geonnax.basis.__all__
    for name in [
        "rbf_basis",
        "wendland_c2",
        "wendland_c4",
        "gabor_frame",
        "gabor_frame_grid",
    ]:
        assert name in geonnax._basis.__all__
