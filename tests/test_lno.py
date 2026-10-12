"""Tests for `geonnax.lno`: the LNO time-marching harness and its receptive field."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from geonnax.cno import CNO
from geonnax.lno import LNO, LocalNeuralOperator, cno_receptive_radius
from geonnax.unet import UNet


KEY = jr.PRNGKey(0)
ND = [1, 2, pytest.param(3, marks=pytest.mark.slow)]


def _small(nd: int, **kw) -> LNO:
    kw = {"hidden_channels": 4, "depth": 1, "n_res_blocks": 1} | kw
    return LNO.init(2, nd, key=KEY, **kw)


def _smooth_1d(n: int) -> jax.Array:
    x = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    return (jnp.sin(x) + 0.3 * jnp.cos(3 * x))[None]


# ---- shapes -------------------------------------------------------------------


@pytest.mark.parametrize("nd", ND)
@pytest.mark.parametrize("n", [12, 19])
def test_lno_step_and_rollout_shapes(nd, n):
    lno = _small(nd)
    u0 = jr.normal(jr.PRNGKey(1), (2, *(n,) * nd))
    assert lno.step(u0).shape == u0.shape
    assert lno(u0).shape == u0.shape
    assert lno.rollout(u0, 4).shape == (4, *u0.shape)


def test_lno_alias():
    assert LocalNeuralOperator is LNO


def test_lno_defaults_halo_to_receptive_radius():
    lno = _small(1, depth=2, kernel_size=5)
    r = cno_receptive_radius(depth=2, n_res_blocks=1, kernel_size=5)
    assert lno.receptive_radius == lno.halo == r
    assert lno.multiple == 4
    assert isinstance(lno.core, CNO) and lno.core.resolution is None


# ---- one step = crop(core(pad(u))) ---------------------------------------------


@pytest.mark.parametrize("boundary", ["periodic", "reflect", "zero"])
def test_lno_step_is_core_on_padded_then_cropped_field(boundary):
    lno = _small(1, boundary=boundary)
    u = jr.normal(jr.PRNGKey(1), (2, 30))
    padded = lno.pad(u)
    # halo 11 per side, then rounded up to a multiple of 2**depth = 2.
    assert padded.shape == (2, 52)
    ref = lno.core(padded)[:, lno.halo : lno.halo + 30]
    assert jnp.array_equal(lno.step(u), ref)


@pytest.mark.parametrize("n", [30, 31])
def test_lno_padded_extent_is_a_multiple_of_two_to_the_depth(n):
    lno = _small(2, depth=2)
    padded = lno.pad(jnp.ones((2, n, n + 3)))
    assert all(m % 4 == 0 for m in padded.shape[1:])
    assert all(
        m >= k + 2 * lno.halo for m, k in zip(padded.shape[1:], (n, n + 3), strict=True)
    )


# ---- boundary modes -------------------------------------------------------------


def test_lno_periodic_pad_wraps():
    lno = LNO.from_core(lambda v: v, 1, halo=3, boundary="periodic")
    u = jnp.arange(8.0)[None]
    expected = jnp.array([[5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]], dtype=u.dtype)
    assert jnp.array_equal(lno.pad(u), expected)


def test_lno_reflect_pad_mirrors_without_repeating_the_edge():
    lno = LNO.from_core(lambda v: v, 1, halo=2, boundary="reflect", multiple=4)
    u = jnp.arange(6.0)[None]  # 6 + 4 = 10 -> 12: 2 extra on the high side
    expected = jnp.array([[2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1]], dtype=u.dtype)
    assert jnp.array_equal(lno.pad(u), expected)


def test_lno_zero_pad_is_constant_zero():
    lno = LNO.from_core(lambda v: v, 2, halo=2, boundary="zero")
    u = jnp.ones((1, 3, 4))
    padded = lno.pad(u)
    assert padded.shape == (1, 7, 8)
    assert float(padded.sum()) == 12.0
    assert jnp.array_equal(lno.unpad(padded, (3, 4)), u)


def test_lno_identity_core_rollout_is_constant():
    lno = LNO.from_core(lambda v: v, 1, halo=3, boundary="reflect")
    u = jr.normal(jr.PRNGKey(1), (2, 10))
    assert jnp.array_equal(lno.rollout(u, 3), jnp.stack([u] * 3))


def test_lno_periodic_step_is_shift_equivariant():
    # The CNO core zero-pads its stencils and resamples with global Fourier
    # filters over the *padded* grid, so equivariance on the torus is
    # approximate: it holds up to the receptive-field tails leaking across the
    # padded seam. On this smooth field the error measured 3.4e-3 with
    # halo = r and 6.3e-2 with no halo, so bound it at 1e-2 and require the
    # halo to buy at least 5x.
    u = _smooth_1d(64)

    def shift_error(lno: LNO, s: int = 5) -> float:
        a = lno.step(jnp.roll(u, s, axis=-1))
        b = jnp.roll(lno.step(u), s, axis=-1)
        return float(jnp.abs(a - b).max() / jnp.abs(b).max())

    kw = {"hidden_channels": 8, "depth": 1, "n_res_blocks": 1}
    with_halo = shift_error(LNO.init(1, 1, key=KEY, **kw))
    no_halo = shift_error(LNO.init(1, 1, key=KEY, halo=0, **kw))
    assert with_halo < 1e-2
    assert no_halo > 5 * with_halo


def test_lno_runs_the_same_core_on_two_domain_sizes():
    lno = _small(2)
    for n in (16, 27):
        traj = lno.rollout(jr.normal(jr.PRNGKey(n), (2, n, n)), 4)
        assert traj.shape == (4, 2, n, n)
        assert bool(jnp.all(jnp.isfinite(traj)))


def test_lno_from_core_wraps_a_shallow_unet():
    unet = UNet.init(
        8, key=KEY, channels=1, num_spatial_dims=1, dim_mults=(1, 2), groups=4
    )
    lno = LNO.from_core(unet, 1, halo=4, multiple=4, boundary="zero")
    u = jr.normal(jr.PRNGKey(1), (1, 21))
    assert lno.rollout(u, 2).shape == (2, 1, 21)
    assert lno.receptive_radius is None


# ---- receptive field --------------------------------------------------------------


@pytest.mark.parametrize(
    ("depth", "n_res", "k", "r"),
    [(1, 0, 3, 7), (1, 1, 3, 11), (2, 1, 3, 25), (1, 1, 5, 22), (3, 2, 3, 69)],
)
def test_cno_receptive_radius_values(depth, n_res, k, r):
    assert cno_receptive_radius(depth, n_res, k) == r


def _energy_within(depth: int, n_res: int, k: int, seed: int) -> float:
    """Fraction of a centre output's squared Jacobian row within the radius."""
    n = 256
    op = CNO.init(
        1,
        1,
        1,
        key=jr.PRNGKey(seed),
        hidden_channels=8,
        depth=depth,
        n_res_blocks=n_res,
        kernel_size=k,
    )
    x = jr.normal(jr.PRNGKey(100 + seed), (1, n))
    row = jax.grad(lambda v: op(v)[0, n // 2])(x)[0]
    energy = np.asarray(row) ** 2
    dist = np.abs(np.arange(n) - n // 2)
    r = cno_receptive_radius(depth, n_res, k)
    return float(energy[dist <= r].sum() / energy.sum())


# The Fourier resampling gives the impulse response global 1/distance tails, so
# the radius is effective, not a hard bound. Over 3 seeds x 6 configurations
# the squared-sensitivity fraction within r measured 0.954-0.999 (the
# docstring's numbers); require 0.90 so a platform's float32 differences
# cannot flip a draw sitting near the low end.
_RF_CONFIGS = [
    (1, 1, 3),
    pytest.param(1, 0, 3, marks=pytest.mark.slow),
    pytest.param(2, 1, 3, marks=pytest.mark.slow),
    pytest.param(1, 1, 5, marks=pytest.mark.slow),
    pytest.param(2, 2, 3, marks=pytest.mark.slow),
]


@pytest.mark.parametrize(("depth", "n_res", "k"), _RF_CONFIGS)
@pytest.mark.parametrize("seed", [0, 1])
def test_cno_receptive_radius_matches_impulse_response(depth, n_res, k, seed):
    assert _energy_within(depth, n_res, k, seed) >= 0.90


# ---- validation -------------------------------------------------------------------


def test_lno_raises_when_receptive_field_exceeds_range():
    with pytest.raises(ValueError, match="exceeds receptive_range=10"):
        LNO.init(1, 1, key=KEY, depth=2, receptive_range=10)
    lno = LNO.init(1, 1, key=KEY, hidden_channels=4, receptive_range=11)
    assert lno.receptive_radius == 11


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"boundary": "neumann"}, "boundary"),
        ({"receptive_range": -1}, "receptive_range"),
        ({"halo": -1}, "halo"),
        ({"depth": 0}, "depth"),
        ({"n_res_blocks": -1}, "n_res_blocks"),
        ({"kernel_size": 4}, "kernel_size"),
        ({"hidden_channels": 0}, "hidden_channels"),
    ],
)
def test_lno_init_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        LNO.init(1, 1, key=KEY, **kw)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"num_spatial_dims": 4}, "num_spatial_dims"),
        ({"halo": -2}, "halo"),
        ({"multiple": 0}, "multiple"),
        ({"boundary": "wrap"}, "boundary"),
    ],
)
def test_lno_from_core_validation(kw, match):
    kw = {"num_spatial_dims": 1, "halo": 1} | kw
    with pytest.raises(ValueError, match=match):
        LNO.from_core(lambda v: v, **kw)


def test_lno_call_time_validation():
    lno = _small(1)
    with pytest.raises(ValueError, match="spatial axes"):
        lno.step(jnp.ones((2, 8, 8)))
    with pytest.raises(ValueError, match="n_steps"):
        lno.rollout(jnp.ones((2, 8)), 0)


# ---- transforms -------------------------------------------------------------------


@pytest.mark.integration
def test_lno_vmap_jit_grad_through_rollout():
    lno = _small(2)
    u0 = jr.normal(jr.PRNGKey(1), (3, 2, 16, 16))
    traj = eqx.filter_jit(jax.vmap(lambda u: lno.rollout(u, 3)))(u0)
    assert traj.shape == (3, 3, 2, 16, 16)
    assert jnp.allclose(traj[0], lno.rollout(u0[0], 3), atol=1e-5)

    @eqx.filter_grad
    def grad_fn(model: LNO) -> jax.Array:
        return jnp.mean(jax.vmap(lambda u: model.rollout(u, 3))(u0) ** 2)

    grads = grad_fn(lno)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(float(jnp.abs(g).max()) > 0 for g in leaves)
