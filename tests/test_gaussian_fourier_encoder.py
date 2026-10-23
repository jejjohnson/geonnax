"""Tests for GaussianFourierEncoder (Tancik et al., 2020)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from geonnax import GaussianFourierEncoder


KEY = jr.PRNGKey(0)


def test_shape_and_layout():
    enc = GaussianFourierEncoder.init(3, 16, sigma=2.0, key=KEY)
    x = jnp.array([0.1, -0.4, 0.7])
    out = enc(x)
    assert out.shape == (32,) == (enc.num_features,)
    z = 2 * jnp.pi * enc.B @ x
    # float32: the phase 2π·Bx is formed in a different order here, and an
    # argument of a few radians carries ~1e-6 absolute round-off.
    np.testing.assert_allclose(
        out, jnp.concatenate([jnp.cos(z), jnp.sin(z)]), atol=1e-5
    )
    assert jax.vmap(enc)(jnp.zeros((7, 3))).shape == (7, 32)


def test_same_key_same_frequencies_and_sigma_scaling():
    a = GaussianFourierEncoder.init(2, 8, sigma=1.0, key=KEY)
    b = GaussianFourierEncoder.init(2, 8, sigma=5.0, key=KEY)
    # Both sides are one float32 rounding of the same product.
    np.testing.assert_allclose(b.B, 5.0 * a.B, rtol=1e-6)


def test_frequency_matrix_is_frozen():
    enc = GaussianFourierEncoder.init(2, 8, sigma=3.0, key=KEY)
    g = eqx.filter_grad(lambda e: jnp.sum(e(jnp.array([0.2, 0.3]))))(enc)
    np.testing.assert_array_equal(g.B, 0.0)


@pytest.mark.parametrize("sigma", [1.0, 10.0])
def test_feature_means_match_closed_form(sigma):
    # For x ~ U[0, 1]^D, E[exp(2πi b·x)] = ∏_d (exp(iω_d) − 1) / (iω_d), ω_d = 2π b_d;
    # its real and imaginary parts are the means of the cos and sin features.
    # Each feature has sd ≤ 1, so with 20,000 draws the standard error is
    # ≤ 0.0071; 0.05 is 7 standard errors, enough for the maximum over the
    # 64 features tested jointly. σ = 1 gives means up to ~0.9 (a sharp check);
    # σ = 10 checks the high-frequency regime.
    enc = GaussianFourierEncoder.init(2, 32, sigma=sigma, key=KEY)
    x = jr.uniform(jr.PRNGKey(1), (20_000, 2))
    f = jax.vmap(enc)(x)
    omega = 2 * jnp.pi * enc.B.astype(jnp.complex64)
    char = jnp.prod((jnp.exp(1j * omega) - 1) / (1j * omega), axis=-1)
    expected = jnp.concatenate([char.real, char.imag])
    np.testing.assert_allclose(f.mean(0), expected, atol=0.05)


def test_dtype_follows_input():
    enc = GaussianFourierEncoder.init(2, 4, sigma=1.0, key=KEY)
    assert enc(jnp.zeros(2, dtype=jnp.float32)).dtype == jnp.float32
    assert enc(jnp.zeros(2, dtype=jnp.int32)).dtype == jnp.float32


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0, 4, 1.0), "in_features"),
        ((2, 0, 1.0), "num_frequencies"),
        ((2, 4, 0.0), "sigma"),
        ((2, 4, float("inf")), "sigma"),
    ],
)
def test_init_validation(args, match):
    with pytest.raises(ValueError, match=match):
        GaussianFourierEncoder.init(*args, key=KEY)


def test_call_validation():
    enc = GaussianFourierEncoder.init(2, 4, sigma=1.0, key=KEY)
    with pytest.raises(ValueError, match=r"jax\.vmap"):
        enc(jnp.zeros((3, 2)))


def test_half_precision_keeps_float32_phase():
    enc = GaussianFourierEncoder.init(2, 32, sigma=10.0, key=KEY)
    x = jnp.array([0.25, 0.75])  # exactly representable in float16
    out16 = enc(x.astype(jnp.float16))
    assert out16.dtype == jnp.float16
    # Only the final cast to float16 rounds: ≤ 2^-11 relative on values in [-1, 1].
    np.testing.assert_allclose(out16.astype(jnp.float32), enc(x), atol=1e-3)


@pytest.mark.integration
def test_jit_and_input_gradient():
    enc = GaussianFourierEncoder.init(3, 8, sigma=2.0, key=KEY)
    xs = jr.normal(jr.PRNGKey(2), (5, 3))
    np.testing.assert_allclose(
        eqx.filter_jit(jax.vmap(enc))(xs), jax.vmap(enc)(xs), atol=1e-6
    )
    x = xs[0]
    z = 2 * jnp.pi * enc.B @ x
    analytic = jnp.concatenate(
        [
            -jnp.sin(z)[:, None] * 2 * jnp.pi * enc.B,
            jnp.cos(z)[:, None] * 2 * jnp.pi * enc.B,
        ]
    )
    # float32: entries are O(2πσ) ~ 10, so compare at 1e-4 relative.
    np.testing.assert_allclose(jax.jacfwd(enc)(x), analytic, rtol=1e-4, atol=1e-4)
