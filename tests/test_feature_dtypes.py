"""Dtype behaviour of the feature transforms in geonnax.basis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import fourier_features, seasonal_features


def _fourier(x):
    return fourier_features(x, 4, rescale=True)


def _seasonal(x):
    return seasonal_features(x, (7.0, 365.25), (2, 3), rescale=True)


@pytest.mark.parametrize("fn", [_fourier, _seasonal], ids=["fourier", "seasonal"])
def test_float_dtypes_follow_the_input_under_x64(fn):
    with jax.enable_x64(True):
        x = jnp.linspace(0.0, 3.0, 5)
        assert fn(x.astype(jnp.float32)).dtype == jnp.float32
        assert fn(x.astype(jnp.float64)).dtype == jnp.float64


@pytest.mark.parametrize("fn", [_fourier, _seasonal], ids=["fourier", "seasonal"])
def test_integer_inputs_are_promoted(fn):
    x = jnp.arange(5)
    out = fn(x)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(out, fn(x.astype(jnp.float32)), rtol=1e-6)


def test_seasonal_phase_is_float64_for_float64_time():
    # Seconds since the epoch with a daily period: a float32-rounded 1/86400
    # puts the phase off by a visible fraction of a cycle at t ≈ 1.7e9 s.
    with jax.enable_x64(True):
        t = jnp.array([1.7e9, 1.7e9 + 3600.0], dtype=jnp.float64)
        out = seasonal_features(t, (86_400.0,), (1,))
        phase = 2 * np.pi * np.asarray([1.7e9, 1.7e9 + 3600.0]) / 86_400.0
        expected = np.concatenate([np.cos(phase)[:, None], np.sin(phase)[:, None]], 1)
        # float64 phase of ~1.2e5 rad: round-off ~1e-11 rad, so 1e-9 is ample.
        np.testing.assert_allclose(np.asarray(out), expected, atol=1e-9)
