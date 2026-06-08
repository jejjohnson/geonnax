"""Tests for the data-driven, vector, geodesic, and wavelet bases.

Covers the four bases added on top of the closed-form eigenbasis zoo:
``eof_basis`` (PCA), ``divfree_basis`` (incompressible vector atoms),
``spherical_rbf_basis`` (geodesic radial basis), and the orthonormal DWT
``wavelet_basis_1d`` / ``wavelet_basis_2d``. Each is reached through the public
``geonnax.basis`` surface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geonnax.basis import (
    divfree_basis,
    eof_basis,
    spherical_rbf_basis,
    wavelet_basis_1d,
    wavelet_basis_2d,
)


class TestEOFBasis:
    def test_shapes_and_orthonormal_columns(self):
        X = jax.random.normal(jax.random.PRNGKey(0), (20, 8))
        phi, s = eof_basis(X, n_modes=5)
        assert phi.shape == (8, 5)
        assert s.shape == (5,)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(5), atol=1e-5)

    def test_singular_values_descending(self):
        X = jax.random.normal(jax.random.PRNGKey(1), (30, 10))
        _, s = eof_basis(X, n_modes=10)
        assert bool(jnp.all(jnp.diff(s) <= 1e-6))

    def test_reconstructs_full_rank_anomalies(self):
        # Keeping all min(T, N) modes reconstructs the centred data exactly.
        X = jax.random.normal(jax.random.PRNGKey(2), (12, 6))
        phi, _ = eof_basis(X, n_modes=6)
        xc = X - X.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(xc @ phi @ phi.T, xc, atol=1e-4)

    def test_recovers_known_low_rank_pattern(self):
        # Data built from a single spatial pattern -> first EOF spans it.
        pattern = jnp.array([1.0, -1.0, 2.0, 0.5])
        pattern = pattern / jnp.linalg.norm(pattern)
        amps = jax.random.normal(jax.random.PRNGKey(3), (40,))
        X = amps[:, None] * pattern[None, :]
        phi, _ = eof_basis(X, n_modes=1, center=False)
        # the leading EOF is the pattern up to sign
        cos = jnp.abs(phi[:, 0] @ pattern)
        np.testing.assert_allclose(cos, 1.0, atol=1e-5)

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            eof_basis(jnp.ones((4, 3, 2)), n_modes=1)
        with pytest.raises(ValueError):
            eof_basis(jnp.ones((4, 3)), n_modes=5)  # > min(T, N)


class TestDivFreeBasis:
    def test_shapes(self):
        xy = jnp.zeros((7, 2))
        phi, lam = divfree_basis(xy, num_basis_per_dim=3, L=1.0)
        assert phi.shape == (7, 9, 2)  # M = 3 * 3, last axis = (u, v)
        assert lam.shape == (9,)

    def test_atoms_are_divergence_free(self):
        # d/dx u + d/dy v == 0 analytically for every atom, at arbitrary points.
        def atom_field(p, j):
            phi, _ = divfree_basis(p[None, :], 4, 1.5)
            return phi[0, j]  # (2,) velocity

        pts = jnp.array([[0.13, -0.21], [0.7, 0.4], [-0.9, 0.05]])
        for p in pts:
            for j in range(16):
                jac = jax.jacfwd(atom_field)(p, j)  # (2, 2): d u_i / d x_k
                div = jac[0, 0] + jac[1, 1]
                np.testing.assert_allclose(div, 0.0, atol=1e-5)

    def test_eigenvalues_match_fourier(self):
        from geonnax.basis import fourier_eigenvalues

        _, lam = divfree_basis(jnp.zeros((2, 2)), num_basis_per_dim=3, L=2.0)
        expected = fourier_eigenvalues(3, 2.0, 2, dtype=lam.dtype)
        np.testing.assert_allclose(lam, expected, atol=1e-6)

    def test_rejects_non_2d_inputs(self):
        with pytest.raises(ValueError):
            divfree_basis(jnp.zeros((5, 3)), num_basis_per_dim=2, L=1.0)


class TestSphericalRBFBasis:
    def test_peak_at_centre(self):
        x = jnp.eye(3)  # three orthonormal points on the sphere
        phi = spherical_rbf_basis(x, x, jnp.ones(3))
        # geodesic distance to itself is 0 -> Gaussian peaks at 1 on the diagonal
        np.testing.assert_allclose(jnp.diag(phi), 1.0, atol=1e-6)

    @pytest.mark.parametrize("kernel", ["wendland_c2", "wendland_c4"])
    def test_wendland_compact_support_antipode(self, kernel):
        north = jnp.array([[0.0, 0.0, 1.0]])
        south = jnp.array([[0.0, 0.0, -1.0]])  # geodesic distance pi
        phi = spherical_rbf_basis(south, north, jnp.array([0.5]), kernel=kernel)
        assert float(phi[0, 0]) == 0.0

    def test_rotation_invariance(self):
        # A rigid rotation of both points and centres leaves the basis unchanged.
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (5, 3))
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        c = jnp.eye(3)
        # a rotation about z by 0.7 rad
        a = 0.7
        rot = jnp.array(
            [[jnp.cos(a), -jnp.sin(a), 0.0], [jnp.sin(a), jnp.cos(a), 0.0], [0, 0, 1.0]]
        )
        base = spherical_rbf_basis(x, c, jnp.ones(3))
        rotated = spherical_rbf_basis(x @ rot.T, c @ rot.T, jnp.ones(3))
        np.testing.assert_allclose(base, rotated, atol=1e-5)

    def test_rejects_unknown_kernel(self):
        with pytest.raises(ValueError):
            spherical_rbf_basis(jnp.eye(3), jnp.eye(3), jnp.ones(3), kernel="nope")


class TestWaveletBasis:
    @pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_orthonormal_and_reconstructs(self, wavelet, n):
        phi = wavelet_basis_1d(n, wavelet=wavelet)
        assert phi.shape == (n, n)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(n), atol=1e-6)
        # synthesis(analysis(f)) == f for an orthonormal basis
        f = jax.random.normal(jax.random.PRNGKey(5), (n,))
        np.testing.assert_allclose(phi @ (phi.T @ f), f, atol=1e-5)

    def test_levels_capped_and_partial(self):
        # one level is still orthonormal; more levels than log2(n) is capped.
        phi1 = wavelet_basis_1d(16, wavelet="haar", levels=1)
        np.testing.assert_allclose(phi1.T @ phi1, jnp.eye(16), atol=1e-6)
        full = wavelet_basis_1d(16, wavelet="haar")
        capped = wavelet_basis_1d(16, wavelet="haar", levels=99)
        np.testing.assert_allclose(full, capped, atol=1e-6)

    def test_2d_orthonormal(self):
        phi = wavelet_basis_2d(4, 8, wavelet="db2")
        assert phi.shape == (32, 32)
        np.testing.assert_allclose(phi.T @ phi, jnp.eye(32), atol=1e-6)

    def test_rejects_non_power_of_two_and_unknown(self):
        with pytest.raises(ValueError):
            wavelet_basis_1d(12, wavelet="haar")
        with pytest.raises(ValueError):
            wavelet_basis_1d(16, wavelet="db8")
