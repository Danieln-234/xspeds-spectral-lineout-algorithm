"""Tests for the shared cone-plane geometry."""

from __future__ import annotations

import numpy as np

from xspeds.geometry import conic_coefficients, rotated_basis


def test_rotated_basis_is_orthonormal() -> None:
    """Basis vectors stay unit length and orthogonal for arbitrary rotations."""
    for theta_z in (-2.45, -0.7, 0.0, 1.2):
        e_i, e_j = rotated_basis(theta_z)
        assert np.isclose(np.linalg.norm(e_i), 1.0)
        assert np.isclose(np.linalg.norm(e_j), 1.0)
        assert np.isclose(np.dot(e_i, e_j), 0.0)


def test_conic_coefficients_match_cone_condition() -> None:
    """The conic value at (u, v) equals the cone condition y^2 + z^2 - tan^2(alpha) x^2
    evaluated at the corresponding 3D point on the detector plane."""
    rng = np.random.default_rng(1)
    alpha, d, theta_z = 0.85, -8000.0, -2.45

    e_i, e_j = rotated_basis(theta_z)
    A, B, C, D, E, F = conic_coefficients(alpha, d, e_i, e_j)
    T2 = np.tan(alpha) ** 2

    for _ in range(20):
        u, v = rng.uniform(-3000, 3000, size=2)
        x, y, z = u * e_i + v * e_j + np.array([-d, 0.0, 0.0])
        conic_val = A * u**2 + B * u * v + C * v**2 + D * u + E * v + F
        cone_val = y**2 + z**2 - T2 * x**2
        assert np.isclose(conic_val, cone_val, rtol=1e-9)
