"""
Shared cone-plane geometry for the mapping and lineout stages.

A photon of energy E leaves the crystal on a cone of half-angle alpha (from Bragg's law).
The intersection of that cone with the tilted CCD plane is a conic in detector
coordinates (u, v). Both mapping (fitting the geometry) and lineout (sweeping
iso-energy conics) need the same two building blocks, kept here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotated_basis(
    theta_z: float,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the CCD in-plane orthonormal basis vectors (e_i, e_j) after rotation.

    NB the pipeline only fits theta_z; theta_x and theta_y are kept as arguments
    in case they are added to the optimiser later.

    Args:
        theta_z: Rotation about z (radians).
        theta_x: Rotation about x (radians), default 0.
        theta_y: Rotation about y (radians), default 0.

    Returns:
        Tuple (e_i, e_j) of unit vectors spanning the detector plane.
    """
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    R = Rx @ Ry @ Rz

    e_i0 = np.array([0.0, 1.0, 0.0])
    e_j0 = np.array([0.0, 0.0, 1.0])
    return (R @ e_i0).astype(np.float64), (R @ e_j0).astype(np.float64)


def conic_coefficients(
    alpha: float,
    d: float,
    e_i: NDArray[np.float64],
    e_j: NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """Return (A, B, C, D, E, F) of the cone-plane intersection in CCD coordinates.

    The conic satisfies A u^2 + B u v + C v^2 + D u + E v + F = 0.
    See the project report for the derivation.

    Args:
        alpha: Cone half-angle (radians).
        d: Source-detector distance (pixels, negative in our convention).
        e_i: First in-plane basis vector from rotated_basis.
        e_j: Second in-plane basis vector from rotated_basis.

    Returns:
        The six conic coefficients as floats.
    """
    T = np.tan(alpha)
    A1, A2, A3 = e_i
    B1, B2, B3 = e_j

    A = -T**2 * (A1**2) + (A2**2 + A3**2)
    B = 2 * (A2 * B2 + A3 * B3) - 2 * T**2 * (A1 * B1)
    C = (B2**2 + B3**2) - T**2 * (B1**2)
    D = 2 * T**2 * d * A1
    E = 2 * T**2 * d * B1
    F = -T**2 * d**2
    return float(A), float(B), float(C), float(D), float(E), float(F)
