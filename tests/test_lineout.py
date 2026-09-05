"""Tests for the lineout building blocks and peak metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lineout import _alpha_from_energy, _horizontal_boxsum, compute_peak_metrics


def test_alpha_energy_roundtrip() -> None:
    """Bragg conversion inverts: E built from a known alpha maps back to it."""
    two_d = 15.96
    alpha_true = 0.8
    E = 12398.0 / (two_d * np.cos(alpha_true))
    alpha = _alpha_from_energy(np.array([E]), two_d)
    assert np.isclose(alpha[0], alpha_true, rtol=1e-9)


def test_horizontal_boxsum_matches_manual_sum() -> None:
    """Box sum with tol=1 equals the three-pixel neighbourhood sum with zero padding."""
    grid: NDArray[np.float64] = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    expected = np.array([[3.0, 6.0, 9.0, 12.0, 9.0]])
    assert np.array_equal(_horizontal_boxsum(grid, 1), expected)


def test_horizontal_boxsum_zero_tol_is_identity() -> None:
    """tol=0 returns the grid unchanged."""
    grid = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert np.array_equal(_horizontal_boxsum(grid, 0), grid)


def test_compute_peak_metrics_recovers_synthetic_peak() -> None:
    """A synthetic Gaussian on a flat noisy baseline is recovered within tolerance."""
    rng = np.random.default_rng(2)
    energies = np.arange(1100.0, 1300.0, 0.1)
    mu_true, sigma_true = 1188.0, 0.85
    intensity = (
        50.0
        + 200.0 * np.exp(-0.5 * ((energies - mu_true) / sigma_true) ** 2)
        + rng.normal(0.0, 2.0, size=energies.size)
    )

    metrics = compute_peak_metrics(energies, intensity)

    assert abs(float(metrics["mu"]) - mu_true) < 0.5
    assert abs(float(metrics["FWHM"]) - 2.355 * sigma_true) < 0.5
    assert float(metrics["SNR"]) > 5.0
