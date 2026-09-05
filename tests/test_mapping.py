"""Tests for ridge extraction and its uncertainty estimate."""

from __future__ import annotations

import numpy as np

from mapping import _sigma_from_peak_curvature, find_scatter_peaks


def test_find_scatter_peaks_locates_ridges() -> None:
    """Two synthetic vertical ridges are found at their known columns in every batch."""
    H, W = 200, 1600
    cols = np.arange(W, dtype=np.float64)
    profile = (
        100.0 * np.exp(-0.5 * ((cols - 1300.0) / 8.0) ** 2)
        + 80.0 * np.exp(-0.5 * ((cols - 1450.0) / 8.0) ** 2)
    )
    frame = np.tile(profile, (H, 1))

    rows, x1, sx1, x2, sx2 = find_scatter_peaks(frame, batch_size=50)

    assert rows.size == 4  # 200 rows / 50 per batch
    assert np.all(np.abs(x1 - 1300.0) <= 2.0)
    assert np.all(np.abs(x2 - 1450.0) <= 2.0)
    assert np.all(sx1 > 0) and np.all(sx2 > 0)


def test_sigma_from_peak_curvature_recovers_gaussian_width() -> None:
    """Curvature estimate at a Gaussian maximum returns roughly its sigma."""
    x = np.arange(200, dtype=np.float64)
    sigma_true = 5.0
    trace = 50.0 * np.exp(-0.5 * ((x - 100.0) / sigma_true) ** 2)

    sigma_est = _sigma_from_peak_curvature(trace, 100)
    assert abs(sigma_est - sigma_true) < 0.5


def test_sigma_from_peak_curvature_flat_trace_hits_ceiling() -> None:
    """A flat trace has no curvature, so the estimate falls back to the ceiling."""
    trace = np.full(50, 10.0)
    assert _sigma_from_peak_curvature(trace, 25, ceil=20.0) == 20.0
