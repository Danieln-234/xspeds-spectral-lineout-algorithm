"""Tests for the cleaning and clustering stage on small synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleaning_and_clustering import detect_clusters, scrubbing


@pytest.fixture
def shapes_frame() -> NDArray[np.float64]:
    """Frame containing one cluster of each canonical shape, well separated."""
    frame = np.zeros((12, 12), dtype=np.float64)
    frame[1, 1] = 50.0                     # single
    frame[1, 4:6] = 40.0                   # line2 (horizontal)
    frame[4, 1] = frame[5, 1] = frame[6, 1] = 30.0   # line3 (vertical)
    frame[4, 4] = frame[5, 4] = frame[5, 5] = 25.0   # lshape3
    frame[8:10, 1:3] = 20.0                # box4
    return frame


def test_detect_clusters_classifies_shapes(shapes_frame: NDArray[np.float64]) -> None:
    """Each canonical cluster shape gets the right label."""
    clusters, _ = detect_clusters(shapes_frame)
    shapes = sorted(str(info["shape"]) for info in clusters.values())
    assert shapes == ["box4", "line2", "line3", "lshape3", "single"]


def test_detect_clusters_marks_centroids(shapes_frame: NDArray[np.float64]) -> None:
    """Photon map has exactly one centroid (value 1) per recognised cluster."""
    clusters, pmap = detect_clusters(shapes_frame)
    assert int(np.sum(pmap == 1)) == len(clusters)
    assert pmap[1, 1] == 1  # the single pixel is its own centroid


def test_detect_clusters_other_flagging() -> None:
    """Irregular clusters are marked 2 only above the flag threshold."""
    frame = np.zeros((7, 7), dtype=np.float64)
    # 5-pixel plus shape, not a canonical template
    frame[2, 3] = frame[4, 3] = frame[3, 2] = frame[3, 4] = 10.0
    frame[3, 3] = 100.0

    clusters, pmap = detect_clusters(frame, other_flag_threshold=90.0)
    assert list(info["shape"] for info in clusters.values()) == ["other"]
    assert pmap[3, 3] == 2

    frame[3, 3] = 50.0  # below threshold, cluster should be dropped from the map
    _, pmap = detect_clusters(frame, other_flag_threshold=90.0)
    assert int(np.count_nonzero(pmap)) == 0


def test_scrubbing_removes_noise_and_keeps_spikes() -> None:
    """Gaussian background is mostly zeroed while photon-like spikes survive."""
    rng = np.random.default_rng(0)
    frame = rng.normal(loc=30.0, scale=5.0, size=(20, 200))
    spikes = [(3, 40), (11, 100), (17, 160)]
    for r, c in spikes:
        frame[r, c] = 300.0

    scrubbed = scrubbing([frame], size_rows=5, lower_bound=1.0, upper_bound=5.0)[0]

    for r, c in spikes:
        assert scrubbed[r, c] == 300.0
    background = np.delete(scrubbed.ravel(), [r * 200 + c for r, c in spikes])
    assert np.count_nonzero(background) / background.size < 0.1


def test_scrubbing_rejects_bad_bounds() -> None:
    """Inverted search window raises."""
    with pytest.raises(ValueError):
        scrubbing([np.zeros((4, 4))], size_rows=2, lower_bound=5.0, upper_bound=1.0)
