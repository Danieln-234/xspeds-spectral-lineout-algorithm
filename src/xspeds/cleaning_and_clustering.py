"""
Cleaning and clustering stage of the XSPEDS algorithm.

Removes CCD background noise via Gaussian pedestal fitting and dynamic thresholding,
then identifies photon clusters (single pixel, lines, L-shapes, 2x2 boxes).

Produces per-frame photon maps (1 = centroid, 2 = large/irregular clusters)
and per-frame cluster metadata.

Example:
    from cleaning_and_clustering import ScrubConfig, run_cleaning_and_clustering

    result = run_cleaning_and_clustering(stack, scrub=ScrubConfig())
    photon_maps, clusters = result.photon_maps, result.clusters
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

Shape = Literal["single", "line2", "line3", "lshape3", "box4", "other"]
ClustersDict = dict[int, dict[str, object]]  # cluster_no -> cluster info


@dataclass(frozen=True)
class ScrubConfig:
    """Configuration for pedestal fitting and dynamic thresholding.

    Args:
        row_batch_size: Number of image rows per batch for the histogram fit.
        k_low: Lower bound (in sigma) of the threshold search window [mu + k_low*sigma, mu + k_high*sigma].
        k_high: Upper bound (in sigma) of the threshold search window.
        fallback_sigma_k: If the Gaussian fit fails, use threshold = mu + fallback_sigma_k*sigma.
        min_bins: Minimum histogram bin count per batch.
        max_bins: Maximum histogram bin count per batch.
        other_flag_threshold: Clusters classified "other" with max_value above this
            get their centroid marked 2 in the photon map (see paper).
    """
    row_batch_size: int = 5
    k_low: float = 1.0
    k_high: float = 5.0
    fallback_sigma_k: float = 3.0
    min_bins: int = 16
    max_bins: int = 256
    other_flag_threshold: float = 90.0


@dataclass(frozen=True)
class ClusteringResult:
    """Outputs of cleaning + clustering.

    Args:
        photon_maps: One (H, W) int map per frame.
        clusters: One metadata dict per frame.
    """
    photon_maps: list[NDArray[np.int_]]
    clusters: list[ClustersDict]


#############################
#        Utilities          #
#############################

def _scotts_rule_bins(batch: NDArray[np.float64], min_bins: int, max_bins: int) -> int:
    """Histogram bin count via Scott's rule, clipped to [min_bins, max_bins].

    NB for these datasets the raw rule usually lands below min_bins, so the
    clip is what matters in practice.

    Args:
        batch: Flattened pixel values for one row batch.
        min_bins: Lower clip.
        max_bins: Upper clip.

    Returns:
        Bin count to use for the batch histogram.
    """
    sigma = float(np.std(batch)) if batch.size > 1 else 0.0
    data_range = float(np.max(batch) - np.min(batch)) if batch.size else 0.0
    if sigma <= 0.0 or data_range <= 0.0:
        return min_bins
    bin_w = 3.5 * sigma / (batch.size ** (1.0 / 3.0))
    bins = max(round(data_range / bin_w), 1)
    return int(np.clip(bins, min_bins, max_bins))


def _gauss(x: NDArray[np.float64], amp: float, mu: float, sigma: float) -> NDArray[np.float64]:
    """Gaussian model for the pedestal fit."""
    sigma = abs(float(sigma)) + 1e-12  # guard against zero width during fitting
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


#####################################
#       Cleaning / Scrubbing        #
#####################################

def scrubbing(
    image_data: Sequence[NDArray[np.float64]],
    size_rows: int,
    lower_bound: float,
    upper_bound: float,
    *,
    min_bins: int = 16,
    max_bins: int = 256,
    fallback_sigma_k: float = 3.0,
) -> list[NDArray[np.float64]]:
    """Fit a Gaussian pedestal per row batch and zero everything below a dynamic threshold.

    The threshold is the bin centre in [mu + k_low*sigma, mu + k_high*sigma] where the
    observed counts are closest to twice the fitted Gaussian prediction. If the fit
    fails the batch falls back to mu + fallback_sigma_k*sigma from sample moments.

    Args:
        image_data: Sequence of 2D frames (H x W, ADU values).
        size_rows: Rows per batch for the histogram fit.
        lower_bound: k_low of the search window (in sigma).
        upper_bound: k_high of the search window (in sigma).
        min_bins: Lower clip on histogram bin count.
        max_bins: Upper clip on histogram bin count.
        fallback_sigma_k: Multiplier for the fallback threshold.

    Returns:
        Scrubbed frames (same shapes as input) with background set to zero.
    """
    if size_rows <= 0:
        raise ValueError("size_rows must be positive.")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must be greater than lower_bound.")

    scrubbed_frames: list[NDArray[np.float64]] = []

    for frame in image_data:
        f = frame.copy()
        H = f.shape[0]

        for r0 in range(0, H, size_rows):
            r1 = min(r0 + size_rows, H)  # include last partial batch
            batch = f[r0:r1, :].ravel().astype(np.float64, copy=False)

            bins = _scotts_rule_bins(batch, min_bins, max_bins)
            counts, edges = np.histogram(batch, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])

            p0 = [float(np.max(counts)), float(np.mean(batch)), float(np.std(batch) + 1e-6)]
            try:
                (amp, mu, sigma), _ = curve_fit(_gauss, centers, counts, p0=p0, maxfev=2000)
                sigma = abs(float(sigma)) + 1e-12
                fit_ok = True
            except RuntimeError:
                mu, sigma = p0[1], p0[2]
                fit_ok = False

            # Pick the bin centre in the window where counts are closest to 2x the fit
            mask = (centers >= mu + lower_bound * sigma) & (centers <= mu + upper_bound * sigma)
            if fit_ok and np.any(mask):
                pred = _gauss(centers, amp, mu, sigma)
                idx = int(np.argmin(np.abs(counts[mask] - 2.0 * pred[mask])))
                threshold = float(centers[mask][idx])
            else:
                threshold = float(mu + fallback_sigma_k * sigma)

            f[r0:r1, :] = np.where(f[r0:r1, :] < threshold, 0.0, f[r0:r1, :])

        scrubbed_frames.append(f)

    return scrubbed_frames


##########################
#      Clustering        #
##########################

# Canonical shape templates, anchored at (min row, min col)
_TEMPLATES: dict[frozenset[tuple[int, int]], Shape] = {
    frozenset({(0, 0)}): "single",
    frozenset({(0, 0), (0, 1)}): "line2",
    frozenset({(0, 0), (1, 0)}): "line2",
    frozenset({(0, 0), (0, 1), (0, 2)}): "line3",
    frozenset({(0, 0), (1, 0), (2, 0)}): "line3",
    frozenset({(0, 0), (1, 0), (1, 1)}): "lshape3",
    frozenset({(0, 1), (1, 1), (1, 0)}): "lshape3",
    frozenset({(0, 0), (0, 1), (1, 0)}): "lshape3",
    frozenset({(0, 0), (0, 1), (1, 1)}): "lshape3",
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}): "box4",
}


def _identify_shape(coords: set[tuple[int, int]]) -> Shape:
    """Classify a cluster by matching its normalised footprint against the templates.

    Args:
        coords: Pixel coordinates of one cluster.

    Returns:
        Shape label, "other" if no template matches.
    """
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = frozenset((r - min_r, c - min_c) for r, c in coords)
    return _TEMPLATES.get(norm, "other")


def detect_clusters(
    image: NDArray[np.float64],
    *,
    other_flag_threshold: float = 90.0,
) -> tuple[ClustersDict, NDArray[np.int_]]:
    """Find 4-connected clusters, classify their shapes, and build the photon map.

    Args:
        image: 2D scrubbed frame (H x W), background pixels are zero.
        other_flag_threshold: If shape is "other" and max_value exceeds this,
            mark the centroid as 2 instead of dropping the cluster.

    Returns:
        clusters: Mapping cluster_no -> info dict with keys
            'coords' (set of (r, c)), 'shape', 'max_value', 'values',
            'sum', 'centroid' (position of the max pixel).
        centroid_map: (H x W) int array, 1 = photon centroid, 2 = large/irregular, 0 = background.
    """
    H, W = image.shape
    visited = np.zeros_like(image, dtype=bool)
    centroid_map = np.zeros_like(image, dtype=int)
    clusters: ClustersDict = {}
    cluster_no = 0

    def neighbors(r: int, c: int) -> Iterator[tuple[int, int]]:
        if r > 0:
            yield (r - 1, c)
        if r + 1 < H:
            yield (r + 1, c)
        if c > 0:
            yield (r, c - 1)
        if c + 1 < W:
            yield (r, c + 1)

    for i in range(H):
        for j in range(W):
            if image[i, j] == 0 or visited[i, j]:
                continue

            # BFS to collect one cluster
            cluster_no += 1
            q: deque[tuple[int, int]] = deque([(i, j)])
            visited[i, j] = True
            coords: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for rr, cc in neighbors(r, c):
                    if not visited[rr, cc] and image[rr, cc] != 0:
                        visited[rr, cc] = True
                        q.append((rr, cc))

            shape = _identify_shape(set(coords))
            values = [float(image[rc]) for rc in coords]
            centroid = coords[int(np.argmax(values))]
            max_val = float(image[centroid])

            clusters[cluster_no] = {
                "coords": set(coords),
                "shape": shape,
                "max_value": max_val,
                "values": values,
                "sum": float(sum(values)),
                "centroid": centroid,
            }

            if shape != "other":
                centroid_map[centroid] = 1
            elif max_val > other_flag_threshold:
                centroid_map[centroid] = 2

    return clusters, centroid_map


##################################
#              Run               #
##################################

def run_cleaning_and_clustering(
    image_data: Sequence[NDArray[np.float64]],
    *,
    scrub: ScrubConfig,
) -> ClusteringResult:
    """Scrub a stack of frames and detect clusters in each.

    Args:
        image_data: Sequence of 2D frames (H x W).
        scrub: Configuration for pedestal fit, threshold search, and flags.

    Returns:
        ClusteringResult with one photon map and one cluster dict per frame.
    """
    scrubbed = scrubbing(
        image_data,
        size_rows=scrub.row_batch_size,
        lower_bound=scrub.k_low,
        upper_bound=scrub.k_high,
        min_bins=scrub.min_bins,
        max_bins=scrub.max_bins,
        fallback_sigma_k=scrub.fallback_sigma_k,
    )

    photon_maps: list[NDArray[np.int_]] = []
    cluster_info: list[ClustersDict] = []
    for frame in scrubbed:
        clusters, pmap = detect_clusters(frame, other_flag_threshold=scrub.other_flag_threshold)
        photon_maps.append(pmap)
        cluster_info.append(clusters)

    return ClusteringResult(photon_maps=photon_maps, clusters=cluster_info)
