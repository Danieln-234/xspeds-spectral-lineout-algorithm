"""
Cleaning and clustering stage of the XSPEDS algorithm.

Removes CCD background noise via Gaussian pedestal fitting and dynamic thresholding,
then identifies photon clusters (single pixel, lines, L-shapes, 2×2 boxes).

Produces:
- scrubbed frames
- per-frame cluster metadata
- per-frame photon maps (1 = centroid, 2 = large/irregular clusters)

Example
-------
from cleaning_and_clustering import ScrubConfig, run_cleaning_and_clustering

scrub = ScrubConfig(row_batch_size=5, k_low=1.0, k_high=5.0)
result = run_cleaning_and_clustering(stack, scrub=scrub)
photon_maps, clusters = result.as_tuple()

Notes
-----
- This module is intentionally self-contained for portfolio/review purposes.
- See ScrubConfig for all tunable parameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


# ------------------------------- Public types -------------------------------

Shape = Literal["single", "line2", "line3", "lshape3", "box4", "other"]
ClustersDict = Dict[int, Dict[str, object]]  # cluster_no -> cluster info


# ------------------------------- Configuration ------------------------------

@dataclass(frozen=True)
class ScrubConfig:
    """
    Configuration for pedestal fitting and dynamic thresholding.

    Parameters
    ----------
    row_batch_size : int
        Number of image rows per batch for histogram/fit (e.g., 5).
    k_low : float
        Lower bound (in σ) for threshold search window [μ + k_low σ, μ + k_high σ].
    k_high : float
        Upper bound (in σ) for threshold search window.
    fallback_sigma_k : float
        If fit/search is unstable, fallback threshold = μ + fallback_sigma_k σ.
    min_bins : int
        Minimum histogram bin count for the batch histogram.
    max_bins : int
        Maximum histogram bin count for the batch histogram.
    other_flag_threshold : float
        If a cluster is classified as "other" and its max_value exceeds this,
        mark its centroid as 2 in the photon map.
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
    """Outputs of cleaning + clustering."""
    photon_maps: List[NDArray[np.int_]]   # list of (H, W)
    clusters: List[ClustersDict]          # list of dicts

    def as_tuple(self):
        """Back-compat: return exactly what the old code expects."""
        return self.photon_maps, self.clusters


# ------------------------------- Utilities ----------------------------------

def _scotts_rule_bins(batch: NDArray[np.float64], *, min_bins: int, max_bins: int) -> int:
    """
    Compute a histogram bin count using Scott’s rule, clipped to [min_bins, max_bins].
    """
    x = batch
    n = max(int(x.size), 1)
    sigma = float(np.std(x)) if n > 1 else 0.0  # Scott’s rule uses σ
    if sigma <= 0.0:
        return min_bins  # degenerate case
    bin_w = (3.5 * sigma) / (n ** (1.0 / 3.0))
    if bin_w <= 0:
        return min_bins
    data_range = float(np.max(x) - np.min(x))
    if data_range <= 0:
        return min_bins
    bins = int(max(round(data_range / bin_w), 1))
    return int(np.clip(bins, min_bins, max_bins))


def _gauss(x: NDArray[np.float64], amp: float, mu: float, sigma: float) -> NDArray[np.float64]:
    """
    Simple Gaussian model used for pedestal fitting.
    """
    sigma = abs(float(sigma)) + 1e-12  # guard against negative / zero
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
) -> List[NDArray[np.float64]]:
    """
    Fit a Gaussian pedestal per row-batch and dynamically threshold each batch.

    Parameters
    ----------
    image_data : Sequence[np.ndarray]
        Sequence of 2D arrays (each frame is H×W, dtype float-like ADU).
    size_rows : int
        Number of rows per batch for histogram/fit (e.g., 5).
    lower_bound, upper_bound : float
        Bounds (in σ) for the threshold search window.
    min_bins, max_bins : int
        Bounds on histogram bin count.
    fallback_sigma_k : float
        Fallback multiplier k for threshold = μ + k σ when fit/search unstable.

    Returns
    -------
    list[np.ndarray]
        Scrubbed image data (same shapes as input), background set to zero.
    """
    if size_rows <= 0:
        raise ValueError("size_rows must be positive.")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must be greater than lower_bound.")

    scrubbed_frames: List[NDArray[np.float64]] = []

    for frame in image_data:
        if frame.ndim != 2:
            raise ValueError("Each item in image_data must be a 2D array.")
        f = frame.copy()

        H, _ = f.shape
        for r0 in range(0, H, size_rows):
            r1 = min(r0 + size_rows, H)  # include last partial batch
            batch = f[r0:r1, :].ravel().astype(np.float64, copy=False)

            # Histogram (Scott's rule, with sane bounds)
            bins = _scotts_rule_bins(batch, min_bins=min_bins, max_bins=max_bins)
            counts, edges = np.histogram(batch, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])

            # Fit Gaussian to pedestal
            p0 = [float(np.max(counts) if counts.size else 1.0),
                  float(np.mean(batch)),
                  float(np.std(batch) + 1e-6)]
            try:
                (amp, mu, sigma), _ = curve_fit(_gauss, centers, counts, p0=p0, maxfev=2000)
                sigma = abs(float(sigma)) + 1e-12
                fit_ok = True
            except Exception:
                # Degenerate/unstable fit; fall back to moments
                mu = float(np.mean(batch))
                sigma = float(np.std(batch) + 1e-6)
                amp = float(np.max(counts)) if counts.size else 1.0
                fit_ok = False

            # Threshold rule: pick V in [μ + k_low σ, μ + k_high σ] where counts ≈ 2 × fitted
            lower_v = mu + lower_bound * sigma
            upper_v = mu + upper_bound * sigma

            mask = (centers >= lower_v) & (centers <= upper_v)
            if fit_ok and np.any(mask):
                pred = _gauss(centers, amp, mu, sigma)
                # choose bin center where |observed - 2*predicted| is minimized
                diffs = np.abs(counts[mask] - 2.0 * pred[mask])
                idx_local = int(np.argmin(diffs))
                threshold = float(centers[mask][idx_local])
            else:
                # Fallback: μ + k σ
                threshold = float(mu + fallback_sigma_k * sigma)

            # Apply threshold to the batch slice
            f[r0:r1, :] = np.where(f[r0:r1, :] < threshold, 0.0, f[r0:r1, :])

        scrubbed_frames.append(f)

    return scrubbed_frames

##########################
#      Clustering        #
##########################

# Canonical shape templates (in local coords anchored at min(r), min(c))
SINGLE = [{(0, 0)}]
LINE_TWO = [{(0, 0), (0, 1)}, {(0, 0), (1, 0)}]
LINE_THREE = [{(0, 0), (0, 1), (0, 2)}, {(0, 0), (1, 0), (2, 0)}]
L_SHAPE = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
]
BOX = [{(0, 0), (0, 1), (1, 0), (1, 1)}]


def detect_clusters(
    image: NDArray[np.float64],
    *,
    other_flag_threshold: float = 90.0
) -> Tuple[ClustersDict, NDArray[np.int_]]:
    """
    Identify 4-connected clusters, classify simple shapes, and return a photon map.

    Parameters
    ----------
    image : np.ndarray
        2D scrubbed NumPy array (H×W), where background pixels are zero.
    other_flag_threshold : float
        If shape=='other' and max_value > this threshold, mark the centroid as 2.

    Returns
    -------
    clusters_dict : dict
        Mapping from cluster number to cluster info dict with keys:
            - 'coords': set[(r, c)]
            - 'shape': {'single','line2','line3','lshape3','box4','other'}
            - 'max_value': float
            - 'values': list[float]
            - 'sum': float
            - 'centroid': (r, c)
    centroid_map : np.ndarray
        2D int array (H×W): 1 = centroid; 2 = large/irregular; 0 = background.
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    H, W = image.shape
    visited = np.zeros_like(image, dtype=bool)

    def neighbors(r: int, c: int) -> Iterable[Tuple[int, int]]:
        if r > 0:
            yield (r - 1, c)
        if r + 1 < H:
            yield (r + 1, c)
        if c > 0:
            yield (r, c - 1)
        if c + 1 < W:
            yield (r, c + 1)

    def identify_shape(coords_set: set[Tuple[int, int]]) -> Shape:
        # Normalize to top-left origin to compare with templates
        min_r = min(r for r, _ in coords_set)
        min_c = min(c for _, c in coords_set)
        norm = {(r - min_r, c - min_c) for (r, c) in coords_set}
        size = len(norm)

        if size == 1 and norm in SINGLE:
            return "single"
        if size == 2 and norm in LINE_TWO:
            return "line2"
        if size == 3:
            if norm in LINE_THREE:
                return "line3"
            if norm in L_SHAPE:
                return "lshape3"
        if size == 4 and norm in BOX:
            return "box4"
        return "other"

    clusters: ClustersDict = {}
    cluster_no = 0

    for i in range(H):
        for j in range(W):
            if image[i, j] == 0 or visited[i, j]:
                continue

            # BFS to collect one cluster
            cluster_no += 1
            q: deque[Tuple[int, int]] = deque([(i, j)])
            visited[i, j] = True
            coords: List[Tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for rr, cc in neighbors(r, c):
                    if not visited[rr, cc] and image[rr, cc] != 0:
                        visited[rr, cc] = True
                        q.append((rr, cc))

            coords_set = set(coords)
            shape = identify_shape(coords_set)
            values = [float(image[r, c]) for (r, c) in coords_set]
            max_coord = max(coords_set, key=lambda rc: image[rc])
            max_val = float(image[max_coord])
            clusters[cluster_no] = {
                "coords": coords_set,
                "shape": shape,
                "max_value": max_val,
                "values": values,
                "sum": float(sum(values)),
                "centroid": max_coord,
            }

    # Build centroid map
    centroid_map = np.zeros_like(image, dtype=int)
    for info in clusters.values():
        r, c = info["centroid"]  # type: ignore[index]
        shape: str = info["shape"]  # type: ignore[assignment]
        max_val: float = info["max_value"]  # type: ignore[assignment]
        if shape != "other":
            if 0 <= r < H and 0 <= c < W:
                centroid_map[r, c] = 1
        else:
            if max_val > other_flag_threshold and 0 <= r < H and 0 <= c < W:
                centroid_map[r, c] = 2

    return clusters, centroid_map

# Back-compat alias so external code doesn’t break if it used the old name
cluster_detecting = detect_clusters


##################################
#              Run               #
##################################

def run_cleaning_and_clustering(
    image_data: Sequence[NDArray[np.float64]],
    *,
    scrub: ScrubConfig,
) -> ClusteringResult:
    """
    Scrub a stack of images and detect clusters, producing photon maps and cluster metadata.

    Parameters
    ----------
    image_data : sequence of 2D NumPy arrays (each frame H×W)
    scrub : ScrubConfig
        Centralized configuration for pedestal fit, threshold search, and flags.

    Returns
    -------
    ClusteringResult
        .photon_maps (list of H×W int arrays) and .clusters (list of dicts).
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

    photon_maps: List[NDArray[np.int_]] = []
    cluster_info: List[ClustersDict] = []

    for frame in scrubbed:
        clusters, pmap = detect_clusters(
            frame,
            other_flag_threshold=scrub.other_flag_threshold
        )
        photon_maps.append(pmap)
        cluster_info.append(clusters)

    return ClusteringResult(photon_maps=photon_maps, clusters=cluster_info)
