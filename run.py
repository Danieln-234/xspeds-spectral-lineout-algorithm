"""XSPEDS pipeline runner (load → clean+cluster → mapping → lineout).

Overview:
--------
This script demonstrates an end-to-end implementation of the XSPEDS algorithm:
it converts raw CCD frames into a physically-meaningful spectrum (counts per eV).

Pipeline stages:
---------------
1) Load:
   Read a CCD frame stack from HDF5 (dataset-specific layout). Each CCD frame is a 2048x2048 2D Numpy array. The first three
   columns are dropped to avoid spurious edge values observed in this dataset. We have 20 frames in the HD5 file example

2) Cleaning + Clustering (SPC):
   Fit Gaussian pedestals per row-batch and derive dynamic thresholds to which we scrub 
   (e.g. threhold of 90, we set every pixel with a number below 90 to 0). Then cluster
   photon hits (single pixels, small shapes), idenitfying them based on the shapes mentioned in paper (with some statistical justification).
   Outputs per-frame "photon maps".

3) Mapping (instrument calibration):
   Fit the cone–plane geometry and mapping offsets from reference ridges to
   obtain energy-dependent conic parameters in CCD coordinates.

4) Lineout (physics output):
   Sum along iso-energy conics and normalize by the local eV window width to
   yield counts per eV. Apply Wiener smoothing for display and
   draw ±k·σ Poisson uncertainty bands as a shaded region. Signal-to-Noise of the Lα is found 
"""

from __future__ import annotations
import itertools
import logging
from pathlib import Path
import time

import h5py
import numpy as np

# Project modules
from cleaning_and_clustering import run_cleaning_and_clustering, ScrubConfig  # ClusterConfig optional
from mapping import run_mapping, MappingConfig
from lineout import run_lineout, LineoutConfig


# Config for the run, most important is E_Step, tolerance, and frame index for lineout
CONFIG = {
    #  Input 
    "INPUT_FILE": "sxro6416-r0504.h5",     # HDF5 CCD dataset

    #  Logging 
    "LOG_LEVEL": "INFO",

    #  Cleaning / SPC (dataset-dependent, but paper-aligned defaults) 
    "ROW_BATCH_SIZE": 5,                   # Rows per batch for pedestal histogram/fit
    "K_LOW": 1.0,                          # Lower σ bound for threshold search
    "K_HIGH": 5.0,                         # Upper σ bound for threshold search
    "FALLBACK_SIGMA_K": 3.0,               # Fallback μ + kσ if fit/search is unstable

    #  Mapping (reference ridge extraction + conic fit) 
    "MAP_FRAME_INDEX": 8,                  # Frame analyzed for mapping
    "MAP_BATCH_SIZE": 50,                  # Rows per batch for column-sum ridge extraction
    "MAP_SMOOTH_SIGMA": 10.0,              # Gaussian σ for smoothing the column-sum trace
    "MAP_R1": (1250, 1370),                # Reference region 1 [start, end) (where the Lβ curve is)
    "MAP_R2": (1380, 1560),                # Reference region 2 [start, end) (where the Lα curve is)
    "MAP_ALPHA1_DEG": 90.0 - 39.632,       # Half-angle for the Lβ emission (1218 eV)
    "MAP_ALPHA2_DEG": 90.0 - 40.86,        # Half-angle for the Lα emission (1188 eV)
    "MAP_DE_MAXITER": 2000,                # Differential-evolution max iterations
    "MAP_DE_SEED": 42,                     # Seed for reproducibility (None to disable)
    "MAP_W_FOCAL": 100.0,                  # Weight for focal-length residual
    "MAP_W_VERTEX": 100.0,                 # Weight for vertex-spacing residual

    #  Lineout (energy sweep and integration) 
    "E_MIN": 1100.0,                       # eV
    "E_MAX": 1604.0,                       # eV (exclusive)
    "E_STEP": 0.1,                         # eV (used in paper; increase for a quicker demo)
    "TOLERANCE_PX": 2,                     # Lateral half-width (pixels) around each conic
    "LINEOUT_FRAME": 1,                    # Photon-map index to analyze
    "HYPERB_BRANCH": "positive",           # "positive" | "negative"
    "X_MIN": 0,                            # Parabola sampling min x (pixels)
    "X_MAX": None,                         # Parabola sampling max x (None → grid width)
    "PARABOLA_SAMPLES": 3000,              # Integration samples along parabola

    #  Plotting and display smoothing 
    "PLOT": True,                          # Plot the lineout
    "PLOT_MODE": "smoothed",               # "raw" | "smoothed" | "both"
    "WIENER_MYSIZE": 30,                   # Wiener window (in bins): ≈ FWHM / E_STEP
    "ERROR_BAND_K": 2.0,                   # Shade ±k·σ Poisson uncertainty
    "Y_SCALE": "linear",                   # "linear" | "log"
    "SAVE_FIG_PATH": None,                 # e.g., "lineout.svg" to save the figure
}


################################
#             Logging          #
################################
def setup_logging(level: str = "INFO") -> None:
    """Configure a root logger so messages from all submodules are consistent."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


#######################################
#              Load Data              #
#######################################
def load_image_data(f_name: str) -> np.ndarray:
    """
    Load CCD frames from an HDF5 file and drop the first three columns.

    The dataset is assumed to follow the Princeton FrameV2 structure used in this
    project. If adapting to a different source, adjust the HDF5 path below and
    revisit the column-drop convention.

    Returns
    -------
    stack : np.ndarray, shape (N, H, W)
        Stack of frames as float64, with the first three columns removed.
    """
    path = Path(f_name)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    frames: list[np.ndarray] = []
    with h5py.File(str(path), "r") as f:
        for i in itertools.count(start=0):
            node = f.get(
                f"Configure:0000/Run:0000/CalibCycle:{i:04d}/"
                "Princeton::FrameV2/SxrEndstation.0:Princeton.0/data"
            )
            if node is None:
                break
            # Drop first 3 columns (dataset-specific spike/edge artefacts).
            frames.append(node[0][:, 3:])

    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")
    return stack


##################################
#            Pipeline            # 
##################################
def main() -> None:
    cfg = CONFIG
    setup_logging(cfg["LOG_LEVEL"])
    log = logging.getLogger("xspeds.run")

    t0 = time.perf_counter()

    # LOAD
    stack = load_image_data(cfg["INPUT_FILE"])
    log.info("Loaded stack: shape=%s (frames, rows, cols)", stack.shape)

    # CLEANING + CLUSTERING (SPC)
    # Gaussian pedestal fit per row-batch → dynamic threshold → BFS-style clustering.
    scrub_cfg = ScrubConfig(
        row_batch_size=cfg["ROW_BATCH_SIZE"],
        k_low=cfg["K_LOW"],
        k_high=cfg["K_HIGH"],
        fallback_sigma_k=cfg["FALLBACK_SIGMA_K"],
    )


    cl_res = run_cleaning_and_clustering(stack, scrub=scrub_cfg)
   

    photon_maps, cluster_meta = cl_res.as_tuple()
    total_clusters = sum(len(d) for d in cluster_meta)
    log.info("Clustering complete: frames=%d, total_clusters=%d",
             len(photon_maps), total_clusters)

    # MAPPING (instrument calibration)
    # Fit geometry + mapping offsets from two reference ridge regions.
    map_cfg = MappingConfig(
        frame_index=cfg["MAP_FRAME_INDEX"],
        batch_size=cfg["MAP_BATCH_SIZE"],
        smooth_sigma=cfg["MAP_SMOOTH_SIGMA"],
        r1=cfg["MAP_R1"],
        r2=cfg["MAP_R2"],
        alpha1_deg=cfg["MAP_ALPHA1_DEG"],
        alpha2_deg=cfg["MAP_ALPHA2_DEG"],
        de_maxiter=cfg["MAP_DE_MAXITER"],
        de_seed=cfg["MAP_DE_SEED"],
        w_focal=cfg["MAP_W_FOCAL"],
        w_vertex=cfg["MAP_W_VERTEX"],
    )

    try:
        map_out = run_mapping(stack, config=map_cfg)
    except TypeError:
        # Back-compat call if older signature is present
        map_out = run_mapping(stack)

    (d_opt, theta_z_opt, C1_opt, b_opt, shift_part_1) = map_out.as_tuple()
    log.info(
        "Mapping parameters: d=%.4g, θz=%.3f° , C1=%.4g px, b=%.4g px, shift=%.4g px",
        d_opt, float(np.rad2deg(theta_z_opt)), C1_opt, b_opt, shift_part_1
    )

    # SPECTRAL LINEOUT
    # Sum along iso-energy conics and normalize by local eV window width.
    lcfg = LineoutConfig(
        energy_min=cfg["E_MIN"],
        energy_max=cfg["E_MAX"],
        energy_step=cfg["E_STEP"],
        tolerance=cfg["TOLERANCE_PX"],
        frame_index=cfg["LINEOUT_FRAME"],
        hyperbola_branch=cfg["HYPERB_BRANCH"],
        x_min=cfg["X_MIN"],
        x_max=cfg["X_MAX"],
        num_points_parabola=cfg["PARABOLA_SAMPLES"],

        plot=cfg["PLOT"],

    )

    _lineout = run_lineout(
        photon_maps, d_opt, theta_z_opt, C1_opt, b_opt, shift_part_1, config=lcfg
    )

    # Optional: save the figure if plotting is enabled and a path is provided.
    if cfg["PLOT"] and cfg["SAVE_FIG_PATH"]:
        import matplotlib.pyplot as plt  # local import only when needed
        Path(cfg["SAVE_FIG_PATH"]).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cfg["SAVE_FIG_PATH"], dpi=300, bbox_inches="tight")
        log.info("Saved figure → %s", Path(cfg["SAVE_FIG_PATH"]).resolve())

    dt = time.perf_counter() - t0
    log.info("Pipeline finished in %.2fs", dt)


# Standard script entry point
if __name__ == "__main__":
    main()
