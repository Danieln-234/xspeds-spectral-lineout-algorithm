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
import matplotlib.pyplot as plt

import h5py
import numpy as np

# Project modules
from cleaning_and_clustering import run_cleaning_and_clustering, ScrubConfig  
from mapping import run_mapping, MappingConfig
from lineout import run_lineout, compute_peak_metrics, LineoutConfig


# Config for the run, most important is E_Step, tolerance, and frame index for lineout
CONFIG = {
    #  Input
    "INPUT_FILE": "sxro6416-r0504.h5",     # HDF5 CCD dataset to process

    #  Logging
    "LOG_LEVEL": "INFO",                   # Console log: "DEBUG" | "INFO" | "WARNING"

    #  Mapping (reference ridge extraction + conic fit)
    "MAP_FRAME_INDEX": 8,                  # Frame index used for mapping (geometry calibration)
    "MAP_ALPHA1_DEG": 90.0 - 39.632,       # Half-angle for the Lβ emission line (~1218 eV)
    "MAP_ALPHA2_DEG": 90.0 - 40.86,        # Half-angle for the Lα emission line (~1188 eV)

    # Lineout (energy sweep and integration)
    "E_MIN": 1100.0,                       # Minimum photon energy (eV)
    "E_MAX": 1604.0,                       # Maximum photon energy (eV, exclusive)
    "E_STEP": 0.1,                         # Energy step (eV). Use 0.1 for paper-accuracy; increase for faster demo
    "TOLERANCE_PX": 2,                     # Lateral half-width (pixels) around each iso-energy conic
    "LINEOUT_FRAME": 8,                    # Photon-map index to analyze for the final lineout

    #  Plotting
    "Y_SCALE": "log",                      # Y-axis scale for spectral plot: "linear" | "log"
    "SAVE_FIG_PATH": None,                 # Optional path to save figure (e.g., "lineout.svg"); None → no save

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
    scrub_cfg = ScrubConfig()
    cl_res = run_cleaning_and_clustering(stack, scrub=scrub_cfg)
   

    photon_maps, cluster_meta = cl_res.as_tuple()
    total_clusters = sum(len(d) for d in cluster_meta)
    log.info("Clustering complete: frames=%d, total_clusters=%d",
             len(photon_maps), total_clusters)

    # MAPPING (instrument calibration)
    # Fit geometry + mapping offsets from two reference ridge regions.
    map_cfg = MappingConfig(
        frame_index=cfg["MAP_FRAME_INDEX"],
        alpha1_deg=cfg["MAP_ALPHA1_DEG"],
        alpha2_deg=cfg["MAP_ALPHA2_DEG"],
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
        yscale=cfg["Y_SCALE"],
    )

    _lineout = run_lineout(
        photon_maps, d_opt, theta_z_opt, C1_opt, b_opt, shift_part_1, config=lcfg
    )


    energies = _lineout.energies
    intensity = _lineout.intensity
    metrics = compute_peak_metrics(
        energies, intensity,
        peak_window=(1180.0, 1196.0),
        mor_half_window=30,
        mor_smooth_hw=30,
        gauss_limit_fwhm=1.5
    )

    log.info(
        "Peak 1188 eV fit: mu=%.3f eV, sigma=%.3f eV, FWHM=%.3f eV, SNR=%.1f",
        metrics["mu"], metrics["sigma"], metrics["FWHM"], metrics["SNR"]
    )

    # Optional: save the figure if plotting is enabled and a path is provided.
    if cfg["SAVE_FIG_PATH"]:
        Path(cfg["SAVE_FIG_PATH"]).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cfg["SAVE_FIG_PATH"], dpi=300, bbox_inches="tight")
        log.info("Saved figure → %s", Path(cfg["SAVE_FIG_PATH"]).resolve())

    dt = time.perf_counter() - t0
    log.info("Pipeline finished in %.2fs", dt)


# Standard script entry point
if __name__ == "__main__":
    main()
    # Keep the plot window open when run outside an interactive environment
    input("Press Enter to close")
