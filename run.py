"""XSPEDS pipeline runner (load -> clean+cluster -> mapping -> lineout).

Converts raw CCD frames into a physically meaningful spectrum (counts per eV).

Pipeline stages:

1) Load:
   Read the CCD frame stack from HDF5 (dataset-specific layout). Each frame is a
   2048x2048 array; the first three columns are dropped to avoid spurious edge
   values seen in this dataset. The example file holds 20 frames.

2) Cleaning + Clustering (SPC):
   Fit Gaussian pedestals per row batch and derive dynamic thresholds (e.g. a
   threshold of 90 zeroes every pixel below 90). Then cluster the surviving
   pixels into photon hits, classifying the shapes described in the paper.
   Outputs per-frame photon maps.

3) Mapping (instrument calibration):
   Fit the cone-plane geometry and mapping offsets from the two reference
   ridges to get energy-dependent conic parameters in CCD coordinates.

4) Lineout (physics output):
   Sum along iso-energy conics and normalise by the local eV window width to
   get counts per eV, with Wiener smoothing for display and +-k sigma Poisson
   uncertainty bands. Finally fit the L-alpha peak for FWHM and SNR.
"""

from __future__ import annotations

import itertools
import logging
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from cleaning_and_clustering import ScrubConfig, run_cleaning_and_clustering
from lineout import LineoutConfig, compute_peak_metrics, run_lineout
from mapping import MappingConfig, run_mapping

# Config for the run; the most important knobs are E_STEP, TOLERANCE_PX,
# and the frame index for the lineout
CONFIG = {
    # Input
    "INPUT_FILE": "sxro6416-r0504.h5",     # HDF5 CCD dataset to process

    # Logging
    "LOG_LEVEL": "INFO",                   # "DEBUG", "INFO", "WARNING"

    # Mapping (reference ridge extraction + conic fit)
    "MAP_FRAME_INDEX": 8,                  # Frame used for geometry calibration
    "MAP_ALPHA1_DEG": 90.0 - 39.632,       # Half-angle for the L-beta line (~1218 eV)
    "MAP_ALPHA2_DEG": 90.0 - 40.86,        # Half-angle for the L-alpha line (~1188 eV)

    # Lineout (energy sweep and integration)
    "E_MIN": 1100.0,                       # Minimum photon energy (eV)
    "E_MAX": 1600.0,                       # Maximum photon energy (eV, exclusive)
    "E_STEP": 0.1,                         # Energy step (eV); 0.1 matches the paper, larger is faster
    "TOLERANCE_PX": 2,                     # Lateral half-width (pixels) around each conic
    "LINEOUT_FRAME": 8,                    # Photon map index for the final lineout

    # Plotting
    "Y_SCALE": "log",                      # "linear" or "log"
}


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging for the pipeline.

    Args:
        level: Logging level name, e.g. "INFO" or "DEBUG".
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def load_image_data(f_name: str) -> np.ndarray:
    """Load CCD frames from an HDF5 file and drop the first three columns.

    Assumes the Princeton FrameV2 layout used in this project. If adapting to a
    different source, adjust the HDF5 path below and revisit the column drop.

    Args:
        f_name: Path to the HDF5 file.

    Returns:
        Stack of frames, shape (N, H, W), with the first three columns removed.
    """
    path = Path(f_name)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    frames: list[np.ndarray] = []
    with h5py.File(str(path), "r") as f:
        for i in itertools.count():
            node = f.get(
                f"Configure:0000/Run:0000/CalibCycle:{i:04d}/"
                "Princeton::FrameV2/SxrEndstation.0:Princeton.0/data"
            )
            if node is None:
                break
            # first 3 columns hold dataset-specific spike/edge artefacts
            frames.append(node[0][:, 3:])

    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")
    return stack


def main() -> None:
    """Run the full pipeline and write the lineout CSV and figure to outputs/."""
    cfg = CONFIG
    setup_logging(cfg["LOG_LEVEL"])
    log = logging.getLogger("xspeds.run")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir.resolve())

    t0 = time.perf_counter()

    # LOAD
    stack = load_image_data(cfg["INPUT_FILE"])
    log.info("Loaded stack: shape=%s (frames, rows, cols)", stack.shape)

    # CLEANING + CLUSTERING (SPC)
    cl_res = run_cleaning_and_clustering(stack, scrub=ScrubConfig())
    photon_maps = cl_res.photon_maps
    total_clusters = sum(len(d) for d in cl_res.clusters)
    log.info("Clustering complete: frames=%d, total_clusters=%d", len(photon_maps), total_clusters)

    # MAPPING (instrument calibration)
    map_out = run_mapping(
        stack,
        config=MappingConfig(
            frame_index=cfg["MAP_FRAME_INDEX"],
            alpha1_deg=cfg["MAP_ALPHA1_DEG"],
            alpha2_deg=cfg["MAP_ALPHA2_DEG"],
        ),
    )
    log.info(
        "Mapping parameters: d=%.4g, theta_z=%.3f deg, C1=%.4g px, b=%.4g px, shift=%.4g px",
        map_out.d, float(np.rad2deg(map_out.theta_z)), map_out.C1, map_out.b, map_out.shift,
    )

    # SPECTRAL LINEOUT
    lineout = run_lineout(
        photon_maps,
        map_out.d, map_out.theta_z, map_out.C1, map_out.b, map_out.shift,
        config=LineoutConfig(
            energy_min=cfg["E_MIN"],
            energy_max=cfg["E_MAX"],
            energy_step=cfg["E_STEP"],
            tolerance=cfg["TOLERANCE_PX"],
            frame_index=cfg["LINEOUT_FRAME"],
            yscale=cfg["Y_SCALE"],
            save_fig_path=str(out_dir / "lineout.png"),
        ),
    )
    lineout.to_dataframe().to_csv(out_dir / "lineout.csv", index=False)

    # PEAK METRICS (L-alpha at ~1188 eV)
    metrics = compute_peak_metrics(
        lineout.energies,
        lineout.intensity,
        peak_window=(1180.0, 1196.0),
        mor_half_window=30,
        mor_smooth_hw=30,
        gauss_limit_fwhm=1.5,
    )
    log.info(
        "Peak 1188 eV fit: mu=%.3f eV, sigma=%.3f eV, FWHM=%.3f eV, SNR=%.1f",
        metrics["mu"], metrics["sigma"], metrics["FWHM"], metrics["SNR"],
    )

    log.info("Pipeline finished in %.2fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
