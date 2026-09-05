"""
Lineout stage: sum photon counts along iso-energy conics and normalise to counts per eV.

For each energy in the sweep, Bragg's law gives the cone half-angle alpha(E); the
cone-plane intersection is an ellipse or hyperbola (parabola exactly at the
transition) in CCD coordinates. Counts within a lateral tolerance of the conic
are summed and divided by the local eV window width, since the dispersion dE/dx
varies across the detector. Poisson uncertainties are propagated through the
normalisation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaselines import morphological
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
from scipy.signal import wiener

from geometry import conic_coefficients, rotated_basis

logger = logging.getLogger("xspeds.lineout")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


###################################
#      Configuration objects      #
###################################

@dataclass(frozen=True)
class LineoutConfig:
    """Configuration for the spectral lineout.

    Args:
        two_d_crystal: Crystal 2d spacing (Angstrom) for the Bragg energy-angle conversion.
        energy_min: Start of the energy sweep (eV).
        energy_max: End of the energy sweep (eV, exclusive).
        energy_step: Energy step (eV).
        tolerance: Lateral half-width (pixels) summed around each iso-energy conic.
        frame_index: Which photon map frame to analyse.
        plot: Whether to generate a Matplotlib figure.
        save_fig_path: Where to save the figure, None for no save.
        wiener_mysize: Neighbourhood length for Wiener smoothing, None to disable.
        error_band_k: Multiplier for the +-k sigma shaded band.
        yscale: Y-axis scale for the plot.
    """
    two_d_crystal: float = 15.96
    energy_min: float = 1100.0
    energy_max: float = 1600.0
    energy_step: float = 0.1
    tolerance: int = 2
    frame_index: int = 1

    plot: bool = True
    save_fig_path: str | None = None
    wiener_mysize: int | None = 30
    error_band_k: float = 2.0
    yscale: Literal["linear", "log"] = "linear"


@dataclass(frozen=True)
class LineoutResult:
    """Output of the spectral lineout.

    Args:
        energies: Energy grid (eV).
        intensity: Counts per eV, before any smoothing.
        raw_sums: Unnormalised counts summed along each conic.
        windows: eV window width used for normalisation at each energy.
        smoothed: Wiener-filtered intensity, None if smoothing disabled.
        sigma_intensity: Propagated Poisson sigma for counts per eV.
    """
    energies: NDArray[np.float64]
    intensity: NDArray[np.float64]
    raw_sums: NDArray[np.float64]
    windows: NDArray[np.float64]
    smoothed: NDArray[np.float64] | None
    sigma_intensity: NDArray[np.float64]

    def to_dataframe(self) -> pd.DataFrame:
        """Return the lineout as a tidy DataFrame, one row per energy bin."""
        return pd.DataFrame(
            {
                "energy_eV": self.energies,
                "intensity_counts_per_eV": self.intensity,
                "raw_sum": self.raw_sums,
                "window_eV": self.windows,
                "sigma_counts_per_eV": self.sigma_intensity,
                "smoothed_counts_per_eV": (
                    self.smoothed
                    if self.smoothed is not None
                    else np.full_like(self.intensity, np.nan)
                ),
            }
        )


##################################
#      Iso-energy conics         #
##################################

@dataclass(frozen=True)
class Conic:
    """One iso-energy conic in shifted (pixel) CCD coordinates.

    Args:
        kind: Conic type.
        vertex: Leftmost vertex (x, y), used for the dispersion dE/dx.
        center: Conic center (x, y), None for a parabola.
        a: Semi-major axis, or the parabola coefficient for kind "parabola".
        b: Semi-minor axis, unused for a parabola.
    """
    kind: Literal["ellipse", "hyperbola", "parabola"]
    vertex: NDArray[np.float64]
    center: NDArray[np.float64] | None
    a: float
    b: float


def isoenergy_conic(
    alpha: float,
    d: float,
    e_i: NDArray[np.float64],
    e_j: NDArray[np.float64],
    shift: NDArray[np.float64],
    *,
    tol: float = 1e-6,
) -> Conic:
    """Compute the cone-plane conic for one energy and translate it into pixel coordinates.

    Args:
        alpha: Cone half-angle (radians).
        d: Source-detector distance (pixels).
        e_i: First detector-plane basis vector.
        e_j: Second detector-plane basis vector.
        shift: (x, y) translation from raw geometry to pixel coordinates.
        tol: Tolerance for classifying the discriminant as parabolic.

    Returns:
        Conic parameters, already shifted.
    """
    A, B, C, D, E, F = conic_coefficients(alpha, d, e_i, e_j)
    disc = B**2 - 4 * A * C
    M = np.array([[2 * A, B], [B, 2 * C]], dtype=np.float64)

    # Parabola: degenerate M, get the vertex from least squares
    if np.isclose(disc, 0.0, atol=tol):
        vertex, *_ = np.linalg.lstsq(M, -np.array([D, E]), rcond=None)
        u0, v0 = vertex
        K = -(A * u0**2 + B * u0 * v0 + C * v0**2 + D * u0 + E * v0 + F)
        return Conic(kind="parabola", vertex=vertex + shift, center=None, a=A / K, b=0.0)

    center = np.linalg.solve(M, -np.array([D, E]))
    ei0, ej0 = center
    K = -(B * ei0 * ej0 + A * ei0**2 + C * ej0**2 + D * ei0 + E * ej0 + F)
    Q = np.array([[A / K, B / (2 * K)], [B / (2 * K), C / K]], dtype=np.float64)
    vals, vecs = np.linalg.eigh(Q)  # Q symmetric, eigh gives real pairs

    if disc < 0:
        # Ellipse: axes from the eigenvalues, leftmost vertex along the major axis
        lam1, lam2 = vals
        vec1 = vecs[:, 0]
        a_axis, b_axis = 1.0 / np.sqrt(lam1), 1.0 / np.sqrt(lam2)
        if b_axis > a_axis:
            a_axis, b_axis = b_axis, a_axis
        ang = float(np.arctan2(vec1[1], vec1[0]))

        cand1 = center + a_axis * np.array([np.cos(ang), np.sin(ang)])
        cand2 = center - a_axis * np.array([np.cos(ang), np.sin(ang)])
        vertex = cand1 if cand1[0] < cand2[0] else cand2
        return Conic(kind="ellipse", vertex=vertex + shift, center=center + shift,
                     a=float(a_axis), b=float(b_axis))

    # Hyperbola: one positive and one negative eigenvalue
    idx = int(np.argmax(vals))
    a_axis = 1.0 / np.sqrt(vals[idx])
    b_axis = 1.0 / np.sqrt(-float(np.min(vals)))
    vecp = vecs[:, idx]
    ang = float(np.arctan2(vecp[1], vecp[0]))
    vertex = center + a_axis * np.array([np.cos(ang), np.sin(ang)])
    return Conic(kind="hyperbola", vertex=vertex + shift, center=center + shift,
                 a=float(a_axis), b=float(b_axis))


###############################################################################
#            curve integration (box-summed grid + vector sampling)            #
###############################################################################

def _horizontal_boxsum(grid: NDArray[np.float64], tol: int) -> NDArray[np.float64]:
    """Precompute grid_box[y, x] = sum of grid[y, x-tol .. x+tol] with zero padding.

    Sampling this once per conic point is equivalent to summing a +-tol strip
    around the conic, without an inner loop.

    Args:
        grid: 2D photon map.
        tol: Lateral half-width in pixels.

    Returns:
        Box-summed grid of the same shape.
    """
    if tol <= 0:
        return grid
    kernel = np.ones(2 * tol + 1, dtype=grid.dtype)
    return convolve1d(grid, kernel, axis=1, mode="constant", cval=0.0)


def sum_ellipse(grid_box: NDArray[np.float64], conic: Conic) -> float:
    """Sum along the left branch of an ellipse, one sample per row.

    Args:
        grid_box: Box-summed photon map.
        conic: Ellipse parameters.

    Returns:
        Total counts along the branch.
    """
    H, W = grid_box.shape
    h, k = conic.center  # type: ignore[misc]

    y = np.arange(H, dtype=np.float64)
    u = (y - k) / conic.b
    m = np.abs(u) <= 1.0
    y_idx = np.nonzero(m)[0]
    if y_idx.size == 0:
        return 0.0

    x = h - conic.a * np.sqrt(np.maximum(0.0, 1.0 - u[m] ** 2))
    xi = np.clip(np.rint(x).astype(np.int64), 0, W - 1)
    return float(grid_box[y_idx, xi].sum())


def sum_hyperbola(grid_box: NDArray[np.float64], conic: Conic) -> float:
    """Sum along the positive branch of a hyperbola, one sample per row.

    Args:
        grid_box: Box-summed photon map.
        conic: Hyperbola parameters.

    Returns:
        Total counts along the branch.
    """
    H, W = grid_box.shape
    h, k = conic.center  # type: ignore[misc]

    y = np.arange(H, dtype=np.float64)
    ratio = (y - k) / conic.b
    x = h + conic.a * np.sqrt(1.0 + ratio * ratio)
    xi = np.clip(np.rint(x).astype(np.int64), 0, W - 1)
    return float(grid_box[np.arange(H), xi].sum())


def sum_parabola(grid_box: NDArray[np.float64], conic: Conic) -> float:
    """Sum samples along the parabola y = a(x - h)^2 + k across the full grid width.

    Only reached at the exact ellipse-hyperbola transition, kept for completeness.

    Args:
        grid_box: Box-summed photon map.
        conic: Parabola parameters (a is the quadratic coefficient).

    Returns:
        Total counts along the curve.
    """
    H, W = grid_box.shape
    h, k = conic.vertex

    xs = np.arange(W, dtype=np.float64)
    ys = conic.a * (xs - h) ** 2 + k
    iy = np.clip(np.rint(ys).astype(np.int64), 0, H - 1)
    return float(grid_box[iy, np.arange(W)].sum())


########################
#       Lineout        #
########################

def _alpha_from_energy(E: NDArray[np.float64], two_d_crystal: float) -> NDArray[np.float64]:
    """Vectorised Bragg conversion from energy to cone half-angle (theta = pi/2 - alpha).

    Args:
        E: Photon energies (eV).
        two_d_crystal: Crystal 2d spacing (Angstrom).

    Returns:
        Half-angles alpha (radians), clipped for numerical safety near cutoff.
    """
    arg = np.clip(12398.0 / (two_d_crystal * E), -1.0, 1.0)
    return np.arccos(arg)


def run_lineout(
    photon_map_all: Sequence[NDArray[np.int_]],
    d_opt: float,
    theta_z_opt: float,
    C1_opt: float,
    b_opt: float,
    shift_part_1: float,
    *,
    config: LineoutConfig | None = None,
) -> LineoutResult:
    """Compute the spectral lineout by summing along iso-energy conics and normalising.

    intensity(E) = raw_sum(E) / W(E), where the eV window width W(E) = |dE/dx|
    comes from the gradient of the conic vertex position x(E).

    Args:
        photon_map_all: Photon maps from the clustering stage, one per frame.
        d_opt: Fitted source-detector distance (pixels).
        theta_z_opt: Fitted CCD tilt (radians).
        C1_opt: Fitted x-vertex offset of ridge 1 (pixels).
        b_opt: Fitted shared y-vertex (pixels).
        shift_part_1: Raw-geometry vertex u-coordinate of cone 1.
        config: Lineout configuration, defaults to LineoutConfig().

    Returns:
        LineoutResult with energies, intensity, raw sums, windows, and uncertainties.
    """
    cfg = config or LineoutConfig()
    if cfg.frame_index >= len(photon_map_all):
        raise IndexError(
            f"frame_index {cfg.frame_index} out of range for {len(photon_map_all)} photon maps"
        )

    grid = np.asarray(photon_map_all[cfg.frame_index], dtype=np.float64)
    H, W = grid.shape
    logger.info(
        f"Lineout on frame={cfg.frame_index} | E=[{cfg.energy_min},{cfg.energy_max}) "
        f"step={cfg.energy_step} eV | tol={cfg.tolerance} | grid={H}x{W}"
    )

    energies = np.arange(cfg.energy_min, cfg.energy_max, cfg.energy_step, dtype=np.float64)
    if energies.size < 2:
        raise ValueError("Energy grid has < 2 points; widen the range or reduce energy_step.")
    alphas = _alpha_from_energy(energies, cfg.two_d_crystal)

    # Geometry basis and box-summed grid are the same for every energy
    e_i, e_j = rotated_basis(theta_z_opt)
    grid_box = _horizontal_boxsum(grid, cfg.tolerance)
    shift = np.array([C1_opt - shift_part_1, b_opt])

    raw_sums = np.empty_like(energies)
    vertex_x = np.empty_like(energies)
    summers = {"ellipse": sum_ellipse, "hyperbola": sum_hyperbola, "parabola": sum_parabola}

    for i, alpha in enumerate(alphas):
        conic = isoenergy_conic(alpha, d_opt, e_i, e_j, shift)
        vertex_x[i] = float(conic.vertex[0])
        raw_sums[i] = summers[conic.kind](grid_box, conic)

    # W(E) = |dE/dx| via the vertex trajectory; floor dx/dE where the vertex
    # barely moves with E so the division stays finite
    dx_dE = np.gradient(vertex_x, energies)
    dx_dE_floor = 1e-6  # pixels per eV
    dx_dE = np.where(np.abs(dx_dE) < dx_dE_floor, np.sign(dx_dE) * dx_dE_floor, dx_dE)
    windows = np.abs(1.0 / dx_dE)

    intensity = raw_sums / windows
    sigma_intensity = np.sqrt(np.maximum(raw_sums, 0.0)) / windows

    smoothed = None
    if cfg.wiener_mysize is not None:
        smoothed = wiener(intensity, mysize=int(max(3, cfg.wiener_mysize)))

    logger.info(
        f"Lineout complete: max(intensity)={float(np.max(intensity)):.4g}, "
        f"nonzero bins={int(np.count_nonzero(intensity))} / {intensity.size}"
    )

    if cfg.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(energies, intensity, "-", linewidth=1.0, label="Intensity (raw, counts/eV)")

        ref_for_band = smoothed if smoothed is not None else intensity
        k = float(cfg.error_band_k)
        plt.fill_between(
            energies,
            ref_for_band - k * sigma_intensity,
            ref_for_band + k * sigma_intensity,
            alpha=0.2,
            label=f"±{k:.0f}σ (Poisson)",
        )
        plt.xlabel("Energy (eV)")
        plt.ylabel("Counts per eV")
        plt.title("Spectral Lineout")
        plt.grid(True)
        if cfg.yscale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        if cfg.save_fig_path:
            p = Path(cfg.save_fig_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300, bbox_inches="tight")
            logger.info("Saved lineout figure %s", p.resolve())
        plt.show()

    return LineoutResult(
        energies=energies,
        intensity=intensity,
        raw_sums=raw_sums,
        windows=windows,
        smoothed=smoothed,
        sigma_intensity=sigma_intensity,
    )


#######################################
#        Further lineout analysis     #
#######################################

def compute_peak_metrics(
    energies: NDArray[np.float64],
    intensity: NDArray[np.float64],
    *,
    peak_window: tuple[float, float] = (1180.0, 1196.0),
    mor_half_window: int = 30,
    mor_smooth_hw: int = 30,
    gauss_limit_fwhm: float = 1.5,
) -> dict[str, object]:
    """Baseline-correct around a target peak, fit a Gaussian, and estimate SNR.

    Args:
        energies: Energy grid (eV).
        intensity: Counts per eV.
        peak_window: (lo, hi) energy window containing the peak to fit.
        mor_half_window: Half-window for the morphological baseline.
        mor_smooth_hw: Smoothing half-window for the baseline.
        gauss_limit_fwhm: Background region excludes +- this many FWHM around the peak.

    Returns:
        Dict with fit parameters (A, mu, sigma, FWHM, C), background_level,
        noise, peak_signal, SNR, and the baseline_corrected array.
    """
    x = np.asarray(energies, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)

    baseline, _ = morphological.mor(y, half_window=mor_half_window, smooth_half_window=mor_smooth_hw)
    y_corr = y - baseline

    lo, hi = peak_window
    m = (x > lo) & (x < hi)
    x_fit, y_fit = x[m], y_corr[m]
    if x_fit.size < 5:
        raise ValueError("Not enough points in peak_window to fit a Gaussian.")

    def gaussian(xv: NDArray[np.float64], A: float, mu: float, sigma: float, C: float) -> NDArray[np.float64]:
        return A * np.exp(-0.5 * ((xv - mu) / sigma) ** 2) + C

    init = [float(np.max(y_fit) - np.median(y_fit)), float(np.median(x_fit)), 2.0, float(np.median(y_fit))]
    popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=init, maxfev=10000)
    A, mu, sigma, C = map(float, popt)
    FWHM = 2.355 * sigma

    side = gauss_limit_fwhm * FWHM
    background_vals = y_corr[(x < (mu - side)) | (x > (mu + side))]
    background_level = float(np.median(background_vals))
    noise = float(np.std(background_vals))

    peak_signal = A + C
    SNR = (peak_signal - background_level) / noise if noise > 0 else np.inf

    return {
        "A": A,
        "mu": mu,
        "sigma": sigma,
        "FWHM": FWHM,
        "C": C,
        "background_level": background_level,
        "noise": noise,
        "peak_signal": peak_signal,
        "SNR": SNR,
        "baseline_corrected": y_corr,
    }
