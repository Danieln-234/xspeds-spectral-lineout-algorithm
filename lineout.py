


"""
Spectral lineout via iso-energy conic summation (ellipse/hyperbola/parabola).

Optimisations vs original:
  - Precompute rotated basis once (no per-energy rotations).
  - Precompute alpha(E) vectorised.
  - Precompute horizontal box-sums of the CCD image once per tolerance,
    then sample a single value per row/point instead of summing tiny slices.
  - Compute normalisation windows W(E) from d(vertex_x)/dE via np.gradient
    using stored vertex_x from the same conic eval used for summation
    (no second isoenergy_curves loop).
  - Use np.linalg.eigh for symmetric eigenproblems (faster/stabler).
  - Plotting bugfix: error band uses intensity when smoothed is None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pybaselines import morphological
from scipy.optimize import curve_fit
from scipy.signal import wiener
from scipy.ndimage import convolve1d

import matplotlib.pyplot as plt


logger = logging.getLogger("xspeds.lineout")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


###################################
#      Configuration objects      #
###################################

@dataclass(frozen=True)
class LineoutConfig:
    """
    Configuration for spectral lineout.

    Core physics/grid:
        two_d_crystal  : 2d (Å) used in energy ↔ angle conversion (Bragg law).
        energy_min/max/step : Energy sweep (eV). 'max' is exclusive.
        tolerance      : Lateral half-width (pixels) around each iso-energy conic.
        frame_index    : Which photon_map frame to use (0/1/2...).
        theta_x,theta_y: CCD rotations around x/y (radians) if applicable.
        num_points_parabola : Samples along parabola for integration.
        x_min/x_max    : x-range for parabolic summation (pixels); None→grid max.
        hyperbola_branch: Which hyperbola branch to integrate ("positive"/"negative").

    Plotting & smoothing:
        plot           : Whether to generate a Matplotlib plot.
        wiener_mysize  : Neighborhood length for scipy.signal.wiener (int).
        error_band_k   : Multiplier for ±k·σ shading (e.g., 2 for ±2σ).
        yscale         : "linear" or "log".
    """
    # physics/grid
    two_d_crystal: float = 15.96
    energy_min: float = 1100.0
    energy_max: float = 1600.0
    energy_step: float = 0.1
    tolerance: int = 2
    frame_index: int = 1
    theta_x: float = 0.0
    theta_y: float = 0.0
    num_points_parabola: int = 3000
    x_min: int = 0
    x_max: int | None = None
    hyperbola_branch: Literal["positive", "negative"] = "positive"

    # plotting/smoothing
    plot: bool = True
    save_fig_path: str | None = None
    wiener_mysize: int | None = 30
    error_band_k: float = 2.0
    yscale: Literal["linear", "log"] = "linear"


@dataclass(frozen=True)
class LineoutResult:
    """
    Output of spectral lineout.

    energies : (N,) eV
    intensity: (N,) counts/eV (raw; i.e., before any optional smoothing)
    raw_sums : (N,) unnormalised counts (sum along conics)
    windows  : (N,) eV per window used for normalisation
    smoothed : (N,) optional Wiener-filtered intensity (None if disabled)
    sigma_intensity : (N,) propagated Poisson σ for counts/eV (√N / W)
    """
    energies: NDArray[np.float64]
    intensity: NDArray[np.float64]
    raw_sums: NDArray[np.float64]
    windows: NDArray[np.float64]
    smoothed: NDArray[np.float64] | None
    sigma_intensity: NDArray[np.float64]

    def to_dataframe(self) -> pd.DataFrame:
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

    def as_tuple(self):
        return (
            self.energies,
            self.intensity,
            self.raw_sums,
            self.windows,
            self.smoothed,
            self.sigma_intensity,
        )


##################################
#      Geometry helpers          #
##################################

def _rotated_basis(
    theta_z: float,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Basis vectors of the CCD plane (e_i, e_j) after rotations about z, y, x.
    """
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

    R = Rx @ Ry @ Rz
    e_i0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e_j0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (R @ e_i0), (R @ e_j0)


def _conic_with_shift(
    alpha: float,
    d: float,
    e_i: NDArray[np.float64],
    e_j: NDArray[np.float64],
    *,
    shift: Tuple[float, float] = (0.0, 0.0),
    tol: float = 1e-6,
) -> Dict[str, object]:
    """
    Compute ellipse/hyperbola/parabola parameters for the cone–plane intersection,
    then translate (x,y) by mapping correction shift.
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

    coeffs = dict(A_coef=A, B_coef=B, C_coef=C, D_coef=D, E_coef=E, F_coef=F)
    disc = B**2 - 4 * A * C
    sx, sy = shift

    # Parabola
    if np.isclose(disc, 0.0, atol=tol):
        M = np.array([[2 * A, B], [B, 2 * C]], dtype=np.float64)
        vertex, *_ = np.linalg.lstsq(M, -np.array([D, E], dtype=np.float64), rcond=None)
        u0, v0 = vertex
        F0 = A * u0**2 + B * u0 * v0 + C * v0**2 + D * u0 + E * v0 + F
        K = -F0
        A_norm = A / K
        p = float(1.0 / (4.0 * A_norm))  # focal length
        return dict(
            type="parabola",
            vertex=vertex,
            focal_length=p,
            coeffs=coeffs,
            discriminant=float(disc),
            vertex_shifted=vertex + np.array([sx, sy], dtype=np.float64),
        )

    M = np.array([[2 * A, B], [B, 2 * C]], dtype=np.float64)
    center = np.linalg.solve(M, -np.array([D, E], dtype=np.float64))
    ei0, ej0 = center
    F0 = B * ei0 * ej0 + A * ei0**2 + C * ej0**2 + D * ei0 + E * ej0 + F
    K = -F0
    Q = np.array([[A / K, B / (2 * K)], [B / (2 * K), C / K]], dtype=np.float64)

    # Ellipse
    if disc < 0:
        # Q is symmetric → use eigh (faster/stable, real eigenpairs)
        vals, vecs = np.linalg.eigh(Q)
        order = np.argsort(vals)
        lam1, lam2 = vals[order[0]], vals[order[1]]
        vec1 = vecs[:, order[0]]

        a_axis, b_axis = 1.0 / np.sqrt(lam1), 1.0 / np.sqrt(lam2)
        if b_axis > a_axis:
            a_axis, b_axis = b_axis, a_axis

        ecc = np.sqrt(max(0.0, 1.0 - (b_axis**2 / a_axis**2)))
        cval = a_axis * ecc
        ang = float(np.arctan2(vec1[1], vec1[0]))

        cand1 = np.array((ei0 + a_axis * np.cos(ang), ej0 + a_axis * np.sin(ang)), dtype=np.float64)
        cand2 = np.array((ei0 - a_axis * np.cos(ang), ej0 - a_axis * np.sin(ang)), dtype=np.float64)
        vertex1 = cand1 if cand1[0] < cand2[0] else cand2

        focus1 = (ei0 + cval * np.cos(ang), ej0 + cval * np.sin(ang))
        focus2 = (ei0 - cval * np.cos(ang), ej0 - cval * np.sin(ang))
        fl = a_axis - cval

        return dict(
            type="ellipse",
            center=center,
            semi_axes=(float(a_axis), float(b_axis)),
            angle=ang,
            eccentricity=float(ecc),
            foci=(focus1, focus2),
            vertex=vertex1,
            focal_length=float(fl),
            coeffs=coeffs,
            discriminant=float(disc),
            center_shifted=center + np.array([sx, sy], dtype=np.float64),
            vertex_shifted=vertex1 + np.array([sx, sy], dtype=np.float64),
            foci_shifted=(
                np.array(focus1, dtype=np.float64) + np.array([sx, sy], dtype=np.float64),
                np.array(focus2, dtype=np.float64) + np.array([sx, sy], dtype=np.float64),
            ),
        )

    # Hyperbola
    vals, vecs = np.linalg.eigh(Q)
    # One eigenvalue should be positive and one negative (after normalisation)
    idx = int(np.argmax(vals))
    lam_p = vals[idx]
    lam_n = float(np.min(vals))
    vecp = vecs[:, idx]

    a_axis = 1.0 / np.sqrt(lam_p)
    b_axis = 1.0 / np.sqrt(-lam_n)
    ang = float(np.arctan2(vecp[1], vecp[0]))
    cval = np.sqrt(a_axis**2 + b_axis**2)

    vertex1 = (ei0 + a_axis * np.cos(ang), ej0 + a_axis * np.sin(ang))
    fl = cval - a_axis
    focus1 = (ei0 + cval * np.cos(ang), ej0 + cval * np.sin(ang))
    focus2 = (ei0 - cval * np.cos(ang), ej0 - cval * np.sin(ang))

    return dict(
        type="hyperbola",
        center=center,
        semi_axes=(float(a_axis), float(b_axis)),
        angle=ang,
        foci=(focus1, focus2),
        vertex=vertex1,
        focal_length=float(fl),
        coeffs=coeffs,
        discriminant=float(disc),
        center_shifted=center + np.array([sx, sy], dtype=np.float64),
        vertex_shifted=np.array(vertex1, dtype=np.float64) + np.array([sx, sy], dtype=np.float64),
        foci_shifted=(
            np.array(focus1, dtype=np.float64) + np.array([sx, sy], dtype=np.float64),
            np.array(focus2, dtype=np.float64) + np.array([sx, sy], dtype=np.float64),
        ),
    )


def isoenergy_curves_fast(
    alpha_rad: float,
    d: float,
    e_i: NDArray[np.float64],
    e_j: NDArray[np.float64],
    C1_opt: float,
    b_opt: float,
    shift_part_1: float,
) -> Dict[str, object]:
    """
    Wrapper to compute conic parameters for one energy (half-angle α),
    with precomputed (e_i, e_j).
    """
    shift = (-shift_part_1 + C1_opt, b_opt)
    return _conic_with_shift(alpha_rad, d, e_i, e_j, shift=shift)


###############################################################################
#            curve-integration (box-summed grid + vector sampling)            #
###############################################################################

def _horizontal_boxsum(grid: NDArray[np.float64], tol: int) -> NDArray[np.float64]:
    """
    grid_box[y, x] = sum_{k=-tol..tol} grid[y, x+k] with constant(0) padding.
    """
    if tol <= 0:
        return grid
    kernel = np.ones(2 * tol + 1, dtype=grid.dtype)
    return convolve1d(grid, kernel, axis=1, mode="constant", cval=0.0)


def sum_ellipse_rowwise_box(
    grid_box: NDArray[np.float64],
    center: Tuple[float, float],
    a: float,
    b: float,
) -> float:
    """
    Sum along the left branch of an ellipse, row-wise, sampling from box-summed grid.
    """
    H, W = grid_box.shape
    h, k = center

    y = np.arange(H, dtype=np.float64)
    u = (y - k) / b
    m = np.abs(u) <= 1.0

    y_idx = np.nonzero(m)[0]
    if y_idx.size == 0:
        return 0.0

    uu = u[m]
    x = h - a * np.sqrt(np.maximum(0.0, 1.0 - uu * uu))
    xi = np.clip(np.rint(x).astype(np.int64), 0, W - 1)

    return float(grid_box[y_idx, xi].sum())


def sum_hyperbola_rowwise_box(
    grid_box: NDArray[np.float64],
    center: Tuple[float, float],
    a: float,
    b: float,
    branch: Literal["positive", "negative"] = "positive",
) -> float:
    """
    Sum along one branch of a hyperbola, row-wise, sampling from box-summed grid.
    """
    H, W = grid_box.shape
    h, k = center

    y = np.arange(H, dtype=np.float64)
    ratio = (y - k) / b
    root = np.sqrt(1.0 + ratio * ratio)
    x = h + a * root if branch == "positive" else h - a * root
    xi = np.clip(np.rint(x).astype(np.int64), 0, W - 1)

    return float(grid_box[np.arange(H), xi].sum())


def sum_parabola_box(
    grid_box: NDArray[np.float64],
    vertex: Tuple[float, float],
    a_coeff: float,
    *,
    x_min: int,
    x_max: int,
    num_points: int = 1000,
) -> float:
    """
    Sum samples along parabola x = a(y - k)^2 + h, but implemented as y(x) sampling:
      y = a*(x - h)^2 + k
    """
    H, W = grid_box.shape
    h, k = vertex

    x_max = min(int(x_max), W - 1)
    x_min = max(int(x_min), 0)
    if x_max < x_min:
        return 0.0

    xs = np.linspace(x_min, x_max, int(num_points), dtype=np.float64)
    ys = a_coeff * (xs - h) ** 2 + k

    ix = np.clip(np.rint(xs).astype(np.int64), 0, W - 1)
    iy = np.clip(np.rint(ys).astype(np.int64), 0, H - 1)

    return float(grid_box[iy, ix].sum())


########################
#       Lineout        #
########################

def _alpha_from_energy_vec(E: NDArray[np.float64], two_d_crystal: float) -> NDArray[np.float64]:
    """
    Vectorised Bragg: alpha(E) from 2d and E (θ = π/2 − α).
    Uses clipping for numerical safety near the cutoff.
    """
    arg = 12398.0 / (two_d_crystal * E)
    arg = np.clip(arg, -1.0, 1.0)
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
    """
    Compute spectral lineout by summing along iso-energy conics and normalising.

    Normalisation:
        intensity(E) = raw_sums(E) / W(E)

    Here W(E) is computed via the stored vertex x(E):
        W(E) = abs(dE/dx) = abs(1 / (dx/dE))   using np.gradient.
    """
    cfg = config or LineoutConfig()
    if cfg.frame_index >= len(photon_map_all):
        raise IndexError(
            f"frame_index {cfg.frame_index} out of range for photon_map_all length {len(photon_map_all)}"
        )

    grid = np.asarray(photon_map_all[cfg.frame_index], dtype=np.float64)
    H, W = grid.shape

    logger.info(
        f"Lineout on frame={cfg.frame_index} | E=[{cfg.energy_min},{cfg.energy_max}) step={cfg.energy_step} eV | "
        f"tol={cfg.tolerance} | grid={H}x{W}"
    )

    # Energy grid + alpha(E)
    energies = np.arange(cfg.energy_min, cfg.energy_max, cfg.energy_step, dtype=np.float64)
    if energies.size < 2:
        raise ValueError("Energy grid has < 2 points; increase range or adjust energy_step.")

    alphas = _alpha_from_energy_vec(energies, cfg.two_d_crystal)

    # Precompute geometry basis once
    e_i, e_j = _rotated_basis(theta_z_opt, theta_x=cfg.theta_x, theta_y=cfg.theta_y)

    # Precompute horizontal boxsum once
    grid_box = _horizontal_boxsum(grid, cfg.tolerance)

    # Prepare output buffers
    raw_sums = np.empty_like(energies)
    vertex_x = np.empty_like(energies)

    xmax = (W - 1) if cfg.x_max is None else min(int(cfg.x_max), W - 1)
    xmin = max(int(cfg.x_min), 0)

    # Main loop: compute conic once per energy; store both raw_sum and vertex_x
    for i, alpha in enumerate(alphas):
        res = isoenergy_curves_fast(alpha, d_opt, e_i, e_j, C1_opt, b_opt, shift_part_1)
        conic_type = str(res.get("type", "")).lower()

        v = np.asarray(res.get("vertex_shifted", res.get("vertex")), dtype=np.float64)
        vertex_x[i] = float(v[0])

        if conic_type == "ellipse":
            center = tuple(np.asarray(res["center_shifted"], dtype=np.float64))  # type: ignore[index]
            a_axis, b_axis = res["semi_axes"]  # type: ignore[index]
            raw_sums[i] = sum_ellipse_rowwise_box(grid_box, center, float(a_axis), float(b_axis))

        elif conic_type == "hyperbola":
            center = tuple(np.asarray(res["center_shifted"], dtype=np.float64))  # type: ignore[index]
            a_axis, b_axis = res["semi_axes"]  # type: ignore[index]
            raw_sums[i] = sum_hyperbola_rowwise_box(
                grid_box, center, float(a_axis), float(b_axis), branch=cfg.hyperbola_branch
            )

        else:
            # Parabola (default)
            vertex = (float(v[0]), float(v[1]))
            p = float(res["focal_length"])  # type: ignore[index]
            a_coeff = 1.0 / (4.0 * p)
            raw_sums[i] = sum_parabola_box(
                grid_box,
                vertex,
                a_coeff,
                x_min=xmin,
                x_max=xmax,
                num_points=cfg.num_points_parabola,
            )

    # Compute W(E) = abs(dE/dx) via dx/dE gradient
    dx_dE = np.gradient(vertex_x, energies)

    
    # Here dx_dE has units pixels per eV; small dx_dE means almost no motion in x with E.
    dx_dE_floor = 1e-6  # pixels/eV 
    dx_dE_safe = np.where(np.abs(dx_dE) < dx_dE_floor, np.sign(dx_dE) * dx_dE_floor, dx_dE)

    windows = np.abs(1.0 / dx_dE_safe)  # eV per pixel-window "width" scaling
    windows_safe = np.maximum(windows, 1e-12)

    # Normalise + Poisson propagation
    intensity = raw_sums / windows_safe
    sigma_intensity = np.sqrt(np.maximum(raw_sums, 0.0)) / windows_safe

    # Optional Wiener smoothing
    smoothed = None
    if cfg.wiener_mysize is not None:
        ms = int(max(3, cfg.wiener_mysize))
        smoothed = wiener(intensity, mysize=ms)

    logger.info(
        "Lineout complete: "
        f"max(intensity)={float(np.max(intensity)):.4g}, "
        f"nonzero bins={(int(np.count_nonzero(intensity)))} / {intensity.size}"
    )

    # Plot
    if cfg.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(energies, intensity, "-", linewidth=1.0, label="Intensity (raw, counts/eV)")

        ref_for_band = smoothed if smoothed is not None else intensity
        k = float(cfg.error_band_k)
        upper = ref_for_band + k * sigma_intensity
        lower = ref_for_band - k * sigma_intensity

        plt.fill_between(energies, lower, upper, alpha=0.2, label=f"±{k:.0f}σ (Poisson)")
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
) -> dict:
    """
    Baseline-correct around a target peak, fit a Gaussian, and estimate SNR.

    Returns dict:
      A, mu, sigma, FWHM, C,
      background_level, noise, peak_signal, SNR,
      baseline_corrected
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

    def gaussian(xv, A, mu, sigma, C):
        return A * np.exp(-0.5 * ((xv - mu) / sigma) ** 2) + C

    init = [float(np.max(y_fit) - np.median(y_fit)), float(np.median(x_fit)), 2.0, float(np.median(y_fit))]
    popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=init, maxfev=10000)
    A, mu, sigma, C = map(float, popt)
    FWHM = 2.355 * sigma

    side = gauss_limit_fwhm * FWHM
    bg_mask = (x < (mu - side)) | (x > (mu + side))
    background_vals = y_corr[bg_mask]
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
