"""
Spectral lineout via iso-energy conic summation (ellipse/hyperbola/parabola).

This module:
  1) Sums photon-map counts along iso-energy conics (derived from fitted geometry).
  2) Normalises by the local energy-window width W(E) to yield counts per eV
     (correcting for non-uniform dispersion across the CCD).
  3) Optionally applies Wiener filtering to the intensity trace and plots ±k·σ
     Poisson uncertainty bands as a shaded region (σ propagated through normalisation).

Physics + method follow the XSPEDS report (Sections 2.3–2.4, 3.4) where window
normalisation and Wiener-smoothed error bands are described.  ⟨Ref: XSPEDS report⟩
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from scipy.signal import wiener  # Wiener filter for denoising the 1D spectrum


# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

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
        plot_mode      : "raw" | "smoothed" | "both".
        wiener_mysize  : Neighborhood length for scipy.signal.wiener (int).  None→off.
                         (A value comparable to the FWHM in bins is typical.)
        error_band_k   : Multiplier for ±k·σ shading (e.g., 2 for ±2σ).
        yscale         : "linear" or "log".
    """
    #  physics/grid 
    two_d_crystal: float = 15.96
    energy_min: float = 1100.0
    energy_max: float = 1604.0
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
    plot_mode: Literal["raw", "smoothed", "both"] = "smoothed"
    wiener_mysize: int | None = 30  # ~neighborhood length in bins; tune to FWHM (see paper)
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
                "smoothed_counts_per_eV": (self.smoothed if self.smoothed is not None else np.full_like(self.intensity, np.nan)),
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

def _rotated_basis(theta_z: float, theta_x: float = 0.0, theta_y: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Basis vectors of the CCD plane (e_i, e_j) after rotations about z, y, x.
    These are used to express the cone–plane intersection (the conic) in CCD coords.
    """
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    R = Rx @ Ry @ Rz
    e_i0 = np.array([0.0, 1.0, 0.0])
    e_j0 = np.array([0.0, 0.0, 1.0])
    return (R @ e_i0).astype(np.float64), (R @ e_j0).astype(np.float64)


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
    then translate (x,y) by a small mapping correction 'shift'.

    Discriminant test decides the conic type. Eigen-analysis yields axes/angle.
    See Appendix B of the report for derivation details.  ⟨Ref: XSPEDS report⟩
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
        M = np.array([[2 * A, B], [B, 2 * C]])
        vertex, *_ = np.linalg.lstsq(M, -np.array([D, E]), rcond=None)
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
            vertex_shifted=vertex + np.array([sx, sy]),
        )

    # Ellipse
    if disc < 0:
        M = np.array([[2 * A, B], [B, 2 * C]])
        center = np.linalg.solve(M, -np.array([D, E]))
        ei0, ej0 = center
        F0 = B * ei0 * ej0 + A * ei0**2 + C * ej0**2 + D * ei0 + E * ej0 + F
        K = -F0
        Q = np.array([[A / K, B / (2 * K)], [B / (2 * K), C / K]])
        vals, vecs = np.linalg.eig(Q)
        order = np.argsort(vals)
        lam1, lam2 = vals[order[0]], vals[order[1]]
        vec1 = vecs[:, order[0]]
        a_axis, b_axis = 1.0 / np.sqrt(lam1), 1.0 / np.sqrt(lam2)
        if b_axis > a_axis:  # ensure a>=b
            a_axis, b_axis = b_axis, a_axis
        ecc = np.sqrt(1.0 - (b_axis**2 / a_axis**2))
        cval = a_axis * ecc
        ang = float(np.arctan2(vec1[1], vec1[0]))

        cand1 = np.array((ei0 + a_axis * np.cos(ang), ej0 + a_axis * np.sin(ang)))
        cand2 = np.array((ei0 - a_axis * np.cos(ang), ej0 - a_axis * np.sin(ang)))
        vertex1 = cand1 if cand1[0] < cand2[0] else cand2

        focus1 = (ei0 + cval * np.cos(ang), ej0 + cval * np.sin(ang))
        focus2 = (ei0 - cval * np.cos(ang), ej0 - cval * np.sin(ang))
        fl = a_axis - cval  # focal length (parabola-limit analogue)

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
            center_shifted=center + np.array([sx, sy]),
            vertex_shifted=vertex1 + np.array([sx, sy]),
            foci_shifted=(
                np.array(focus1) + np.array([sx, sy]),
                np.array(focus2) + np.array([sx, sy]),
            ),
        )

    # Hyperbola
    M = np.array([[2 * A, B], [B, 2 * C]])
    center = np.linalg.solve(M, -np.array([D, E]))
    ei0, ej0 = center
    F0 = B * ei0 * ej0 + A * ei0**2 + C * ej0**2 + D * ei0 + E * ej0 + F
    K = -F0
    Q = np.array([[A / K, B / (2 * K)], [B / (2 * K), C / K]])
    vals, vecs = np.linalg.eig(Q)
    idx = int(np.argmax(vals))
    lam_p = vals[idx]
    lam_n = np.min(vals)
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
        center_shifted=center + np.array([sx, sy]),
        vertex_shifted=np.array(vertex1) + np.array([sx, sy]),
        foci_shifted=(
            np.array(focus1) + np.array([sx, sy]),
            np.array(focus2) + np.array([sx, sy]),
        ),
    )


def isoenergy_curves(
    alpha_rad: float,
    d: float,
    theta_z: float,
    theta_y: float,
    theta_x: float,
    C1_opt: float,
    b_opt: float,
    shift_part_1: float,
) -> Dict[str, object]:
    """
    Wrapper to compute conic parameters for one energy (half-angle α).
    The 'shift' recentres the conic in CCD coords using mapping offsets.
    """
    e_i, e_j = _rotated_basis(theta_z, theta_x=theta_x, theta_y=theta_y)
    shift = (-shift_part_1 + C1_opt, b_opt)
    return _conic_with_shift(alpha_rad, d, e_i, e_j, shift=shift)

###############################################################################
#       Curve-integration (row-wise sampling around each conic)               #
###############################################################################

def sum_along_ellipse_by_row(
    grid: NDArray[np.float64],
    center: Tuple[float, float],
    a: float,
    b: float,
    *,
    tolerance: int = 3,
) -> float:
    """
    Sum pixel values within ±tolerance columns of the left branch of an ellipse,
    marching across rows (v-direction). This mirrors the CCD readout geometry.
    """
    H, W = grid.shape
    h, k = center
    total = 0.0
    for y in range(H):
        u = (y - k) / b
        if abs(u) > 1.0:
            continue
        x_float = h - a * np.sqrt(max(0.0, 1.0 - u * u))
        xi = int(round(x_float))
        x_start = max(0, xi - tolerance)
        x_end = min(W, xi + tolerance + 1)
        total += grid[y, x_start:x_end].sum()
    return float(total)


def sum_along_hyperbola_by_row(
    grid: NDArray[np.float64],
    center: Tuple[float, float],
    a: float,
    b: float,
    *,
    tolerance: int = 2,
    branch: Literal["positive", "negative"] = "positive",
) -> float:
    """
    Sum pixel values within ±tolerance columns of one hyperbola branch, row-wise.
    """
    H, W = grid.shape
    h, k = center
    total = 0.0
    for y in range(H):
        ratio = (y - k) / b
        root = np.sqrt(1.0 + ratio * ratio)
        x_float = h + a * root if branch == "positive" else h - a * root
        xi = int(round(x_float))
        x_start = max(0, xi - tolerance)
        x_end = min(W, xi + tolerance + 1)
        total += grid[y, x_start:x_end].sum()
    return float(total)


def sum_along_parabola_with_tolerance(
    grid: NDArray[np.float64],
    vertex: Tuple[float, float],
    a_coeff: float,
    *,
    x_min: int,
    x_max: int,
    tolerance: int = 0,
    num_points: int = 1000,
) -> float:
    """
    Sum pixel values within ±tolerance columns of a parabola x = a(y - k)^2 + h.
    """
    H, W = grid.shape
    h, k = vertex
    x_max = min(x_max, W - 1)
    xs = np.linspace(x_min, x_max, num_points)
    ys = a_coeff * (xs - h) ** 2 + k
    ix = np.clip(np.round(xs).astype(int), 0, W - 1)
    iy = np.clip(np.round(ys).astype(int), 0, H - 1)
    total = 0.0
    for xi, yi in zip(ix, iy):
        x_start = max(0, xi - tolerance)
        x_end = min(W, xi + tolerance + 1)
        total += grid[yi, x_start:x_end].sum()
    return float(total)


def sum_along_conic(
    grid: NDArray[np.float64],
    alpha_rad: float,
    d: float,
    theta_z: float,
    theta_y: float,
    theta_x: float,
    C1_opt: float,
    b_opt: float,
    shift_part_1: float,
    *,
    tolerance: int = 2,
    num_points: int = 3000,
    x_min: int = 0,
    x_max: int | None = None,
    hyperbola_branch: Literal["positive", "negative"] = "positive",
) -> float:
    """
    Sum pixel values near the iso-energy conic for half-angle α, using the appropriate
    curve routine. Parabolic case uses dense sampling in x for stability.
    """
    result = isoenergy_curves(alpha_rad, d, theta_z, theta_y, theta_x, C1_opt, b_opt, shift_part_1)
    conic_type = str(result.get("type", "")).lower()

    if conic_type == "ellipse":
        center = tuple(np.asarray(result["center_shifted"], dtype=float))  # type: ignore[index]
        a_axis, b_axis = result["semi_axes"]  # type: ignore[index]
        return sum_along_ellipse_by_row(grid, center, float(a_axis), float(b_axis), tolerance=tolerance)

    if conic_type == "hyperbola":
        center = tuple(np.asarray(result["center_shifted"], dtype=float))  # type: ignore[index]
        a_axis, b_axis = result["semi_axes"]  # type: ignore[index]
        return sum_along_hyperbola_by_row(
            grid, center, float(a_axis), float(b_axis), tolerance=tolerance, branch=hyperbola_branch
        )

    # Parabola (default)
    vertex = tuple(np.asarray(result.get("vertex_shifted", result.get("vertex")), dtype=float))  # type: ignore[arg-type]
    p = float(result["focal_length"])  # type: ignore[index]
    a_coeff = 1.0 / (4.0 * p)
    H, W = grid.shape
    xmax = (W - 1) if x_max is None else x_max
    return sum_along_parabola_with_tolerance(
        grid, vertex, a_coeff, x_min=x_min, x_max=xmax, tolerance=tolerance, num_points=num_points
    )

########################
#       Lineout        #
########################


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

    where W(E) is the energy width associated with the lateral pixel window around
    the iso-energy conic (window width scales with local dispersion). This yields a
    differential spectrum (counts per eV), as described in the report.  ⟨Ref: XSPEDS report⟩
    """
    cfg = config or LineoutConfig()
    if cfg.frame_index >= len(photon_map_all):
        raise IndexError(f"frame_index {cfg.frame_index} out of range for photon_map_all length {len(photon_map_all)}")

    grid = np.asarray(photon_map_all[cfg.frame_index], dtype=np.float64)
    H, W = grid.shape
    if cfg.x_max is not None and cfg.x_max >= W:
        logger.info(f"Clamping x_max={cfg.x_max} to grid width {W-1}.")

    logger.info(
        f"Lineout on frame={cfg.frame_index} | E=[{cfg.energy_min},{cfg.energy_max}) step={cfg.energy_step} eV | "
        f"tol={cfg.tolerance} | grid={H}x{W}"
    )

    # Energy grid
    energies = np.arange(cfg.energy_min, cfg.energy_max, cfg.energy_step, dtype=np.float64)

    # Bragg: alpha(E) from 2d and E (θ = π/2 − α); see report Sec. 2.3.
    def alpha_from_energy(E: float) -> float:
        return float(np.arccos(12398.0 / (cfg.two_d_crystal * E)))

    # Raw sums along the conics
    raw_sums = np.empty_like(energies)
    for idx, E in enumerate(energies):
        alpha_rad = alpha_from_energy(float(E))
        raw_sums[idx] = sum_along_conic(
            grid,
            alpha_rad,
            d_opt,
            theta_z_opt,
            cfg.theta_y,
            cfg.theta_x,
            C1_opt,
            b_opt,
            shift_part_1,
            tolerance=cfg.tolerance,
            num_points=cfg.num_points_parabola,
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            hyperbola_branch=cfg.hyperbola_branch,
        )

    # Local eV window W(E) via vertex spacing u(E+ΔE) - u(E); see Eq. (3) in report.
    windows = np.empty_like(energies)
    last_dx = None
    dx_min = 0.1  # pixels; safeguard for near-constant vertex spacing (parabolic case)
    for idx, E in enumerate(energies):
        a0 = alpha_from_energy(float(E))
        a1 = alpha_from_energy(float(E + cfg.energy_step))
        res0 = isoenergy_curves(a0, d_opt, theta_z_opt, cfg.theta_y, cfg.theta_x, C1_opt, b_opt, shift_part_1)
        res1 = isoenergy_curves(a1, d_opt, theta_z_opt, cfg.theta_y, cfg.theta_x, C1_opt, b_opt, shift_part_1)
        x0 = float(np.asarray(res0.get("vertex_shifted", res0.get("vertex")))[0])
        x1 = float(np.asarray(res1.get("vertex_shifted", res1.get("vertex")))[0])
        dx = x1 - x0
        if abs(dx) < dx_min:
            dx = dx_min if last_dx is None else last_dx
        else:
            last_dx = dx
        windows[idx] = abs(cfg.energy_step / dx)

    # Normalise to counts per eV
    # (Use small floor to avoid division blow-ups in completely empty regions.)
    windows_safe = np.maximum(windows, 1e-12)
    intensity = raw_sums / windows_safe

    # Propagate Poisson shot noise: σ_counts = √N  →  σ_intensity = √N / W
    sigma_intensity = np.sqrt(np.maximum(raw_sums, 0.0)) / windows_safe

    # Optional Wiener smoothing of the intensity for display (not used to compute σ)
    smoothed = None
    if cfg.wiener_mysize and wiener is not None:
        # SciPy Wiener operates on the 1D array with a given neighborhood length.
        # Choose mysize ≈ FWHM (in bins) for gentle denoising without peak bias (report Sec. 2.4).
        ms = int(max(3, cfg.wiener_mysize))
        smoothed = wiener(intensity, mysize=ms)

    logger.info(
        "Lineout complete: "
        f"max(intensity)={float(np.max(intensity)):.4g}, "
        f"nonzero bins={(int(np.count_nonzero(intensity)))} / {intensity.size}"
    )

    # Plot
    if cfg.plot and plt is not None:
        plt.figure(figsize=(8, 5))
        # Decide which traces to draw
        if cfg.plot_mode in ("raw", "both"):
            plt.plot(energies, intensity, "-", linewidth=1, label="Intensity (counts/eV)")
        if cfg.plot_mode in ("smoothed", "both") and smoothed is not None:
            plt.plot(energies, smoothed, "-", linewidth=1, label="Spectral Linout")

        # Error band around the *smoothed* curve if present, else the raw curve
        ref = smoothed if (smoothed is not None and cfg.plot_mode != "raw") else intensity
        k = float(cfg.error_band_k)
        upper = ref + k * sigma_intensity
        lower = ref - k * sigma_intensity
        plt.fill_between(energies, lower, upper, alpha=0.2, label=f"±{k:.0f}σ (Poisson)")

        plt.xlabel("Energy (eV)")
        plt.ylabel("Counts per eV")
        plt.title("Spectral Lineout")
        plt.grid(True)
        if cfg.yscale == "log":
            plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return LineoutResult(
        energies=energies,
        intensity=intensity,
        raw_sums=raw_sums,
        windows=windows,
        smoothed=smoothed,
        sigma_intensity=sigma_intensity,
    )
