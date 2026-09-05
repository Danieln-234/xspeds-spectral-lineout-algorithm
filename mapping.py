"""
Mapping stage: fit the detector geometry from two reference ridges.

Extracts the Ge L-alpha and L-beta ridge positions across row batches, then fits
parabolas to both ridges together with the cone-plane geometry (distance d, tilt
theta_z) via differential evolution followed by local least squares. The fitted
parameters feed the lineout stage.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution, least_squares

from geometry import conic_coefficients, rotated_basis

logger = logging.getLogger("xspeds.mapping")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


###################################
#      Configuration objects      #
###################################

@dataclass(frozen=True)
class MappingConfig:
    """Configuration for the mapping stage (ridge finding + geometry fit).

    Args:
        frame_index: Which frame to analyse.
        batch_size: Rows per batch for ridge extraction.
        smooth_sigma: Gaussian sigma for smoothing the summed column trace.
        r1: Column window for the first reference ridge.
        r2: Column window for the second reference ridge.
        alpha1_deg: Half-angle for reference line 1 (degrees).
        alpha2_deg: Half-angle for reference line 2 (degrees).
        de_maxiter: Differential evolution iteration cap.
        de_seed: Differential evolution seed, None for nondeterministic.
        w_curv: Weight for the curvature-matching residuals (A_fit - A_theory).
        w_vertex: Weight for the signed vertex-spacing residual.
    """
    frame_index: int = 8
    batch_size: int = 50
    smooth_sigma: float = 10.0

    r1: tuple[int, int] = (1250, 1370)
    r2: tuple[int, int] = (1380, 1560)

    alpha1_deg: float = 90.0 - 39.632
    alpha2_deg: float = 90.0 - 40.86

    de_maxiter: int = 2000
    de_seed: int | None = 42

    w_curv: float = 100.0
    w_vertex: float = 100.0


@dataclass(frozen=True)
class MappingResult:
    """Optimised XSPEDS geometry and mapping offsets.

    Args:
        d: Source-detector distance (negative, in pixels, consistent with u and v).
        theta_z: CCD tilt angle (radians).
        C1: x-vertex offset of reference ridge 1 (pixels).
        b: Shared y-vertex position (pixels).
        shift: Raw-geometry vertex u-coordinate of cone 1, used downstream to
            translate theoretical conics into pixel coordinates.
        residual_norms: Diagnostic norms of the final residual blocks.
    """
    d: float
    theta_z: float
    C1: float
    b: float
    shift: float
    residual_norms: dict[str, float]


###############################
#        Peak finding         #
###############################

def _sigma_from_peak_curvature(
    smooth: NDArray[np.float64],
    peak_idx: int,
    *,
    floor: float = 0.75,
    ceil: float = 20.0,
) -> float:
    """Estimate peak position uncertainty (pixels) from the curvature of a smoothed trace.

    Uses sigma ~ sqrt(f0 / -f'') at the maximum (Gaussian approximation).
    Returns ceil when the curvature is non-negative, i.e. not actually a peak.

    Args:
        smooth: Smoothed 1D trace of length W.
        peak_idx: Global index of the maximum in the trace.
        floor: Lower clamp on the returned sigma.
        ceil: Upper clamp on the returned sigma.

    Returns:
        Estimated sigma of the peak position in pixels.
    """
    i = int(np.clip(peak_idx, 1, smooth.size - 2))
    f0 = float(smooth[i])
    fpp = float(smooth[i - 1] - 2.0 * smooth[i] + smooth[i + 1])  # discrete 2nd derivative

    if f0 <= 0.0 or fpp >= 0.0:
        return float(ceil)
    sigma = np.sqrt(max(1e-12, f0 / (-fpp)))
    return float(np.clip(sigma, floor, ceil))


def find_scatter_peaks(
    array_dat: NDArray[np.float64],
    *,
    batch_size: int = 50,
    sigma: float = 10.0,
    r1: tuple[int, int] = (1250, 1370),
    r2: tuple[int, int] = (1380, 1560),
) -> tuple[NDArray[np.float64], ...]:
    """Locate the two ridge peaks in each row batch of a frame.

    Each batch of rows is summed into a column trace, smoothed, and the maximum
    inside each reference window taken as the ridge position. Peak position
    uncertainty comes from the local curvature of the smoothed trace.

    Args:
        array_dat: 2D frame (H x W).
        batch_size: Rows per batch.
        sigma: Gaussian smoothing sigma for the column trace.
        r1: Column window (start, end) for ridge 1.
        r2: Column window (start, end) for ridge 2.

    Returns:
        Tuple of arrays (rows, x1, sx1, x2, sx2): batch mid-row, peak column and
        column uncertainty for each ridge, one entry per batch.
    """
    H = array_dat.shape[0]
    rows, x1, sx1, x2, sx2 = [], [], [], [], []

    for start in range(0, H, batch_size):
        end = min(start + batch_size, H)
        trace = gaussian_filter1d(np.sum(array_dat[start:end, :], axis=0), sigma=sigma)

        for (lo, hi), xs, sxs in ((r1, x1, sx1), (r2, x2, sx2)):
            window = trace[lo:hi]
            # mean index in case the smoothed peak is flat-topped
            idx = lo + int(np.round(np.mean(np.where(window == np.max(window))[0])))
            xs.append(float(idx))
            sxs.append(_sigma_from_peak_curvature(trace, idx))
        rows.append(0.5 * (start + end))

    return tuple(np.asarray(a, dtype=np.float64) for a in (rows, x1, sx1, x2, sx2))


########################
#     Conic geometry   #
########################

def _left_vertex_curvature(
    coeffs: tuple[float, float, float, float, float, float],
    *,
    tol: float = 1e-12,
) -> tuple[float, float, float] | None:
    """Find the leftmost u-vertex of a conic and the local parabola curvature there.

    For the representation u(v), the vertex satisfies dF/dv = 0. Substituting
    v(u) = -(Bu + E)/(2C) into F gives a quadratic in u; the smaller root is the
    left vertex. See the report for details.

    Args:
        coeffs: Conic coefficients (A, B, C, D, E, F).
        tol: Degeneracy tolerance for the algebra.

    Returns:
        (u0, v0, A_local), or None if the geometry is degenerate or the local
        curvature is not positive (conic branches flipped).
    """
    A, B, C, D, E, F = coeffs
    if abs(C) < tol:
        return None

    a_u = 4.0 * C * A - B * B
    b_u = 4.0 * C * D - 2.0 * B * E
    c_u = 4.0 * C * F - E * E

    if abs(a_u) < tol:
        if abs(b_u) < tol:
            return None
        roots = [-c_u / b_u]
    else:
        disc = b_u * b_u - 4.0 * a_u * c_u
        if disc < 0:
            return None
        sdisc = float(np.sqrt(disc))
        roots = [(-b_u + sdisc) / (2.0 * a_u), (-b_u - sdisc) / (2.0 * a_u)]

    u0 = min(roots)
    v0 = -(B * u0 + E) / (2.0 * C)

    Fu = 2.0 * A * u0 + B * v0 + D
    if abs(Fu) < tol:
        return None

    A_local = -C / Fu
    # expect A_local > 0 for a left vertex opening rightwards in u(v)
    if not np.isfinite(A_local) or A_local <= 0:
        return None

    return float(u0), float(v0), float(A_local)


########################
#        Residuals     #
########################

def residuals(
    p: NDArray[np.float64],
    y1: NDArray[np.float64],
    x1: NDArray[np.float64],
    sx1: NDArray[np.float64],
    y2: NDArray[np.float64],
    x2: NDArray[np.float64],
    sx2: NDArray[np.float64],
    alpha1: float,
    alpha2: float,
    *,
    w_curv: float = 1.0,
    w_vertex: float = 1.0,
    sigma_gap: float = 5.0,
    eps_curv: float = 1e-18,
) -> NDArray[np.float64]:
    """Weighted residual vector for the joint parabola + geometry fit.

    Data residuals are normalised by per-point sigma_x, curvature residuals by
    the theoretical curvature, and the vertex-gap residual by sigma_gap, so all
    blocks are dimensionless and comparable.

    Args:
        p: Parameters [A1, A2, b, C1, dC, d, theta_z] in pixel units.
        y1: Batch mid-rows for ridge 1.
        x1: Peak columns for ridge 1.
        sx1: Peak column uncertainties for ridge 1.
        y2: Batch mid-rows for ridge 2.
        x2: Peak columns for ridge 2.
        sx2: Peak column uncertainties for ridge 2.
        alpha1: Half-angle for ridge 1 (radians).
        alpha2: Half-angle for ridge 2 (radians).
        w_curv: Weight on the curvature residuals.
        w_vertex: Weight on the vertex-gap residual.
        sigma_gap: Normalisation (pixels) for the vertex-gap residual.
        eps_curv: Guard against division by zero curvature.

    Returns:
        Concatenated residual vector [data1, data2, curvature, vertex gap].
    """
    A1_fit, A2_fit, b, C1_fit, dC_fit, d, theta_z = p
    C2_fit = C1_fit + dC_fit

    # Predicted ridge parabolas in pixel space (u = x, v = y)
    res_data1 = (x1 - (A1_fit * (y1 - b) ** 2 + C1_fit)) / np.maximum(sx1, 1e-6)
    res_data2 = (x2 - (A2_fit * (y2 - b) ** 2 + C2_fit)) / np.maximum(sx2, 1e-6)

    e_i, e_j = rotated_basis(theta_z)
    vtx1 = _left_vertex_curvature(conic_coefficients(alpha1, d, e_i, e_j))
    vtx2 = _left_vertex_curvature(conic_coefficients(alpha2, d, e_i, e_j))

    # Invalid geometry region: return a big penalty so the optimiser moves away
    if vtx1 is None or vtx2 is None:
        penalty = np.full(3, 1e6, dtype=np.float64)
        return np.concatenate([res_data1, res_data2, penalty])

    u01, _, A1_th = vtx1
    u02, _, A2_th = vtx2

    res_curv = np.array([
        (A1_fit - A1_th) / (abs(A1_th) + eps_curv),
        (A2_fit - A2_th) / (abs(A2_th) + eps_curv),
    ])
    res_gap = np.array([((C2_fit - C1_fit) - (u02 - u01)) / sigma_gap])

    return np.concatenate([res_data1, res_data2, w_curv * res_curv, w_vertex * res_gap])


# The optimiser works on scaled O(1) parameters, needed for DE to search well
# (least_squares gets the same effect from x_scale="jac").

def _unscale_params(p_s: NDArray[np.float64], *, S: float) -> NDArray[np.float64]:
    """Convert scaled optimiser params [a1, a2, bN, C1N, dCN, dN, theta_z] to pixel units.

    Args:
        p_s: Scaled parameter vector.
        S: Scale factor (longest image side in pixels).

    Returns:
        Physical parameters [A1, A2, b, C1, dC, d, theta_z].
    """
    a1, a2, bN, C1N, dCN, dN, theta_z = p_s
    S2 = S * S
    return np.array([a1 / S2, a2 / S2, bN * S, C1N * S, dCN * S, dN * S, theta_z])


def residuals_scaled(
    p_s: NDArray[np.float64],
    y1: NDArray[np.float64],
    x1: NDArray[np.float64],
    sx1: NDArray[np.float64],
    y2: NDArray[np.float64],
    x2: NDArray[np.float64],
    sx2: NDArray[np.float64],
    alpha1: float,
    alpha2: float,
    *,
    S: float,
    w_curv: float = 1.0,
    w_vertex: float = 1.0,
    sigma_gap: float = 5.0,
) -> NDArray[np.float64]:
    """residuals() evaluated at scaled parameters, see _unscale_params for the layout."""
    return residuals(
        _unscale_params(p_s, S=S),
        y1, x1, sx1, y2, x2, sx2, alpha1, alpha2,
        w_curv=w_curv, w_vertex=w_vertex, sigma_gap=sigma_gap,
    )


################################
#        Top-level run         #
################################

def run_mapping(
    image_data: Sequence[NDArray[np.float64]],
    *,
    config: MappingConfig | None = None,
) -> MappingResult:
    """Fit the detector geometry from the reference ridges of one frame.

    Global differential evolution over scaled parameters, then local least
    squares to polish, then unscale back to pixel units.

    Args:
        image_data: Stack of frames, indexed by config.frame_index.
        config: Mapping configuration, defaults to MappingConfig().

    Returns:
        MappingResult with the fitted geometry and residual norm diagnostics.
    """
    cfg = config or MappingConfig()
    sigma_gap = 5.0

    array_dat = np.asarray(image_data[cfg.frame_index], dtype=np.float64)
    y, x1, sx1, x2, sx2 = find_scatter_peaks(
        array_dat,
        batch_size=cfg.batch_size,
        sigma=cfg.smooth_sigma,
        r1=cfg.r1,
        r2=cfg.r2,
    )
    if y.size == 0:
        raise RuntimeError("No ridge peaks found.")

    alpha1 = float(np.deg2rad(cfg.alpha1_deg))
    alpha2 = float(np.deg2rad(cfg.alpha2_deg))

    H, W = array_dat.shape
    S = float(max(H, W))
    S2 = S * S

    # Physical bounds for p = [A1, A2, b, C1, dC, d, theta_z], with the factor
    # that converts each to its scaled counterpart
    bounds_phys = [
        (1e-10, 1e-2),              # A1
        (1e-10, 1e-2),              # A2
        (0.0, float(H)),            # b
        (-20.0, float(W) + 20.0),   # C1
        (-float(W), float(W)),      # dC
        (-100000.0, 0.0),           # d
        (-np.pi, 0.0),              # theta_z
    ]
    scale = [S2, S2, 1.0 / S, 1.0 / S, 1.0 / S, 1.0 / S, 1.0]
    bounds_scaled = [(lo * s, hi * s) for (lo, hi), s in zip(bounds_phys, scale)]

    def cost(p_s: NDArray[np.float64]) -> float:
        r = residuals_scaled(
            p_s, y, x1, sx1, y, x2, sx2, alpha1, alpha2,
            S=S, w_curv=cfg.w_curv, w_vertex=cfg.w_vertex, sigma_gap=sigma_gap,
        )
        return float(np.dot(r, r))

    de = differential_evolution(
        cost,
        bounds=bounds_scaled,
        maxiter=cfg.de_maxiter,
        polish=False,
        seed=cfg.de_seed,
        workers=1,
    )

    lsq = least_squares(
        residuals_scaled,
        de.x,
        args=(y, x1, sx1, y, x2, sx2, alpha1, alpha2),
        kwargs=dict(S=S, w_curv=cfg.w_curv, w_vertex=cfg.w_vertex, sigma_gap=sigma_gap),
        method="trf",
        x_scale="jac",
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=2000,
        loss="soft_l1", f_scale=1.0,
    )

    A1_opt, A2_opt, b_opt, C1_opt, dC_opt, d_opt, theta_z_opt = _unscale_params(lsq.x, S=S)
    C2_opt = C1_opt + dC_opt

    # Raw-geometry vertex of cone 1, passed downstream as the pixel-space shift
    e_i_opt, e_j_opt = rotated_basis(theta_z_opt)
    vtx1_opt = _left_vertex_curvature(conic_coefficients(alpha1, d_opt, e_i_opt, e_j_opt))
    if vtx1_opt is None:
        raise RuntimeError("Fitted geometry is degenerate for cone 1.")
    shift_part_1 = vtx1_opt[0]

    # Residual norms per block for diagnostics
    res_all = residuals_scaled(
        lsq.x, y, x1, sx1, y, x2, sx2, alpha1, alpha2,
        S=S, w_curv=cfg.w_curv, w_vertex=cfg.w_vertex, sigma_gap=sigma_gap,
    )
    n = y.size
    r1_norm = float(np.linalg.norm(res_all[:n]))
    r2_norm = float(np.linalg.norm(res_all[n:2 * n]))
    curv_norm = float(np.linalg.norm(res_all[2 * n:2 * n + 2]))
    vertex_norm = float(np.linalg.norm(res_all[-1:]))

    logger.info(
        f"Optimized: d={d_opt:.6g}, theta_z={np.rad2deg(theta_z_opt):.3f}deg, "
        f"C1={C1_opt:.6g}, C2={C2_opt:.6g}, b={b_opt:.6g}, "
        f"(A1,A2)=({A1_opt:.3e},{A2_opt:.3e}), shift_part_1={shift_part_1:.6g}"
    )
    logger.info(
        f"Residual norms: r1={r1_norm:.6g}, r2={r2_norm:.6g}, "
        f"curv={curv_norm:.6g}, vertex={vertex_norm:.6g}"
    )

    return MappingResult(
        d=float(d_opt),
        theta_z=float(theta_z_opt),
        C1=float(C1_opt),
        b=float(b_opt),
        shift=float(shift_part_1),
        residual_norms={
            "r1_data": r1_norm,
            "r2_data": r2_norm,
            "curvature": curv_norm,
            "vertex": vertex_norm,
        },
    )
