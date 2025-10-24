"""
'Energy/position mapping via reference-peak scatter fitting and conic geometry.'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution, least_squares

logger = logging.getLogger("xspeds.mapping")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


###################################
#      Configuration objects      #
###################################

@dataclass(frozen=True)
class MappingConfig:
    """
    'Configuration for the mapping stage (ridge finding + conic fit).'

    Args:
        frame_index: Frame to analyze for mapping.
        batch_size: Rows per batch for sum-over-rows ridge extraction.
        smooth_sigma: Gaussian sigma for smoothing the column-sum trace.
        r1: (start, end) indices for reference region 1 (where the Lβ curve is).
        r2: (start, end) indices for reference region 2 (where the Lα curve is).
        alpha1_deg: Half-angle for the Lβ emission (1218 eV)
        alpha2_deg: Half-angle for the Lα emission (1188 eV)
        de_maxiter: Max iterations for differential evolution (global search).
        de_seed: Random seed for DE for reproducibility (None disables).
        w_focal: Weight for focal-length residuals.
        w_vertex: Weight for vertex-spacing residual.

    Returns:
        None.
    """
    frame_index: int = 8
    batch_size: int = 50
    smooth_sigma: float = 10.0
    r1: Tuple[int, int] = (1250, 1370)
    r2: Tuple[int, int] = (1380, 1560)
    alpha1_deg: float = 90.0 - 39.632
    alpha2_deg: float = 90.0 - 40.86
    de_maxiter: int = 2000
    de_seed: int | None = 42
    w_focal: float = 100.0
    w_vertex: float = 100.0


@dataclass(frozen=True)
class MappingResult:
    """Optimised geometry & parabola offsets."""
    d: float                  # source distance (negative; same units as your model)
    theta_z: float            # radians
    C1: float                 # pixels
    b: float                  # pixels (shared y-vertex)
    shift: float              # pixels, convenience from cone1 vertex x

    # Optional diagnostics 
    focal_fit: tuple[float, float]        # (p1_fit, p2_fit)
    focal_theory: tuple[float, float]     # (p1_theory, p2_theory)
    residual_norms: Dict[str, float]      # {'r1_data':..., 'r2_data':..., 'focal':..., 'vertex':...}

    def as_tuple(self):
        return (self.d, self.theta_z, self.C1, self.b, self.shift)

###############################
#        Peak finding         #
###############################

def find_scatter_peaks(
    array_dat: NDArray[np.float64],
    *,
    batch_size: int = 5,
    sigma: float = 10.0,
    r1_start: int = 1250,
    r1_end: int = 1370,
    r2_start: int = 1380,
    r2_end: int = 1560,
) -> pd.DataFrame:
    """
    'Locate ridge peaks for two reference regions across row batches.'

    Args:
        array_dat: 2D array (H×W) for the selected frame (float-like ADU).
        batch_size: Rows per batch for summation (last batch may be smaller).
        sigma: Gaussian smoothing sigma for the column-sum trace.
        r1_start: Inclusive column start for region 1.
        r1_end: Exclusive column end for region 1.
        r2_start: Inclusive column start for region 2.
        r2_end: Exclusive column end for region 2.

    Returns:
        DataFrame with: row_start, row_end, peak_value1, peak_index1, peak_index2.
    """
    H, W = array_dat.shape
    no_batches = int(np.ceil(H / batch_size))
    results: list[Dict[str, Any]] = []

    logger.info(
        f"Finding scatter peaks: H={H}, W={W}, batches={no_batches}, "
        f"regions=({r1_start}:{r1_end}),({r2_start}:{r2_end})"
    )

    for i in range(no_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, H)
        sum_batch = np.sum(array_dat[start_index:end_index, :], axis=0)
        smooth = gaussian_filter1d(sum_batch, sigma=sigma)

        r1 = smooth[r1_start:r1_end]
        r2 = smooth[r2_start:r2_end]
        if r1.size == 0 or r2.size == 0:
            logger.info(f"Empty reference window in batch {i}, skipping.")
            continue

        v1 = float(np.max(r1))
        v2 = float(np.max(r2))
        idx1 = int(r1_start + int(np.round(np.mean(np.where(r1 == v1)[0]))))
        idx2 = int(r2_start + int(np.round(np.mean(np.where(r2 == v2)[0]))))

        results.append(
            dict(
                row_start=start_index,
                row_end=end_index,
                peak_value1=v1,
                peak_index1=idx1,
                peak_index2=idx2,
            )
        )

    df = pd.DataFrame(results)
    logger.info(f"Scatter peaks found for {len(df)} batches.")
    return df

########################
#     Conic model      #
########################

def rotated_basis(theta_z: float, theta_x: float = 0.0, theta_y: float = 0.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    'Return CCD in-plane orthonormal basis vectors after rotation.'

    Args:
        theta_z: Rotation around z-axis (radians).
        theta_x: Rotation around x-axis (radians).
        theta_y: Rotation around y-axis (radians).

    Returns:
        (e_i, e_j): i- and j-axes (3-vectors) in the CCD plane after rotation.
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


def compute_conic_params(alpha: float, d: float, e_i: NDArray[np.float64], e_j: NDArray[np.float64], *, tol: float = 1e-6) -> Dict[str, Any]:
    """
    'Compute conic-section parameters induced by geometry (Appendix B).'

    Args:
        alpha: Half-angle for the cone (radians).
        d: Source distance (negative).
        e_i: CCD i-axis (3-vector).
        e_j: CCD j-axis (3-vector).
        tol: Tolerance for discriminant/eigenvalue checks.

    Returns:
        Dictionary with keys like 'type', 'vertex', 'focal_length', 'foci', etc.
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

    if np.isclose(disc, 0.0, atol=tol):
        M = np.array([[2 * A, B], [B, 2 * C]])
        vertex, *_ = np.linalg.lstsq(M, -np.array([D, E]), rcond=None)
        eigvals, eigvecs = np.linalg.eig(M)
        nz = np.where(np.abs(eigvals) > tol)[0]
        lam = eigvals[nz[0]] if nz.size else 1.0
        axis = (eigvecs[:, nz[0]] if nz.size else np.array([1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        focal_length = 1.0 / (2.0 * np.abs(lam))
        focus = (vertex[0] + focal_length * axis[0], vertex[1] + focal_length * axis[1])
        return dict(type="parabola", vertex=vertex, focus=focus, focal_length=float(focal_length), foci=0, coeffs=coeffs, discriminant=float(disc))

    if disc < 0:
        M = np.array([[2 * A, B], [B, 2 * C]])
        center = np.linalg.solve(M, -np.array([D, E]))
        ei0, ej0 = center
        F0 = B * ei0 * ej0 + A * ei0**2 + C * ej0**2 + D * ei0 + E * ej0 + F
        K = -F0
        Q = np.array([[A / K, B / (2 * K)], [B / (2 * K), C / K]])
        eigvals, eigvecs = np.linalg.eig(Q)
        order = np.argsort(eigvals)
        lam1, lam2 = eigvals[order[0]], eigvals[order[1]]
        vec1 = eigvecs[:, order[0]]
        a, b = 1.0 / np.sqrt(lam1), 1.0 / np.sqrt(lam2)
        if b > a:
            a, b = b, a
        ecc = np.sqrt(1.0 - (b**2 / a**2))
        cval = a * ecc
        ang = float(np.arctan2(vec1[1], vec1[0]))
        focus1 = (ei0 + cval * np.cos(ang), ej0 + cval * np.sin(ang))
        focus2 = (ei0 - cval * np.cos(ang), ej0 - cval * np.sin(ang))
        vertex1 = (ei0 + a * np.cos(ang), ej0 + a * np.sin(ang))
        fl = a - cval
        return dict(type="ellipse", center=center, semi_axes=(float(a), float(b)), angle=ang, eccentricity=float(ecc), foci=(focus1, focus2), vertex=vertex1, focal_length=float(fl), coeffs=coeffs, discriminant=float(disc))

    M = np.array([[2 * A, B], [B, 2 * C]])
    center = np.linalg.solve(M, -np.array([D, E]))
    ei0, ej0 = center
    F0 = B * ei0 * ej0 + A * ei0**2 + C * ej0**2 + D * ei0 + E * ej0 + F
    K = -F0
    Q = np.array([[A / K, B / (2 * K)], [B / (2 * K), C / K]])
    eigvals, eigvecs = np.linalg.eig(Q)
    idx = np.argmax(eigvals)
    lam_p = eigvals[idx]
    lam_n = np.min(eigvals)
    vec_p = eigvecs[:, idx]
    a = 1.0 / np.sqrt(lam_p)
    b = 1.0 / np.sqrt(-lam_n)
    ang = float(np.arctan2(vec_p[1], vec_p[0]))
    cval = np.sqrt(a**2 + b**2)
    vertex1 = (ei0 + a * np.cos(ang), ej0 + a * np.sin(ang))
    fl = cval - a
    focus1 = (ei0 + cval * np.cos(ang), ej0 + cval * np.sin(ang))
    focus2 = (ei0 - cval * np.cos(ang), ej0 - cval * np.sin(ang))
    return dict(type="hyperbola", center=center, semi_axes=(float(a), float(b)), angle=ang, foci=(focus1, focus2), vertex=vertex1, focal_length=float(fl), coeffs=coeffs, discriminant=float(disc))


# Residuals 
#TODO: normalise by expected uncertainty for weighted least-squares
def residuals(
    p: NDArray[np.float64],
    y1: NDArray[np.float64],
    x1: NDArray[np.float64],
    y2: NDArray[np.float64],
    x2: NDArray[np.float64],
    alpha1: float,
    alpha2: float,
    *,
    w_focal: float = 100.0,
    w_vertex: float = 100.0,
) -> NDArray[np.float64]:
    """
    'Concatenate data, focal-length, and vertex-spacing residuals for both regions.'

    Args:
        p: [A1, A2, b, C1, C2, d, theta_z].
        y1, x1: Ridge coords for region 1.
        y2, x2: Ridge coords for region 2.
        alpha1: Cone half-angle (radians) for region 1.
        alpha2: Cone half-angle (radians) for region 2.
        w_focal: Weight for focal length residuals.
        w_vertex: Weight for vertex spacing residual.

    Returns:
        1D residual vector.
    """
    A1, A2, b, C1, C2, d, theta_z = p
    e_i, e_j = rotated_basis(theta_z, 0.0, 0.0)
    cone1 = compute_conic_params(alpha1, d, e_i, e_j)
    cone2 = compute_conic_params(alpha2, d, e_i, e_j)

    x1_pred = A1 * (y1 - b) ** 2 + C1
    x2_pred = A2 * (y2 - b) ** 2 + C2
    res_data1 = x1 - x1_pred
    res_data2 = x2 - x2_pred

    p1_fit = 1.0 / (4.0 * A1)
    p2_fit = 1.0 / (4.0 * A2)
    p1_th = float(cone1["focal_length"])
    p2_th = float(cone2["focal_length"])
    res_focal = np.array([p1_fit - p1_th, p2_fit - p2_th], dtype=np.float64)

    v1x = float(cone1["vertex"][0])
    v2x = float(cone2["vertex"][0])
    res_vertex = np.array([np.abs(C2 - C1) - np.abs(v2x - v1x)], dtype=np.float64)

    return np.concatenate([res_data1, res_data2, w_focal * res_focal, w_vertex * res_vertex])


# Top level 

def run_mapping(image_data: Sequence[NDArray[np.float64]], *, config: MappingConfig | None = None) -> MappingResult:
    """
    'Estimate (d, theta_z) and parabola offsets from scatter ridges.'

    Args:
        image_data: Sequence of 2D frames (H×W).
        config: MappingConfig with all tunables. If None, defaults are used.

    Returns:
        (d_opt, theta_z_opt, C1_opt, b_opt, shift_part_1).
    """
    cfg = config or MappingConfig()
    if len(image_data) <= cfg.frame_index:
        raise IndexError(f"frame_index {cfg.frame_index} out of range for stack length {len(image_data)}")

    array_dat = np.asarray(image_data[cfg.frame_index], dtype=np.float64)
    (r1_start, r1_end), (r2_start, r2_end) = cfg.r1, cfg.r2
    logger.info(
        f"Mapping: frame={cfg.frame_index}, batch={cfg.batch_size}, "
        f"regions=({r1_start}:{r1_end}),({r2_start}:{r2_end}), "
        f"alphas=({cfg.alpha1_deg:.3f}°, {cfg.alpha2_deg:.3f}°)"
    )

    df = find_scatter_peaks(
        array_dat,
        batch_size=cfg.batch_size,
        sigma=cfg.smooth_sigma,
        r1_start=r1_start,
        r1_end=r1_end,
        r2_start=r2_start,
        r2_end=r2_end,
    )

    df["Batch_Avg_Row"] = 0.5 * (df["row_start"] + df["row_end"])
    r1_df = df.groupby("Batch_Avg_Row", as_index=False)["peak_index1"].mean()
    r2_df = df.groupby("Batch_Avg_Row", as_index=False)["peak_index2"].mean()
    y1, x1 = r1_df["Batch_Avg_Row"].to_numpy(np.float64), r1_df["peak_index1"].to_numpy(np.float64)
    y2, x2 = r2_df["Batch_Avg_Row"].to_numpy(np.float64), r2_df["peak_index2"].to_numpy(np.float64)

    alpha1 = np.deg2rad(cfg.alpha1_deg)
    alpha2 = np.deg2rad(cfg.alpha2_deg)

    #TODO: normalise by expected uncertainty for weighted least-squares
    def cost(p: NDArray[np.float64]) -> float:
        r = residuals(p, y1, x1, y2, x2, alpha1, alpha2, w_focal=cfg.w_focal, w_vertex=cfg.w_vertex)
        return float(np.dot(r, r))

    p0 = np.array([1e-5, 1e-5, 1000.0, 0.0, 0.0, -10348.0, np.deg2rad(-40.0)], dtype=np.float64)
    bounds = [
        (1e-10, 1e-2), (1e-10, 1e-2),
        (0.0, 2000.0), (-20.0, 2000.0), (-20.0, 2000.0),
        (-100000.0, 0.0), (-np.pi, 0.0),
    ]

    logger.info(f"Global search (DE) maxiter={cfg.de_maxiter}, seed={cfg.de_seed}")
    de = differential_evolution(cost, bounds=bounds, maxiter=cfg.de_maxiter, polish=False, seed=cfg.de_seed, updating="deferred", workers=1)
    logger.info(f"DE params: {np.array2string(de.x, precision=6, floatmode='fixed')}")

    logger.info("Local refinement (least-squares)…")
    lsq = least_squares(
        residuals, de.x, args=(y1, x1, y2, x2, alpha1, alpha2),
        kwargs=dict(w_focal=cfg.w_focal, w_vertex=cfg.w_vertex),
        method="trf", x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000
    )
    A1, A2, b_opt, C1_opt, C2_opt, d_opt, theta_z_opt = lsq.x

    p1_fit, p2_fit = 1.0 / (4.0 * A1), 1.0 / (4.0 * A2)
    e_i_opt, e_j_opt = rotated_basis(theta_z_opt, 0.0, 0.0)
    cone1 = compute_conic_params(alpha1, d_opt, e_i_opt, e_j_opt)
    cone2 = compute_conic_params(alpha2, d_opt, e_i_opt, e_j_opt)
    shift_part_1 = float(cone1["vertex"][0])

    logger.info(
        "Optimized: "
        f"d={d_opt:.6g}, theta_z={np.rad2deg(theta_z_opt):.3f}°, "
        f"C1={C1_opt:.6g}, b={b_opt:.6g}; "
        f"fit_focals=({p1_fit:.6g},{p2_fit:.6g}) "
        f"theory=({cone1['focal_length']:.6g},{cone2['focal_length']:.6g})"
    )

    # Residual norms for quick health check
    res_all = residuals(lsq.x, y1, x1, y2, x2, alpha1, alpha2, w_focal=cfg.w_focal, w_vertex=cfg.w_vertex)
    n1, n2 = len(y1), len(y2)

    r1_norm, r2_norm, focal_norm, vertex_norm = (
        np.linalg.norm(res_all[:n1]),
        np.linalg.norm(res_all[n1:n1+n2]),
        np.linalg.norm(res_all[n1+n2:n1+n2+2]),
        np.linalg.norm(res_all[-1:]),
    )

    logger.info(
        f"Residual norms: r1={r1_norm:.6g}, r2={r2_norm:.6g}, "
        f"focal={focal_norm:.6g}, vertex={vertex_norm:.6g}"
    )


    return MappingResult(d_opt, theta_z_opt, C1_opt, b_opt, shift_part_1,
                         focal_fit=(p1_fit, p2_fit),
                         focal_theory=(cone1['focal_length'], cone2['focal_length']),
                         residual_norms={
                                "r1_data": float(r1_norm),
                                "r2_data": float(r2_norm),
                                "focal": float(focal_norm),
                                "vertex": float(vertex_norm),
                            }
                         )