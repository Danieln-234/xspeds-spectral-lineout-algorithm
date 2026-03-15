"""
Energy/position mapping via reference-peak scatter fitting and cone–plane conic geometry.

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
    Configuration for the mapping stage (ridge finding + geometry fit).

    frame_index : which frame to analyse
    batch_size  : rows per batch for ridge extraction
    smooth_sigma: Gaussian sigma for smoothing the summed column trace
    r1, r2      : reference windows (column indices) for the two ridges
    alpha1_deg  : half-angle for reference line 1 (deg)
    alpha2_deg  : half-angle for reference line 2 (deg)

    de_maxiter, de_seed : differential evolution search controls

    w_curv   : weight for curvature-matching residuals (A_fit - A_theory)
    w_vertex : weight for signed vertex-spacing residual
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

    # “physics tie” weights
    w_curv: float = 100.0
    w_vertex: float = 100.0


@dataclass(frozen=True)
class MappingResult:
    """
    Optimised XSPEDS geometry and mapping offsets.

    d: source–detector distance (negative; in *pixels*, consistent with u,v)
    theta_z  : CCD tilt angle (radians)
    C1: x-vertex offset of reference ridge 1 (pixels)
    b: shared y-vertex position (pixels)
    shift: raw-geometry vertex u-coordinate for cone 1 (used downstream as shift_part_1)

    """
    d: float
    theta_z: float
    C1: float
    b: float
    shift: float

    # Diagnostics (optional)
    residual_norms: Dict[str, float]

    def as_tuple(self):
        # Keep ordering
        return (self.d, self.theta_z, self.C1, self.b, self.shift)


###############################
#        Peak finding         #
###############################





# NB for the scaling, the lsq part has x = 'jac' modifier but we need this for DE.

def _unscale_params(
    p_s: NDArray[np.float64],
    *,
    S: float,
) -> NDArray[np.float64]:
    """
    Convert scaled optimiser params to physical pixel params.

    Input p_s: [a1, a2, bN, C1N, dCN, dN, theta_z]
    Output p : [A1, A2, b,  C1,  dC,  d,  theta_z]  
    """
    a1, a2, bN, C1N, dCN, dN, theta_z = p_s
    S2 = S * S

    A1 = a1 / S2
    A2 = a2 / S2
    b  = bN * S
    C1 = C1N * S
    dC = dCN * S
    d  = dN * S
    return np.array([A1, A2, b, C1, dC, d, theta_z], dtype=np.float64)





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
    eps_curv: float = 1e-18,
) -> NDArray[np.float64]:
    """
    Wrapper around existing residuals
    """
    p = _unscale_params(p_s, S=S)
    return residuals(
        p,
        y1, x1, sx1,
        y2, x2, sx2,
        alpha1, alpha2,
        w_curv=w_curv,
        w_vertex=w_vertex,
        sigma_gap=sigma_gap,
        eps_curv=eps_curv,
    )


def _sigma_from_peak_curvature(
    smooth: NDArray[np.float64],
    peak_idx: int,
    *,
    floor: float = 0.75,  # TODO: feels arbitrary
    ceil: float = 20.0,
) -> float:
    """
    Estimate peak position uncertainty (in pixels) from the curvature of a smoothed 1D trace.

    Uses sigma roughly= sqrt( f0 / (-f'') ) at the maximum (Gaussian form approx).
    Falls back to 'ceil' when curvature is too small / unstable - does not happen for the current datasets.

    Inputs:
    smooth : (W,) smoothed trace
    peak_idx : index of maximum in the trace (global index)
    floor/ceil : clamp range for returned sigma

    Returns:
    sigma_x : float (pixels)
    """
    W = smooth.size

    # In the rare case the peaks are at the ends of the array (it really should not be)
    i = int(np.clip(peak_idx, 1, W - 2))
    f0 = float(smooth[i])
    fpp = float(smooth[i - 1] - 2.0 * smooth[i] + smooth[i + 1])  # discrete 2nd derivative

    # For a peak, fpp should be negative. If not, we default to ceil
    if f0 <= 0.0 or fpp >= 0.0:
        return float(ceil)

    sigma = np.sqrt(max(1e-12, f0 / (-fpp)))
    return float(np.clip(sigma, floor, ceil))


def find_scatter_peaks(
    array_dat: NDArray[np.float64],
    *,
    batch_size: int = 50,
    sigma: float = 10.0,
    r1_start: int = 1250,
    r1_end: int = 1370,
    r2_start: int = 1380,
    r2_end: int = 1560,
) -> pd.DataFrame:
    """
    Locate ridge peaks for two reference regions across row batches.

    Returns DataFrame with:
      row_start, row_end,
      peak_index1, peak_index2,
      peak_value1, peak_value2,
      sigma_x1, sigma_x2 - (estimated x-uncertainty in pixels for each peak)
    """
    H, W = array_dat.shape
    no_batches = int(np.ceil(H / batch_size))
    results: list[Dict[str, Any]] = []

    for i in range(no_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, H)

        sum_batch = np.sum(array_dat[start_index:end_index, :], axis=0)
        smooth_trace = gaussian_filter1d(sum_batch, sigma=sigma).astype(np.float64)

        r1 = smooth_trace[r1_start:r1_end]
        r2 = smooth_trace[r2_start:r2_end]
        if r1.size == 0 or r2.size == 0:
            continue

        v1 = float(np.max(r1))
        v2 = float(np.max(r2))

        # Mean index if flat-topped
        idx1_local = int(np.round(np.mean(np.where(r1 == v1)[0])))
        idx2_local = int(np.round(np.mean(np.where(r2 == v2)[0])))

        idx1 = int(r1_start + idx1_local)
        idx2 = int(r2_start + idx2_local)

        # Estimate uncertainty in the peak position (pixels)
        sx1 = _sigma_from_peak_curvature(smooth_trace, idx1)
        sx2 = _sigma_from_peak_curvature(smooth_trace, idx2)

        results.append(
            dict(
                row_start=start_index,
                row_end=end_index,
                peak_value1=v1,
                peak_value2=v2,
                peak_index1=idx1,
                peak_index2=idx2,
                sigma_x1=sx1,
                sigma_x2=sx2,
            )
        )

    return pd.DataFrame(results)


########################
#     Conic geometry   #
########################

def rotated_basis(
    theta_z: float,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return CCD in-plane orthonormal basis vectors after rotation.
    NB we assume for the method and pipeline,but it's here incase we add it to the optimiser thetax and y are 0

    """
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cx, sx = np.cos(theta_x), np.sin(theta_x)

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    R = Rx @ Ry @ Rz

    # Axesn u and v in detector plane
    e_i0 = np.array([0.0, 1.0, 0.0])    
    e_j0 = np.array([0.0, 0.0, 1.0])

    # tbf probably doesn't need to be float64 but just to be safe for scipy

    return (R @ e_i0).astype(np.float64), (R @ e_j0).astype(np.float64)


def cone_conic_coeffs(
    alpha: float,
    d: float,
    e_i: NDArray[np.float64],
    e_j: NDArray[np.float64],
) -> Tuple[float, float, float, float, float, float]:
    """
    Return (A,B,C,D,E,F) for the cone–plane intersection in CCD coordinates (u,v):
        F(u,v) = A u^2 + B u v + C v^2 + D u + E v + F = 0


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
    return float(A), float(B), float(C), float(D), float(E), float(F)


def left_vertex_and_local_curvature_u_of_v(
    coeffs: Tuple[float, float, float, float, float, float],
    *,
    tol: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute the “u-vertex” (leftmost) for the representation u(v), and the local
    parabola curvature coefficient A_theory at that vertex.

    See paper for more details

    Returns:
        dict(u0=..., v0=..., A_local=..., valid=1.0)
        If invalid/degenerate, valid=0.0 and values may be NaN.
    """
    A, B, C, D, E, F = coeffs

    # If C is around 0, the algebraic elimination v(u) = -(Bu+E)/(2C) is unstable.
    if abs(C) < tol:
        return {"u0": np.nan, "v0": np.nan, "A_local": np.nan, "valid": 0.0}

    # Solve F(u,v(u))=0 with v(u) from F_v=0:
    #   F_v = B u + 2 C v + E = 0 => v(u) = -(B u + E)/(2C)
    #
    # Substituting gives a quadratic in u:
    #   (4 C A - B^2) u^2 + (4 C D - 2 B E) u + (4 C F - E^2) = 0
    a_u = 4.0 * C * A - B * B
    b_u = 4.0 * C * D - 2.0 * B * E
    c_u = 4.0 * C * F - E * E

    # If a_u around 0, it becomes linear; handle that.
    roots_u: list[float] = []
    if abs(a_u) < tol:
        if abs(b_u) < tol:
            return {"u0": np.nan, "v0": np.nan, "A_local": np.nan, "valid": 0.0}
        roots_u = [float(-c_u / b_u)]
    else:
        disc = b_u * b_u - 4.0 * a_u * c_u
        if disc < 0:
            return {"u0": np.nan, "v0": np.nan, "A_local": np.nan, "valid": 0.0}
        sdisc = float(np.sqrt(max(0.0, disc)))
        roots_u = [float((-b_u + sdisc) / (2.0 * a_u)), float((-b_u - sdisc) / (2.0 * a_u))]

    # Compute corresponding v for each root and pick the “left” vertex (min u).
    candidates: list[Tuple[float, float]] = []
    for u in roots_u:
        v = float(-(B * u + E) / (2.0 * C))
        candidates.append((u, v))

    u0, v0 = min(candidates, key=lambda t: t[0])  # leftmost in u

    # Compute local curvature coefficient A_local = -C / F_u(u0,v0)
    Fu = 2.0 * A * u0 + B * v0 + D
    if abs(Fu) < tol:
        return {"u0": u0, "v0": v0, "A_local": np.nan, "valid": 0.0}

    A_local = -C / Fu

    # We expect A_local > 0 for a left-vertex opening to the right in u(v).
    # NB below is probably not needed, but there was one instance of this being an error in testing.
    # Likely due to the bounds being too loose. and the conic branches flipped.
    if not np.isfinite(A_local) or A_local <= 0:
        return {"u0": u0, "v0": v0, "A_local": float(A_local), "valid": 0.0}

    return {"u0": u0, "v0": v0, "A_local": float(A_local), "valid": 1.0}


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
    """
    Weighted residual vector.

    Data residuals are normalised by per-point sigma_x (pixels):
        r_data = (x - x_pred) / sigma_x

    Curvature residuals:
        r_curv = (A_fit - A_th) / (abs(A_th) + eps)   NB the eps is neeeded

    Vertex-gap residual is normalised by sigma_gap (pixels):
        r_gap = ((C2-C1) - (u02-u01)) / sigma_gap


    """
    # Unpack parameters
    A1_fit, A2_fit, b, C1_fit, dC_fit, d, theta_z = p
    C2_fit = C1_fit + dC_fit

    # Predict ridge parabolas in pixel space (u = x, v = y)
    x1_pred = A1_fit * (y1 - b) ** 2 + C1_fit
    x2_pred = A2_fit * (y2 - b) ** 2 + C2_fit

    # Weighted data residuals
    sx1_safe = np.maximum(sx1, 1e-6)
    sx2_safe = np.maximum(sx2, 1e-6)
    res_data1 = (x1 - x1_pred) / sx1_safe
    res_data2 = (x2 - x2_pred) / sx2_safe

    # Geometry terms
    e_i, e_j = rotated_basis(theta_z, 0.0, 0.0)
    coeffs1 = cone_conic_coeffs(alpha1, d, e_i, e_j)
    coeffs2 = cone_conic_coeffs(alpha2, d, e_i, e_j)

    vtx1 = left_vertex_and_local_curvature_u_of_v(coeffs1)
    vtx2 = left_vertex_and_local_curvature_u_of_v(coeffs2)

    # If invalid geometry region, return big penalty so optimiser moves away
    if vtx1["valid"] < 0.5 or vtx2["valid"] < 0.5:
        penalty = np.full(3, 1e6, dtype=np.float64)
        return np.concatenate([res_data1, res_data2, penalty])

    A1_th = float(vtx1["A_local"])
    A2_th = float(vtx2["A_local"])
    u01 = float(vtx1["u0"])
    u02 = float(vtx2["u0"])

    # Dimensionless curvature residuals
    res_curv = np.array(
        [
            (A1_fit - A1_th) / (abs(A1_th) + eps_curv),
            (A2_fit - A2_th) / (abs(A2_th) + eps_curv),
        ],
        dtype=np.float64,
    )

    # Signed vertex-gap residual, normalised
    res_gap = np.array([((C2_fit - C1_fit) - (u02 - u01)) / max(sigma_gap, 1e-6)], dtype=np.float64)

    return np.concatenate([res_data1, res_data2, w_curv * res_curv, w_vertex * res_gap])


################################
#        Top-level run         #
################################

def run_mapping(image_data, *, config=None):
    cfg = config or MappingConfig()

    array_dat = np.asarray(image_data[cfg.frame_index], dtype=np.float64)
    (r1_start, r1_end), (r2_start, r2_end) = cfg.r1, cfg.r2

    df = find_scatter_peaks(
        array_dat,
        batch_size=cfg.batch_size,
        sigma=cfg.smooth_sigma,
        r1_start=r1_start, r1_end=r1_end,
        r2_start=r2_start, r2_end=r2_end,
    )
    if df.empty:
        raise RuntimeError("No ridge peaks found.")

    df["Batch_Avg_Row"] = 0.5 * (df["row_start"] + df["row_end"])

    # Keep one point per batch (they’re already per batch), but groupby is fine
    r1_df = df.groupby("Batch_Avg_Row", as_index=False)[["peak_index1", "sigma_x1"]].mean()
    r2_df = df.groupby("Batch_Avg_Row", as_index=False)[["peak_index2", "sigma_x2"]].mean()

    y1 = r1_df["Batch_Avg_Row"].to_numpy(np.float64)
    x1 = r1_df["peak_index1"].to_numpy(np.float64)
    sx1 = r1_df["sigma_x1"].to_numpy(np.float64)

    y2 = r2_df["Batch_Avg_Row"].to_numpy(np.float64)
    x2 = r2_df["peak_index2"].to_numpy(np.float64)
    sx2 = r2_df["sigma_x2"].to_numpy(np.float64)

    alpha1 = float(np.deg2rad(cfg.alpha1_deg))
    alpha2 = float(np.deg2rad(cfg.alpha2_deg))



    # p = [A1, A2, b, C1, dC, d, theta_z]
    H, W = array_dat.shape
    S = float(max(H, W))  # 2048 for the etsted samples

    # Bounds
    bounds_phys = [
        (1e-10, 1e-2),              # A1
        (1e-10, 1e-2),              # A2
        (0.0, float(H)),            # b
        (-20.0, float(W) + 20.0),   # C1
        (-float(W), float(W)),      # dC
        (-100000.0, 0.0),           # d
        (-np.pi, 0.0),              # theta_z
    ]

    # Convert bounds for scaled variables
    bounds_scaled = []
    S2 = S * S

    bounds_scaled.append((bounds_phys[0][0] * S2, bounds_phys[0][1] * S2))  # a1
    bounds_scaled.append((bounds_phys[1][0] * S2, bounds_phys[1][1] * S2))  # a2

    bounds_scaled.append((bounds_phys[2][0] / S, bounds_phys[2][1] / S))    # bN
    bounds_scaled.append((bounds_phys[3][0] / S, bounds_phys[3][1] / S))    # C1N
    bounds_scaled.append((bounds_phys[4][0] / S, bounds_phys[4][1] / S))    # dCN
    bounds_scaled.append((bounds_phys[5][0] / S, bounds_phys[5][1] / S))    # dN
    # theta_z unchanged
    bounds_scaled.append(bounds_phys[6])                                     # theta_z

    #  Cost function
    def cost(p_s: NDArray[np.float64]) -> float:
        r = residuals_scaled(
            p_s, y1, x1, sx1, y2, x2, sx2, alpha1, alpha2,
            S=S, w_curv=cfg.w_curv, w_vertex=cfg.w_vertex, sigma_gap=5.0
        )
        return float(np.dot(r, r))

    # DE 
    de = differential_evolution(
        cost,
        bounds=bounds_scaled,
        maxiter=cfg.de_maxiter,
        polish=False,
        seed=cfg.de_seed,
        workers=1,
    )

    # LSQ
    lsq = least_squares(
        residuals_scaled,
        de.x,
        args=(y1, x1, sx1, y2, x2, sx2, alpha1, alpha2),
        kwargs=dict(S=S, w_curv=cfg.w_curv, w_vertex=cfg.w_vertex, sigma_gap=5.0),
        method="trf",
        x_scale="jac",
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=2000,
        loss="soft_l1", f_scale=1.0,
    )

    # Convert optimum back to physical pixel params
    p_opt = _unscale_params(lsq.x, S=S)
    A1_opt, A2_opt, b_opt, C1_opt, dC_opt, d_opt, theta_z_opt = p_opt
    C2_opt = C1_opt + dC_opt


    # Compute the raw geometry vertex for cone 1 to pass downstream
    e_i_opt, e_j_opt = rotated_basis(theta_z_opt, 0.0, 0.0)
    coeffs1_opt = cone_conic_coeffs(alpha1, d_opt, e_i_opt, e_j_opt)
    vtx1_opt = left_vertex_and_local_curvature_u_of_v(coeffs1_opt)
    shift_part_1 = float(vtx1_opt["u0"])  # matches the old “cone1 vertex x” idea

    # Diagnostic residual norms 

    res_all = residuals_scaled(
        lsq.x,
        y1, x1, sx1,
        y2, x2, sx2,
        alpha1, alpha2,
        S=S,
        w_curv=cfg.w_curv,
        w_vertex=cfg.w_vertex,
        sigma_gap=5.0,
    )


    n1, n2 = len(y1), len(y2)
    r1_norm = float(np.linalg.norm(res_all[:n1]))
    r2_norm = float(np.linalg.norm(res_all[n1:n1 + n2]))
    curv_norm = float(np.linalg.norm(res_all[n1 + n2:n1 + n2 + 2]))
    vertex_norm = float(np.linalg.norm(res_all[-1:]))


    logger.info(
        "Optimized: "
        f"d={d_opt:.6g}, theta_z={np.rad2deg(theta_z_opt):.3f}°, "
        f"C1={C1_opt:.6g}, C2={C2_opt:.6g}, b={b_opt:.6g}, "
        f"(A1,A2)=({A1_opt:.3e},{A2_opt:.3e}), "
        f"shift_part_1(u0_1)={shift_part_1:.6g}"
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
