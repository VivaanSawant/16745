"""
C2_jacobian_covariance_optimizer_SPEC.py
==========================================
**PDF section:** Track C — C2 (Analysis layers)

**Jacobian (PDF):** ∂(landing Δy, Δz) / ∂(release state) with release state the **6 scalars**
(x0, y0, z0, vx0, vy0, vz0) at release — computed here with **finite differences** (ε small).

**Predicted landing covariance (PDF):** Σ_land ≈ J Σ_release Jᵀ (first-order), compare to MC scatter.

**Optimizer (PDF):** `scipy.optimize.minimize` (Nelder–Mead) with objective = **negative expected score**
from Monte Carlo — **stub** provided; warm-start and target generalization are project extensions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from track_B_projectile_SPEC.B2_ode_integrator_event_board_plane_SPEC import integrate_until_board_SPEC


def landing_deltas_for_release_SPEC(s6: np.ndarray, wind_xyz=(0.0, 0.0, 0.0), drag_enabled: bool = True) -> np.ndarray:
    """Return (Δy, Δz) in meters or (0,0) if miss."""
    s6 = np.asarray(s6, dtype=float).reshape(6)
    info = integrate_until_board_SPEC(s6, wind_xyz_mps=wind_xyz, drag_enabled=drag_enabled)
    if not info["hit"]:
        return np.array([0.0, 0.0])
    return np.array([info["delta_y_m"], info["delta_z_m"]], dtype=float)


def landing_info_for_release_SPEC(
    s6: np.ndarray,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
) -> dict:
    """
    Return a compact landing record for one release state.

    The `"landing_m"` entry is NaN for misses so downstream covariance estimates can
    decide explicitly whether to ignore misses or impute them.
    """
    s6 = np.asarray(s6, dtype=float).reshape(6)
    info = integrate_until_board_SPEC(s6, wind_xyz_mps=wind_xyz, drag_enabled=drag_enabled)
    if not info["hit"]:
        return {
            "hit": False,
            "landing_m": np.array([np.nan, np.nan], dtype=float),
            "delta_y_m": np.nan,
            "delta_z_m": np.nan,
            "hit_info": info,
        }
    landing = np.array([info["delta_y_m"], info["delta_z_m"]], dtype=float)
    return {
        "hit": True,
        "landing_m": landing,
        "delta_y_m": float(landing[0]),
        "delta_z_m": float(landing[1]),
        "hit_info": info,
    }


def jacobian_landing_wrt_release_SPEC(
    s6: np.ndarray,
    eps: float = 1e-4,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
) -> np.ndarray:
    """
    **PDF C2:** 2×6 Jacobian J where rows are [∂Δy, ∂Δz] and cols are ∂/∂ each release component.
    """
    s6 = np.asarray(s6, dtype=float).reshape(6)
    base = landing_deltas_for_release_SPEC(s6, wind_xyz, drag_enabled)
    J = np.zeros((2, 6))
    for j in range(6):
        sp = s6.copy()
        sp[j] += eps
        d = landing_deltas_for_release_SPEC(sp, wind_xyz, drag_enabled)
        J[:, j] = (d - base) / eps
    return J


def predicted_landing_covariance_SPEC(J: np.ndarray, Sigma_release6: np.ndarray) -> np.ndarray:
    """**PDF C2:** Σ_land = J Σ_release Jᵀ (Σ_release is 6×6)."""
    return J @ Sigma_release6 @ J.T


def empirical_landing_covariance_SPEC(
    landings_m: np.ndarray,
    hit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Empirical 2×2 covariance of landing deltas.

    Misses should generally be excluded from first-order local covariance studies,
    so callers can pass `hit_mask` or simply provide rows with NaNs.
    """
    landings_m = np.asarray(landings_m, dtype=float)
    if landings_m.size == 0:
        return np.zeros((2, 2))
    if hit_mask is not None:
        mask = np.asarray(hit_mask, dtype=bool).reshape(-1)
    else:
        mask = np.all(np.isfinite(landings_m), axis=1)
    valid = landings_m[mask]
    if valid.shape[0] <= 1:
        return np.zeros((2, 2))
    return np.cov(valid.T)


def release_variance_contributions_SPEC(
    J: np.ndarray,
    Sigma_release6: np.ndarray,
    labels: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz"),
) -> list[dict]:
    """
    Rank release-state components by their diagonal-only contribution to landing spread.

    This is a local first-order sensitivity heuristic that is especially useful for
    deciding what an RL reward or curriculum should focus on first.
    """
    J = np.asarray(J, dtype=float).reshape(2, 6)
    Sigma_release6 = np.asarray(Sigma_release6, dtype=float).reshape(6, 6)
    rows = []
    for idx, label in enumerate(labels):
        sigma_ii = float(Sigma_release6[idx, idx])
        isolated = np.zeros((6, 6), dtype=float)
        isolated[idx, idx] = sigma_ii
        cov_i = J @ isolated @ J.T
        rows.append({
            "label": label,
            "release_variance": sigma_ii,
            "landing_trace_contribution": float(np.trace(cov_i)),
            "landing_covariance": cov_i,
        })
    rows.sort(key=lambda row: row["landing_trace_contribution"], reverse=True)
    return rows


def summarize_release_robustness_SPEC(
    nominal_release6: np.ndarray,
    release_states6: np.ndarray,
    landings_m: np.ndarray,
    scores: np.ndarray,
    hit_mask: np.ndarray | None = None,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
) -> dict:
    """
    Bundle the main C2 quantities into one RL-friendly summary dict.
    """
    nominal_release6 = np.asarray(nominal_release6, dtype=float).reshape(6)
    release_states6 = np.asarray(release_states6, dtype=float)
    landings_m = np.asarray(landings_m, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if hit_mask is None:
        hit_mask = np.all(np.isfinite(landings_m), axis=1)
    else:
        hit_mask = np.asarray(hit_mask, dtype=bool).reshape(-1)

    Sigma_release = np.cov(release_states6.T) if release_states6.shape[0] > 1 else np.zeros((6, 6))
    J = jacobian_landing_wrt_release_SPEC(nominal_release6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
    Sigma_land_predicted = predicted_landing_covariance_SPEC(J, Sigma_release)
    Sigma_land_empirical = empirical_landing_covariance_SPEC(landings_m, hit_mask=hit_mask)
    return {
        "jacobian_2x6": J,
        "Sigma_release_6x6": Sigma_release,
        "Sigma_land_predicted_2x2": Sigma_land_predicted,
        "Sigma_land_empirical_2x2": Sigma_land_empirical,
        "sensitivity_ranking": release_variance_contributions_SPEC(J, Sigma_release),
        "hit_rate": float(np.mean(hit_mask)) if hit_mask.size else 0.0,
        "mean_score": float(np.mean(scores)) if scores.size else 0.0,
        "std_score": float(np.std(scores)) if scores.size else 0.0,
        "n_samples": int(scores.size),
        "n_hits": int(np.sum(hit_mask)),
    }


def minimize_negative_mc_score_stub_SPEC(
    mc_objective_fn,
    x0_9params: np.ndarray,
    maxiter: int = 30,
) -> object:
    """
    **PDF C2 (pattern):** Nelder–Mead on a **scalar** objective returned by `mc_objective_fn(x)`.

    Here `x` is an abstract vector (e.g. 9 spline knots). The caller supplies
    `mc_objective_fn` that runs MC and returns **mean score**; we minimize **negative** mean.
    """
    fun = lambda x: -float(mc_objective_fn(x))
    return minimize(fun, x0_9params, method="Nelder-Mead", options={"maxiter": maxiter, "disp": False})
