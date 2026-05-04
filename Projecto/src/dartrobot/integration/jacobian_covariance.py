"""
integration_jacobian_covariance_optimizer_SPEC.py
==========================================
**PDF section:** Track C — C2 (Analysis layers)

**Jacobian (PDF):** ∂(landing Δy, Δz) / ∂(release state) with release state the **6 scalars**
(x0, y0, z0, vx0, vy0, vz0) at release — computed here with **finite differences** (ε small).

**Predicted landing covariance (PDF):** Σ_land ≈ J Σ_release Jᵀ (first-order), compare to MC scatter.

**Optimizer (PDF):** `scipy.optimize.minimize` (Nelder–Mead) with objective = **negative expected score**
from Monte Carlo — **stub** provided; warm-start and target generalization are project extensions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from dartrobot.flight.integrator import integrate_until_board_SPEC


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


def solve_release_velocity_for_target_SPEC(
    release_position_xyz: np.ndarray,
    target_deltas_yz_m: np.ndarray,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    speed_bounds_mps: tuple[float, float] = (2.0, 25.0),
    x_velocity_floor_mps: float = 0.5,
    v0_guess_xyz: np.ndarray | None = None,
) -> dict:
    """
    Inverse aiming in release space: solve for velocity that lands near target (Δy, Δz).
    """
    p = np.asarray(release_position_xyz, dtype=float).reshape(3)
    target = np.asarray(target_deltas_yz_m, dtype=float).reshape(2)
    if v0_guess_xyz is None:
        v0_guess_xyz = np.array([6.0, target[0] * 2.0, 2.5], dtype=float)
    v0_guess_xyz = np.asarray(v0_guess_xyz, dtype=float).reshape(3)
    vmin, vmax = float(speed_bounds_mps[0]), float(speed_bounds_mps[1])

    def objective(vxyz: np.ndarray) -> float:
        vxyz = np.asarray(vxyz, dtype=float).reshape(3)
        s6 = np.concatenate([p, vxyz])
        info = landing_info_for_release_SPEC(s6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
        if not info["hit"]:
            # Misses are strongly penalized so optimizer prefers board-crossing states.
            return 1e4 + 10.0 * float(np.sum(vxyz**2))
        err = info["landing_m"] - target
        speed = float(np.linalg.norm(vxyz))
        speed_penalty = 0.0
        if speed < vmin:
            speed_penalty += 30.0 * (vmin - speed) ** 2
        if speed > vmax:
            speed_penalty += 30.0 * (speed - vmax) ** 2
        if vxyz[0] < x_velocity_floor_mps:
            speed_penalty += 50.0 * (x_velocity_floor_mps - vxyz[0]) ** 2
        return float(np.sum(err**2) + speed_penalty)

    out = minimize(
        objective,
        x0=v0_guess_xyz,
        method="Nelder-Mead",
        options={"maxiter": 200, "disp": False},
    )
    v_opt = np.asarray(out.x, dtype=float).reshape(3)
    s6_opt = np.concatenate([p, v_opt])
    landing = landing_info_for_release_SPEC(s6_opt, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
    return {
        "success": bool(out.success),
        "message": str(out.message),
        "release_state6": s6_opt,
        "velocity_xyz_mps": v_opt,
        "landing_m": landing["landing_m"],
        "hit": bool(landing["hit"]),
        "objective_value": float(out.fun),
        "iterations": int(out.nit),
    }


def optimize_release_state_robust_score_SPEC(
    nominal_release6: np.ndarray,
    Sigma_release6: np.ndarray,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    n_mc_samples: int = 48,
    seed: int = 0,
    score_fn=None,
    risk_lambda: float = 0.25,
) -> dict:
    """
    Robust release-state tuning with Jacobian covariance + MC score refinement.
    """
    nominal_release6 = np.asarray(nominal_release6, dtype=float).reshape(6)
    Sigma_release6 = np.asarray(Sigma_release6, dtype=float).reshape(6, 6)
    if score_fn is None:
        from dartrobot.flight.scoring import score_from_deltas_SPEC

        score_fn = score_from_deltas_SPEC
    rng = np.random.default_rng(seed)

    J = jacobian_landing_wrt_release_SPEC(nominal_release6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
    Sigma_land_pred = predicted_landing_covariance_SPEC(J, Sigma_release6)

    def sampled_score(s6_center: np.ndarray) -> tuple[float, float, float]:
        samples = rng.multivariate_normal(s6_center, Sigma_release6, size=max(2, n_mc_samples))
        scores = []
        hit_count = 0
        for s6 in samples:
            info = landing_info_for_release_SPEC(s6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
            if not info["hit"]:
                scores.append(0.0)
                continue
            hit_count += 1
            scores.append(float(score_fn(float(info["delta_y_m"]), float(info["delta_z_m"]))))
        scores = np.asarray(scores, dtype=float)
        return float(np.mean(scores)), float(np.std(scores)), float(hit_count / len(scores))

    def objective(s6_center: np.ndarray) -> float:
        s6_center = np.asarray(s6_center, dtype=float).reshape(6)
        mean_score, std_score, _ = sampled_score(s6_center)
        # Maximize mean score while reducing variability and keeping states near nominal.
        reg = 0.01 * float(np.sum((s6_center - nominal_release6) ** 2))
        return -mean_score + risk_lambda * std_score + reg

    out = minimize(
        objective,
        x0=nominal_release6,
        method="Nelder-Mead",
        options={"maxiter": 120, "disp": False},
    )
    s6_opt = np.asarray(out.x, dtype=float).reshape(6)
    mean_score, std_score, hit_rate = sampled_score(s6_opt)
    return {
        "success": bool(out.success),
        "message": str(out.message),
        "release_state6_opt": s6_opt,
        "objective_value": float(out.fun),
        "mean_score_mc": mean_score,
        "std_score_mc": std_score,
        "hit_rate_mc": hit_rate,
        "jacobian_2x6_nominal": J,
        "Sigma_land_predicted_2x2_nominal": Sigma_land_pred,
    }


def minimize_negative_mc_score_stub_SPEC(
    mc_objective_fn,
    x0_12params: np.ndarray,
    maxiter: int = 30,
) -> object:
    """
    **PDF C2 (pattern):** Nelder–Mead on a **scalar** objective returned by `mc_objective_fn(x)`.

    Here `x` is an abstract vector (e.g. 12 spline knots for the 4-DOF arm). The caller supplies
    `mc_objective_fn` that runs MC and returns **mean score**; we minimize **negative** mean.
    """
    fun = lambda x: -float(mc_objective_fn(x))
    return minimize(fun, x0_12params, method="Nelder-Mead", options={"maxiter": maxiter, "disp": False})
