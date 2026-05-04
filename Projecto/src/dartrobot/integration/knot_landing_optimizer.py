"""
integration_knot_landing_optimizer_SPEC.py
=========================================
Refine **12 spline knots** with **scipy.optimize** so a deterministic MuJoCo throw lands closer to
a board-plane target ``(Δy, Δz)`` and, with a strong penalty, on the **dartboard face**
(``r ≤ R_BOARD_MISS_MM``).

Used by ``run_target_score_monte_carlo_SPEC.py`` with ``--optimize-knots`` before Monte Carlo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from dartrobot.constants import R_BOARD_MISS_MM
from dartrobot.integration.jacobian_covariance import (
    landing_info_for_release_SPEC,
)
from dartrobot.integration.release_to_score import (
    score_from_release_state_SPEC,
)
from dartrobot.integration.rl_env_scaffold import clip_knots_action_SPEC
from dartrobot.motion.controller import simulate_throw_mujoco_SPEC


def _landing_loss_for_knots_SPEC(
    knots12: np.ndarray,
    target_dy_m: float,
    target_dz_m: float,
    q_start4: np.ndarray,
    *,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    plane_miss_penalty: float = 2.5e4,
    off_face_penalty_coeff: float = 120.0,
    mujoco_kwargs: dict[str, Any] | None = None,
) -> float:
    """Scalar loss: L2 to target plus strong off-face / plane-miss penalties."""
    mk = mujoco_kwargs or {}
    knots12 = clip_knots_action_SPEC(np.asarray(knots12, dtype=float).reshape(12))
    sim = simulate_throw_mujoco_SPEC(
        knots12,
        q_start4=np.asarray(q_start4, dtype=float).reshape(4),
        torque_noise=False,
        **mk,
    )
    s6 = sim.get("release_state6")
    if s6 is None:
        return float(plane_miss_penalty * 2.0)

    s6 = np.asarray(s6, dtype=float).reshape(6)
    info = landing_info_for_release_SPEC(s6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
    if not info["hit"]:
        return float(plane_miss_penalty)

    dy = float(info["delta_y_m"])
    dz = float(info["delta_z_m"])
    ey = dy - float(target_dy_m)
    ez = dz - float(target_dz_m)
    pos = float(ey * ey + ez * ez)
    r_mm = float(np.hypot(dy * 1000.0, dz * 1000.0))
    if r_mm > float(R_BOARD_MISS_MM):
        ex = (r_mm - float(R_BOARD_MISS_MM)) / 1000.0
        pos += float(off_face_penalty_coeff) * ex * ex
    return pos


def optimize_knots_for_board_target_SPEC(
    knots12_init: np.ndarray,
    target_dy_m: float,
    target_dz_m: float,
    q_start4: np.ndarray,
    *,
    maxiter: int = 120,
    n_restarts: int = 8,
    restart_sigma_rad: float = 0.12,
    restart_seed: int = 0,
    plane_miss_penalty: float = 2.5e4,
    off_face_penalty_coeff: float = 220.0,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    mujoco_kwargs: dict[str, Any] | None = None,
) -> dict:
    """
    Optimize **12 knots** (L-BFGS-B, joint-limit bounds) to reduce landing error vs
    ``(target_dy_m, target_dz_m)`` and to pull landings inside the regulation face when possible.

    Uses **multi-start** L-BFGS-B (warm start plus small Gaussian perturbations) to escape poor
    plateaus from the IK/heuristic initializer.

    Returns ``knots12`` clipped, ``success``, ``fun``, ``n_restarts_used``, aggregate ``nfev``.
    """
    from dartrobot.integration.rl_env_scaffold import knots_action_bounds_SPEC

    x0 = clip_knots_action_SPEC(np.asarray(knots12_init, dtype=float).reshape(12))
    low, high = knots_action_bounds_SPEC()
    bounds = [(float(low[i]), float(high[i])) for i in range(12)]

    mk = dict(mujoco_kwargs or {})

    def fun(x: np.ndarray) -> float:
        return _landing_loss_for_knots_SPEC(
            x,
            target_dy_m,
            target_dz_m,
            q_start4,
            wind_xyz=wind_xyz,
            drag_enabled=drag_enabled,
            plane_miss_penalty=plane_miss_penalty,
            off_face_penalty_coeff=off_face_penalty_coeff,
            mujoco_kwargs=mk,
        )

    rng = np.random.default_rng(int(restart_seed))
    best_x = x0.copy()
    best_f = float(fun(best_x))
    total_nfev = 1
    n_used = 1
    for r in range(max(1, int(n_restarts))):
        if r == 0:
            x_try = x0.copy()
        else:
            noise = rng.normal(0.0, float(restart_sigma_rad), size=12)
            x_try = clip_knots_action_SPEC(x0 + noise)
        out = minimize(
            fun,
            x0=x_try,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "disp": False, "ftol": 1e-10},
        )
        total_nfev += int(getattr(out, "nfev", 0) or 0)
        n_used += 1
        xf = float(out.fun)
        if xf < best_f:
            best_f = xf
            best_x = clip_knots_action_SPEC(np.asarray(out.x, dtype=float).reshape(12))
    x_opt = best_x
    return {
        "knots12": x_opt,
        "success": bool(best_f < plane_miss_penalty * 0.5),
        "message": "multistart_lbfgsb",
        "fun": float(best_f),
        "nit": int(n_used),
        "nfev": int(total_nfev),
        "n_restarts_used": int(n_restarts),
    }


def evaluate_knots_landing_diagnostics_SPEC(
    knots12: np.ndarray,
    q_start4: np.ndarray,
    target_dy_m: float,
    target_dz_m: float,
    *,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    mujoco_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Deterministic rollout: plane hit, face hit, score, deltas (for logging)."""
    mk = dict(mujoco_kwargs or {})
    k = clip_knots_action_SPEC(np.asarray(knots12, dtype=float).reshape(12))
    sim = simulate_throw_mujoco_SPEC(
        k,
        q_start4=np.asarray(q_start4, dtype=float).reshape(4),
        torque_noise=False,
        **mk,
    )
    s6 = sim.get("release_state6")
    if s6 is None:
        return {
            "plane_hit": False,
            "dartboard_face_hit": False,
            "score": 0,
            "delta_y_m": float("nan"),
            "delta_z_m": float("nan"),
            "r_mm": float("nan"),
        }
    s6 = np.asarray(s6, dtype=float).reshape(6)
    out = score_from_release_state_SPEC(s6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
    plane = bool(out.get("hit"))
    dy = float(out.get("delta_y_m", float("nan")))
    dz = float(out.get("delta_z_m", float("nan")))
    r_mm = float(np.hypot(dy * 1000.0, dz * 1000.0)) if plane and np.isfinite(dy) and np.isfinite(dz) else float("nan")
    face = plane and np.isfinite(r_mm) and (r_mm <= float(R_BOARD_MISS_MM))
    return {
        "plane_hit": plane,
        "dartboard_face_hit": bool(face),
        "score": int(out.get("score", 0)),
        "delta_y_m": dy,
        "delta_z_m": dz,
        "r_mm": r_mm,
        "l2_err_m2": float((dy - target_dy_m) ** 2 + (dz - target_dz_m) ** 2) if plane else float("nan"),
    }
