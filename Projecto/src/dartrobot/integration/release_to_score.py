"""
integration_release_to_score_pipeline_SPEC.py
====================================================
**PDF section:** Track C — C1 (End-to-end pipeline)

**Purpose:** Connect **arm release state** (6D: position + velocity at fingertip / `release_site`)
to the **projectile integrator** (**B2**) and **dartboard scoring** (**B3**).

**PDF C1 bullets:**
- Wire arm release (x,y,z,vx,vy,vz) at time t_r into projectile initial condition.
- Planar arm ⇒ **y = 0, vy = 0** at release in nominal model (PDF).
- Run: throw trajectory → release → flight → landing → score.
- Monte Carlo: N=200 noisy throws; mean score, scatter, covariance (**see also C2 / validation**).

This module is **pure wiring** + small helpers so RL / analysis code imports one place.
"""

from __future__ import annotations

import numpy as np

from dartrobot.flight.integrator import integrate_until_board_SPEC
from dartrobot.flight.scoring import score_from_deltas_SPEC


def score_from_release_state_SPEC(release_state6: np.ndarray, wind_xyz=(0.0, 0.0, 0.0), drag_enabled: bool = True) -> dict:
    """
    **End-to-end (single rollout):** integrate projectile from `release_state6`, return score + diagnostics.

    **PDF coordinate note:** `release_state6` must be in world frame
    (x toward board, y left, z up) with x0 < 2.37 typically.
    """
    release_state6 = np.asarray(release_state6, dtype=float).reshape(6)
    hit_info = integrate_until_board_SPEC(release_state6, wind_xyz_mps=wind_xyz, drag_enabled=drag_enabled)
    if not hit_info["hit"]:
        return {"score": 0, "hit": False, "delta_y_m": 0.0, "delta_z_m": 0.0, "hit_info": hit_info}
    dy = hit_info["delta_y_m"]
    dz = hit_info["delta_z_m"]
    return {
        "score": score_from_deltas_SPEC(dy, dz),
        "hit": True,
        "delta_y_m": dy,
        "delta_z_m": dz,
        "hit_info": hit_info,
    }


def monte_carlo_expected_score_SPEC(
    release_sampler,
    n_rollouts: int = 200,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    seed: int | None = None,
) -> dict:
    """
    **PDF C1:** Monte Carlo over noisy throws.

    `release_sampler(i, rng) -> (6,) release_state6` for rollout index i.
    Returns mean score, per-rollout scores, landing covariance estimate.
    """
    rng = np.random.default_rng(seed)
    scores = []
    landings = []
    for i in range(n_rollouts):
        s6 = release_sampler(i, rng)
        out = score_from_release_state_SPEC(s6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
        scores.append(out["score"])
        landings.append([out["delta_y_m"], out["delta_z_m"]])
    scores = np.asarray(scores, dtype=float)
    landings = np.asarray(landings, dtype=float)
    mean_score = float(np.mean(scores))
    cov = np.cov(landings.T) if landings.shape[0] > 1 else np.zeros((2, 2))
    return {
        "mean_score": mean_score,
        "scores": scores,
        "landings_m": landings,
        "covariance_deltay_deltaz": cov,
        "std_score": float(np.std(scores)),
    }


def evaluate_knots12_throw_once_SPEC(
    knots12: np.ndarray,
    q_start4: np.ndarray | None = None,
    *,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    **mujoco_kwargs,
) -> dict:
    """
    **C1 helper:** MuJoCo arm rollout from a **12-knot** spline policy → release → score.

    Return key ``hit`` is **True** when the analytic flight integrator finds a **forward crossing**
    of the board **plane** ``x = 2.37`` m (B2), not when the landing lies on the regulation
    dartboard **disk** (``r ≤ 170`` mm). Use ``score`` and ``(delta_y_m, delta_z_m)`` for face / ring
    outcomes.

    Extra `mujoco_kwargs` are forwarded to `simulate_throw_mujoco_SPEC` (e.g. `xml_path`,
    `torque_noise`, PD gains).
    """
    from dartrobot.motion.controller import simulate_throw_mujoco_SPEC
    from dartrobot.constants import (
        KEYFRAME_ELBOW_DEG,
        KEYFRAME_SHOULDER_DEG,
        KEYFRAME_SHOULDER_YAW_DEG,
        KEYFRAME_WRIST_DEG,
    )

    if q_start4 is None:
        q_start4 = np.radians(
            [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
        )
    sim = simulate_throw_mujoco_SPEC(
        np.asarray(knots12, dtype=float).reshape(12),
        q_start4=np.asarray(q_start4, dtype=float).reshape(4),
        **mujoco_kwargs,
    )
    s6 = sim["release_state6"]
    if s6 is None:
        return {
            "score": 0,
            "hit": False,
            "delta_y_m": np.nan,
            "delta_z_m": np.nan,
            "release_state6": None,
            "sim": sim,
        }
    out = score_from_release_state_SPEC(s6, wind_xyz=wind_xyz, drag_enabled=drag_enabled)
    return {**out, "release_state6": np.asarray(s6, dtype=float).reshape(6), "sim": sim}


def evaluate_ilqr_target_pack_mc_SPEC(
    ilqr_per_sector: dict,
    *,
    q_start4: np.ndarray | None = None,
    n_rollouts: int = 24,
    seed: int = 0,
    torque_noise: bool = True,
    **mujoco_kwargs,
) -> dict:
    """
    Monte Carlo over torque noise for each sector's iLQR-produced `knots12`.

    `ilqr_per_sector` maps `sector -> dict` containing at least `"knots12"`.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for seg, payload in ilqr_per_sector.items():
        knots12 = np.asarray(payload["knots12"], dtype=float).reshape(12)
        scores = []
        dys = []
        dzs = []
        for _ in range(int(n_rollouts)):
            out = evaluate_knots12_throw_once_SPEC(
                knots12,
                q_start4=q_start4,
                rng=rng,
                torque_noise=torque_noise,
                **mujoco_kwargs,
            )
            sc = float(out["score"])
            scores.append(sc)
            # Radial error only for **scoring** landings (avoids huge hypot when the dart
            # crosses x=2.37 far off-board but still gets hit=True from the plane event).
            if sc > 0.0 and np.isfinite(out.get("delta_y_m", np.nan)) and np.isfinite(out.get("delta_z_m", np.nan)):
                dys.append(float(out["delta_y_m"]))
                dzs.append(float(out["delta_z_m"]))
        hit_rate = float(np.mean(np.asarray(scores) > 0.0)) if scores else 0.0
        if dys:
            err = np.hypot(np.asarray(dys), np.asarray(dzs))
            mean_rad_mm = float(np.mean(err) * 1000.0)
        else:
            mean_rad_mm = float("nan")
        rows[int(seg)] = {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "hit_rate": hit_rate,
            "mean_radial_error_mm": mean_rad_mm,
            "scores": np.asarray(scores, dtype=float),
        }
    return {"per_sector": rows, "n_rollouts": int(n_rollouts)}
