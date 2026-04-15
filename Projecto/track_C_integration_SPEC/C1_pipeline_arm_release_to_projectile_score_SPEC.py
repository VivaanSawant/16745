"""
C1_pipeline_arm_release_to_projectile_score_SPEC.py
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

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from track_B_projectile_SPEC.B2_ode_integrator_event_board_plane_SPEC import integrate_until_board_SPEC
from track_B_projectile_SPEC.B3_dartboard_scoring_radial_angular_SPEC import score_from_deltas_SPEC


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
