"""
integration_ilqr_motion_targeting_SPEC.py
=========================================
**Motion targeting** for the **4-DOF** arm toward board-plane `(Δy, Δz)` objectives.

**Implementation note (MuJoCo-aligned):** A toy double-integrator LQR trajectory does **not**
match the cubic-spline + PD rollout used in `simulate_throw_mujoco_SPEC`, and previously
produced **near-static** terminal postures (~0.5 m/s release). The primary path now:

1. Map `(Δy, Δz)` → soft fingertip workspace goal (`workspace_goal_xyz_from_board_deltas_SPEC`).
2. **IK** (`_ik_ref_q4_for_position_SPEC`) → joint posture `q_ik`.
3. **Blend** `q_ik` with the **joint-space endpoint** taken from the tuned overhand preset
   (`DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC`, last knot per joint), preserving dart-like
   motion magnitude while steering aim.
4. **Minimum-jerk knots** from cocked `q_start` → blended `q_goal`.
5. Optional **MuJoCo refinement**: if forward release speed is low, reduce IK influence
   (move blend toward the overhand nominal).
6. Optional **landing retarget** (``landing_retarget_iters``): after knots exist, compare
   deterministic board-plane ``(Δy, Δz)`` to the objective and nudge the workspace goal in a
   short loop (heuristic map from board error to hand position → re-IK → same blend). Each
   nudge in the hand ``y,z`` plane is **capped** in magnitude so large board misses (meters)
   do not move the workspace goal by meters per iteration.

"""

from __future__ import annotations

import numpy as np

# Max workspace (hand y/z) correction per retarget iteration — avoids exploding p_goal when
# board-plane error is meters (heuristic was tuned for ~15 mm convergence).
_LANDING_RETARGET_MAX_WS_STEP_M = 0.025

from dartrobot.motion.link_geometry import clamp_joint_vector_SPEC
from dartrobot.motion.controller import (
    DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
    DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
    nominal_minimum_jerk_knots_SPEC,
    release_site_fk_SPEC,
    release_site_position_jacobian_wrt_q_SPEC,
    simulate_throw_mujoco_SPEC,
)
from dartrobot.integration.release_to_score import (
    evaluate_knots12_throw_once_SPEC,
)


def _ik_ref_q4_for_position_SPEC(
    p_goal_world3: np.ndarray,
    q_seed4: np.ndarray,
    *,
    steps: int = 14,
    alpha: float = 0.38,
) -> np.ndarray:
    """Gauss–Newton style IK in joint space (position-only)."""
    q = np.asarray(q_seed4, dtype=float).reshape(4).copy()
    target = np.asarray(p_goal_world3, dtype=float).reshape(3)
    for _ in range(int(steps)):
        p_cur, _ = release_site_fk_SPEC(q)
        J = release_site_position_jacobian_wrt_q_SPEC(q)
        dq = alpha * np.linalg.pinv(J, rcond=1e-4) @ (target - p_cur)
        q = clamp_joint_vector_SPEC(q + dq)
    return q


def _overhand_terminal_joint_posture4_SPEC() -> np.ndarray:
    """Last Bezier knot per joint from the tuned overhand spline preset (rad)."""
    k = np.asarray(DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC, dtype=float).reshape(12)
    return np.array([k[j * 3 + 2] for j in range(4)], dtype=float)


def solve_ilqr_motion_knots_for_board_deltas_SPEC(
    q_start4: np.ndarray,
    delta_y_m: float,
    delta_z_m: float,
    *,
    blend_ik_with_overhand: float = 0.42,
    refine_with_mujoco: bool = True,
    refine_min_release_vx_mps: float = 4.0,
    refine_max_iters: int = 6,
    landing_retarget_iters: int = 6,
    workspace_goal_fn=None,
) -> dict:
    """
    Build **12 knots** aiming toward `(Δy, Δz)` using **IK + overhand nominal blend**
    (MuJoCo-validated design path).

    `blend_ik_with_overhand` ∈ [0, 1]: **1** = pure IK posture, **0** = pure overhand terminal
    joints (strong forward throw, weak aim correction). Typical **0.35–0.50**.

    `landing_retarget_iters`: run up to this many MuJoCo evaluations to correct workspace goal
    from board-space error (set **0** to disable).
    """
    if workspace_goal_fn is None:
        from dartrobot.integration.target_board_regions import (
            workspace_goal_xyz_from_board_deltas_SPEC,
        )

        workspace_goal_fn = workspace_goal_xyz_from_board_deltas_SPEC

    q0 = np.asarray(q_start4, dtype=float).reshape(4)
    p_goal = np.asarray(workspace_goal_fn(float(delta_y_m), float(delta_z_m)), dtype=float).reshape(3)

    q_nom = _overhand_terminal_joint_posture4_SPEC()
    q_ik = _ik_ref_q4_for_position_SPEC(p_goal, q0)

    blend = float(np.clip(blend_ik_with_overhand, 0.05, 0.85))

    def _blended_goal(b: float) -> np.ndarray:
        b = float(np.clip(b, 0.0, 1.0))
        q_mix = (1.0 - b) * q_nom + b * q_ik
        return clamp_joint_vector_SPEC(q_mix)

    q_goal4 = _blended_goal(blend)
    knots12 = nominal_minimum_jerk_knots_SPEC(q0, q_goal4)
    blend_used = blend

    if refine_with_mujoco:
        b = blend
        for _ in range(int(refine_max_iters)):
            sim = simulate_throw_mujoco_SPEC(
                knots12,
                q_start4=q0,
                torque_noise=False,
                kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
                kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
                release_time_s=DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
            )
            s6 = sim["release_state6"]
            if s6 is None:
                b *= 0.86
                q_goal4 = _blended_goal(b)
                knots12 = nominal_minimum_jerk_knots_SPEC(q0, q_goal4)
                continue
            if float(s6[3]) >= float(refine_min_release_vx_mps):
                break
            b *= 0.88
            if b < 0.08:
                b = 0.08
            q_goal4 = _blended_goal(b)
            knots12 = nominal_minimum_jerk_knots_SPEC(q0, q_goal4)
        blend_used = float(b)

    # --- Optional: nudge workspace goal using MuJoCo landing error (board Δy, Δz) ---
    p_goal_work = p_goal.copy()
    for _ret_i in range(max(0, int(landing_retarget_iters))):
        ev = evaluate_knots12_throw_once_SPEC(
            knots12,
            q_start4=q0,
            torque_noise=False,
            kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            release_time_s=DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        )
        if not ev.get("hit"):
            break
        ay = float(ev["delta_y_m"])
        az = float(ev["delta_z_m"])
        ey = float(delta_y_m) - ay
        ez = float(delta_z_m) - az
        if abs(ey) < 0.015 and abs(ez) < 0.015:
            break
        # Heuristic map from board error to workspace goal (hand y/z); works with IK + blend.
        raw_delta = np.array([0.0, 1.05 * ey, 1.35 * ez], dtype=float)
        delta_ws = raw_delta.copy()
        step_norm = float(np.linalg.norm(delta_ws[1:3]))
        if step_norm > float(_LANDING_RETARGET_MAX_WS_STEP_M) and step_norm > 0.0:
            delta_ws[1:3] *= float(_LANDING_RETARGET_MAX_WS_STEP_M) / step_norm
        p_goal_work = p_goal_work + delta_ws
        q_ik = _ik_ref_q4_for_position_SPEC(p_goal_work, q0)
        q_goal4 = _blended_goal(blend_used)
        knots12 = nominal_minimum_jerk_knots_SPEC(q0, q_goal4)

    p_end, _ = release_site_fk_SPEC(q_goal4)

    return {
        "knots12": knots12,
        "q_terminal_ik4": q_ik.copy(),
        "q_terminal_overhand4": q_nom.copy(),
        "q_goal_blended4": q_goal4.copy(),
        "blend_ik_with_overhand_used": float(blend_used),
        "workspace_goal_xyz_m": p_goal.copy(),
        "workspace_goal_xyz_after_retarget_m": p_goal_work.copy(),
        "p_release_fk_xyz_m": p_end.copy(),
        "delta_y_m": float(delta_y_m),
        "delta_z_m": float(delta_z_m),
    }


def knots12_with_ilqr_warmstart_for_targets_SPEC(
    q_start4: np.ndarray,
    stage_pack: dict,
    *,
    stage: str = "single",
) -> dict:
    """
    Convenience: produce a per-sector knot dictionary for either **single** or **treble** stage.

    `stage_pack` should be the dict returned by `stage_targets_single_then_treble_SPEC`.
    """
    if stage not in ("single", "treble"):
        raise ValueError("stage must be 'single' or 'treble'")
    key = "stage_single_deltas_m" if stage == "single" else "stage_treble_deltas_m"
    targets = stage_pack[key]
    out = {}
    for seg, (dy, dz) in targets.items():
        out[int(seg)] = solve_ilqr_motion_knots_for_board_deltas_SPEC(q_start4, dy, dz)
    return {"stage": stage, "per_sector": out}
