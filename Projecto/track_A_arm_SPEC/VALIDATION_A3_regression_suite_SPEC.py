"""
VALIDATION_A3_regression_suite_SPEC.py
======================================
Assertion-based regression checks for Track A3 MuJoCo throw dynamics.

Run: `python track_A_arm_SPEC/VALIDATION_A3_regression_suite_SPEC.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_WRIST_DEG,
    TORQUE_LIMIT_ELBOW_NM,
    TORQUE_LIMIT_SHOULDER_NM,
    TORQUE_LIMIT_WRIST_NM,
)
from track_A_arm_SPEC.A1_link_geometry_and_inertia_SPEC import joint_limits_rad_SPEC
from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
    DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
    release_site_fk_SPEC,
    release_site_fk_vel_SPEC,
    simulate_throw_mujoco_SPEC,
)

_XML_PATH = _ROOT / "track_A_arm_SPEC" / "A2_mujoco_mjcf_3link_arm_SPEC.xml"
_Q_START = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])

# Golden rollout reference from deterministic seed/settings below.
_GOLDEN_RELEASE6 = np.array([0.164205, 0.0, 1.983236, 5.174755, 0.0, -0.016401], dtype=float)
_GOLDEN_FINAL_Q3 = np.array([-0.125825, -0.053132, 0.301275], dtype=float)


def fk_position_parity_SPEC(n_samples: int = 128, tol_m: float = 2.0e-3, seed: int = 11) -> float:
    """Check FK release_site position agreement between Python kinematics and MuJoCo."""
    rng = np.random.default_rng(seed)
    lim = joint_limits_rad_SPEC()
    m = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    d = mujoco.MjData(m)
    release_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "release_site")

    max_err = 0.0
    for _ in range(n_samples):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lim], dtype=float)
        d.qpos[:3] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        pos_mj = d.site_xpos[release_id].copy()
        pos_py, _ = release_site_fk_SPEC(q)
        err = float(np.linalg.norm(pos_mj - pos_py, ord=np.inf))
        max_err = max(max_err, err)
    assert max_err < tol_m, f"FK position parity failed: max inf-norm error {max_err:.6f} m >= {tol_m:.6f} m"
    print(f"[A3 FK parity] max |pos_mj - pos_py|_inf = {max_err:.6f} m")
    return max_err


def fk_velocity_parity_SPEC(n_samples: int = 128, tol_mps: float = 2.5e-3, seed: int = 17) -> float:
    """Check release velocity agreement: MuJoCo Jacobian*qdot vs analytic FK velocity."""
    rng = np.random.default_rng(seed)
    lim = joint_limits_rad_SPEC()
    m = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    d = mujoco.MjData(m)
    release_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "release_site")
    jacp = np.zeros((3, m.nv))

    max_err = 0.0
    for _ in range(n_samples):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lim], dtype=float)
        qdot = rng.uniform(-4.0, 4.0, size=3).astype(float)
        d.qpos[:3] = q
        d.qvel[:3] = qdot
        if m.nv > 3:
            d.qvel[3:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_jacSite(m, d, jacp, None, release_id)
        vel_mj = jacp[:, :3] @ d.qvel[:3]
        vel_py = release_site_fk_vel_SPEC(q, qdot)
        err = float(np.linalg.norm(vel_mj - vel_py, ord=np.inf))
        max_err = max(max_err, err)
    assert max_err < tol_mps, f"FK velocity parity failed: max inf-norm error {max_err:.6f} m/s >= {tol_mps:.6f} m/s"
    print(f"[A3 Jacobian parity] max |vel_mj - vel_py|_inf = {max_err:.6f} m/s")
    return max_err


def golden_rollout_and_torque_feasibility_SPEC(
    rel_tol_release: float = 2.0e-2,
    rel_tol_final_q: float = 2.0e-2,
) -> dict:
    """
    Deterministic rollout regression plus actuator feasibility summary.
    """
    out = simulate_throw_mujoco_SPEC(
        DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
        q_start3=_Q_START.copy(),
        xml_path=_XML_PATH,
        rng=np.random.default_rng(1234),
        torque_noise=False,
        release_time_s=DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        enforce_joint_limits=True,
    )
    s6 = np.asarray(out["release_state6"], dtype=float)
    q_final = np.asarray(out["q"][-1], dtype=float)
    tau = np.asarray(out["tau"], dtype=float)
    limits = np.array([TORQUE_LIMIT_SHOULDER_NM, TORQUE_LIMIT_ELBOW_NM, TORQUE_LIMIT_WRIST_NM], dtype=float)

    rel_release = float(np.linalg.norm(s6 - _GOLDEN_RELEASE6) / max(np.linalg.norm(_GOLDEN_RELEASE6), 1e-8))
    rel_q = float(np.linalg.norm(q_final - _GOLDEN_FINAL_Q3) / max(np.linalg.norm(_GOLDEN_FINAL_Q3), 1e-8))
    sat_ratio = np.mean(np.abs(tau) >= limits[None, :], axis=0) if tau.size else np.zeros(3)
    max_tau = np.max(np.abs(tau), axis=0) if tau.size else np.zeros(3)

    assert s6[3] > 0.0, f"Expected forward release vx>0, got {s6[3]:.3f} m/s"
    assert rel_release <= rel_tol_release, (
        f"Golden release drift too high: rel={rel_release:.4f} > {rel_tol_release:.4f}"
    )
    assert rel_q <= rel_tol_final_q, (
        f"Final-q drift too high: rel={rel_q:.4f} > {rel_tol_final_q:.4f}"
    )
    assert float(np.max(sat_ratio)) <= 0.80, (
        "Actuator saturation ratio unexpectedly high; controller may be unstable. "
        f"max ratio={float(np.max(sat_ratio)):.2%}"
    )

    print(f"[A3 Golden] relative release drift = {rel_release:.4f}")
    print(f"[A3 Golden] relative final-q drift = {rel_q:.4f}")
    print(
        "[A3 Torque] max |tau| (Nm) shoulder/elbow/wrist = "
        f"{max_tau[0]:.2f}, {max_tau[1]:.2f}, {max_tau[2]:.2f}"
    )
    print(
        "[A3 Torque] saturation ratio shoulder/elbow/wrist = "
        f"{sat_ratio[0]:.2%}, {sat_ratio[1]:.2%}, {sat_ratio[2]:.2%}"
    )
    return {
        "release_state6": s6,
        "q_final": q_final,
        "sat_ratio": sat_ratio,
        "max_tau": max_tau,
        "rel_release": rel_release,
        "rel_q": rel_q,
    }


if __name__ == "__main__":
    print("--- A3 assertion-based regression suite ---")
    fk_position_parity_SPEC()
    fk_velocity_parity_SPEC()
    golden_rollout_and_torque_feasibility_SPEC()
    print("A3 regression suite passed.")

