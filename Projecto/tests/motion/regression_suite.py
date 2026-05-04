"""
VALIDATION_motion_regression_suite_SPEC.py
======================================
Assertion-based regression checks for Track A3 MuJoCo throw dynamics (4-DOF arm).

Run: `pytest tests/motion/regression_suite.py -q` (or `dartrobot demo` includes related checks).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from dartrobot.paths import mjcf_path_SPEC

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
    TORQUE_LIMIT_ELBOW_NM,
    TORQUE_LIMIT_SHOULDER_NM,
    TORQUE_LIMIT_SHOULDER_YAW_NM,
    TORQUE_LIMIT_WRIST_NM,
)
from dartrobot.motion.link_geometry import joint_limits_rad_SPEC
from dartrobot.motion.controller import (
    DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
    DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
    release_site_fk_SPEC,
    release_site_fk_vel_SPEC,
    simulate_throw_mujoco_SPEC,
)

_XML_PATH = mjcf_path_SPEC("arm_4dof.xml")
_Q_START = np.radians(
    [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
)

# Golden rollout reference — 4-DOF + motion_mujoco_mjcf_4dof_arm_SPEC.xml (seed 1234).
_GOLDEN_RELEASE6 = np.array([0.16394, -0.011557, 1.983239, 5.159944, -0.376052, -0.014489], dtype=float)
_GOLDEN_FINAL_Q4 = np.array([-0.044446, -0.125946, -0.052688, 0.295891], dtype=float)


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
        d.qpos[:4] = q
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
        qdot = rng.uniform(-4.0, 4.0, size=4).astype(float)
        d.qpos[:4] = q
        d.qvel[:4] = qdot
        if m.nv > 4:
            d.qvel[4:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_jacSite(m, d, jacp, None, release_id)
        vel_mj = jacp[:, :4] @ d.qvel[:4]
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
    """Deterministic rollout regression plus actuator feasibility summary."""
    out = simulate_throw_mujoco_SPEC(
        DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
        q_start4=_Q_START.copy(),
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
    limits = np.array(
        [
            TORQUE_LIMIT_SHOULDER_YAW_NM,
            TORQUE_LIMIT_SHOULDER_NM,
            TORQUE_LIMIT_ELBOW_NM,
            TORQUE_LIMIT_WRIST_NM,
        ],
        dtype=float,
    )

    rel_release = float(np.linalg.norm(s6 - _GOLDEN_RELEASE6) / max(np.linalg.norm(_GOLDEN_RELEASE6), 1e-8))
    rel_q = float(np.linalg.norm(q_final - _GOLDEN_FINAL_Q4) / max(np.linalg.norm(_GOLDEN_FINAL_Q4), 1e-8))
    sat_ratio = np.mean(np.abs(tau) >= limits[None, :], axis=0) if tau.size else np.zeros(4)
    max_tau = np.max(np.abs(tau), axis=0) if tau.size else np.zeros(4)

    assert s6[3] > 0.0, f"Expected forward release vx>0, got {s6[3]:.3f} m/s"
    assert rel_release <= rel_tol_release, (
        f"Golden release drift too high: rel={rel_release:.4f} > {rel_tol_release:.4f}"
    )
    assert rel_q <= rel_tol_final_q, f"Final-q drift too high: rel={rel_q:.4f} > {rel_tol_final_q:.4f}"
    assert float(np.max(sat_ratio)) <= 0.80, (
        "Actuator saturation ratio unexpectedly high; controller may be unstable. "
        f"max ratio={float(np.max(sat_ratio)):.2%}"
    )

    print(f"[A3 Golden] relative release drift = {rel_release:.4f}")
    print(f"[A3 Golden] relative final-q drift = {rel_q:.4f}")
    print(
        "[A3 Torque] max |tau| (Nm) yaw/shoulder/elbow/wrist = "
        f"{max_tau[0]:.2f}, {max_tau[1]:.2f}, {max_tau[2]:.2f}, {max_tau[3]:.2f}"
    )
    print(
        "[A3 Torque] saturation ratio yaw/shoulder/elbow/wrist = "
        f"{sat_ratio[0]:.2%}, {sat_ratio[1]:.2%}, {sat_ratio[2]:.2%}, {sat_ratio[3]:.2%}"
    )
    return {
        "release_state6": s6,
        "q_final": q_final,
        "sat_ratio": sat_ratio,
        "max_tau": max_tau,
        "rel_release": rel_release,
        "rel_q": rel_q,
    }


def test_motion_regression_suite_SPEC() -> None:
    fk_position_parity_SPEC()
    fk_velocity_parity_SPEC()
    golden_rollout_and_torque_feasibility_SPEC()


if __name__ == "__main__":
    print("--- A3 assertion-based regression suite ---")
    test_motion_regression_suite_SPEC()
    print("A3 regression suite passed.")
