"""
motion_controller_cubic_spline_pd_SPEC.py
=========================================
**PDF section:** Track A — A3 (Trajectory and controller)

**Components (PDF + 4-DOF extension):**
1. **Cubic spline** per joint: **3 knots per joint** → **12 policy parameters** (shoulder yaw,
   shoulder, elbow, wrist). Starting posture is **fixed** (cocked keyframe); knots define the
   forward throw over ~**0.2 s**.
2. **PD control:** τ = Kp (q* − q) + Kd (q̇* − q̇) with default **Kp=100**, **Kd=10** per joint.
3. **Motor noise (PDF / Lawrence):** perturb τ with **additive Gaussian** + **multiplicative**
   noise proportional to |τ|.
4. **Release:** at time **t_r** (Gaussian jitter σ=0.01 s per PDF), read **release_site**
   position and velocity → **6D initial condition** for projectile (**Track B / C1**).

**Coordinate convention:** Matches **B1/B2** world frame (x toward board, y left, z up).

**FK convention:** Shoulder **yaw** about **world +z**, then three hinges about **local y**
(pitch chain) after yaw — consistent with `motion_mujoco_mjcf_4dof_arm_SPEC.xml`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dartrobot.constants import (
    FOREARM_LENGTH_M,
    HAND_LENGTH_M,
    PD_KP_DEFAULT,
    PD_KD_DEFAULT,
    RELEASE_TIME_SIGMA_S,
    SHOULDER_MOUNT_X_M,
    SHOULDER_MOUNT_Y_M,
    SHOULDER_MOUNT_Z_M,
    THROW_DURATION_S,
    TORQUE_LIMIT_ELBOW_NM,
    TORQUE_LIMIT_SHOULDER_NM,
    TORQUE_LIMIT_SHOULDER_YAW_NM,
    TORQUE_LIMIT_WRIST_NM,
    UPPER_ARM_LENGTH_M,
)
from dartrobot.motion.link_geometry import clamp_joint_vector_SPEC
from dartrobot.paths import mjcf_path_SPEC


def minimum_jerk_progress_SPEC(u: float) -> float:
    """Quintic minimum-jerk progress scalar in [0, 1]."""
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def minimum_jerk_progress_dot_SPEC(u: float) -> float:
    """d/du of minimum-jerk progress."""
    u = float(np.clip(u, 0.0, 1.0))
    return 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4


def minimum_jerk_progress_ddot_SPEC(u: float) -> float:
    """d²/du² of minimum-jerk progress."""
    u = float(np.clip(u, 0.0, 1.0))
    return 60.0 * u - 180.0 * u**2 + 120.0 * u**3


def minimum_jerk_joint_reference_SPEC(
    t_s: float,
    q_start4: np.ndarray,
    q_goal4: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-jerk reference (q, qdot, qddot) between two 4D joint postures."""
    T = max(float(duration_s), 1e-6)
    u = np.clip(float(t_s) / T, 0.0, 1.0)
    q_start4 = np.asarray(q_start4, dtype=float).reshape(4)
    q_goal4 = np.asarray(q_goal4, dtype=float).reshape(4)
    dq = q_goal4 - q_start4

    s = minimum_jerk_progress_SPEC(u)
    s_u = minimum_jerk_progress_dot_SPEC(u)
    s_uu = minimum_jerk_progress_ddot_SPEC(u)

    q = q_start4 + s * dq
    qd = (s_u / T) * dq
    qdd = (s_uu / (T**2)) * dq
    return q, qd, qdd


def nominal_minimum_jerk_knots_SPEC(
    q_start4: np.ndarray,
    q_goal4: np.ndarray,
) -> np.ndarray:
    """Build the 12D knot vector from a minimum-jerk profile sampled at 1/3, 2/3, and 1."""
    q_start4 = np.asarray(q_start4, dtype=float).reshape(4)
    q_goal4 = np.asarray(q_goal4, dtype=float).reshape(4)
    knots = []
    for joint_idx in range(4):
        q0 = q_start4[joint_idx]
        q1 = q_goal4[joint_idx]
        dq = q1 - q0
        for u in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            knots.append(q0 + minimum_jerk_progress_SPEC(u) * dq)
    return np.asarray(knots, dtype=float)


def plan_nominal_throw_knots_min_jerk_SPEC(
    q_start4: np.ndarray,
    q_goal4: np.ndarray | None = None,
    alpha_goal: float = 0.25,
) -> dict:
    """
    Create a minimum-jerk nominal throw in knot space for warm starts / OC scaffolding.
    """
    q_start4 = np.asarray(q_start4, dtype=float).reshape(4)
    if q_goal4 is None:
        preset_goal = DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC.reshape(4, 3)[:, -1]
        q_goal4 = (1.0 - alpha_goal) * q_start4 + alpha_goal * preset_goal
    else:
        q_goal4 = np.asarray(q_goal4, dtype=float).reshape(4)
    knots12 = nominal_minimum_jerk_knots_SPEC(q_start4, q_goal4)
    return {
        "knots12": knots12,
        "q_start4": q_start4.copy(),
        "q_goal4": q_goal4.copy(),
    }


def _Rz_mat(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _dRz_dangle(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]], dtype=float)


def _Ry_mat(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _dRy_dangle(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[-s, 0.0, c], [0.0, 0.0, 0.0], [-c, 0.0, -s]], dtype=float)


def _unit_x_world_from_yaw_pitch(q_yaw: float, Q_pitch: float) -> np.ndarray:
    """World unit vector along a link segment: Rz(q_yaw) @ Ry(Q_pitch) @ e_x."""
    return (_Rz_mat(q_yaw) @ _Ry_mat(Q_pitch) @ np.array([1.0, 0.0, 0.0])).ravel()


def _dunit_x_d_yaw(q_yaw: float, Q_pitch: float) -> np.ndarray:
    return (_dRz_dangle(q_yaw) @ _Ry_mat(Q_pitch) @ np.array([1.0, 0.0, 0.0])).ravel()


def _dunit_x_d_pitch(q_yaw: float, Q_pitch: float) -> np.ndarray:
    return (_Rz_mat(q_yaw) @ _dRy_dangle(Q_pitch) @ np.array([1.0, 0.0, 0.0])).ravel()


def release_site_fk_SPEC(q4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics for fingertip **release_site** (world m).

    q4 = [q_yaw, q_shoulder, q_elbow, q_wrist] (rad). Pitch angles are **relative**;
    segment world directions use cumulative pitch Q = q_sh, q_sh+q_el, q_sh+q_el+q_wr.
    """
    q4 = np.asarray(q4, dtype=float).reshape(4)
    qy, qs, qe, qw = float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3])
    o = np.array([SHOULDER_MOUNT_X_M, SHOULDER_MOUNT_Y_M, SHOULDER_MOUNT_Z_M], dtype=float)
    Q0 = qs
    Q1 = qs + qe
    Q2 = qs + qe + qw
    p = (
        o
        + UPPER_ARM_LENGTH_M * _unit_x_world_from_yaw_pitch(qy, Q0)
        + FOREARM_LENGTH_M * _unit_x_world_from_yaw_pitch(qy, Q1)
        + HAND_LENGTH_M * _unit_x_world_from_yaw_pitch(qy, Q2)
    )
    return p, np.zeros(3)


def release_chain_positions_world_SPEC(q4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return **world-frame** (m) positions: shoulder mount, elbow, wrist joint, fingertip/release.

    Useful for A4 stick figures and debugging FK vs MuJoCo.
    """
    q4 = np.asarray(q4, dtype=float).reshape(4)
    qy, qs, qe, qw = float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3])
    o = np.array([SHOULDER_MOUNT_X_M, SHOULDER_MOUNT_Y_M, SHOULDER_MOUNT_Z_M], dtype=float)
    Q0 = qs
    Q1 = qs + qe
    Q2 = qs + qe + qw
    u0 = _unit_x_world_from_yaw_pitch(qy, Q0)
    u1 = _unit_x_world_from_yaw_pitch(qy, Q1)
    u2 = _unit_x_world_from_yaw_pitch(qy, Q2)
    p_elbow = o + UPPER_ARM_LENGTH_M * u0
    p_wrist = p_elbow + FOREARM_LENGTH_M * u1
    p_tip = p_wrist + HAND_LENGTH_M * u2
    return o, p_elbow, p_wrist, p_tip


def release_site_fk_vel_SPEC(q4: np.ndarray, qdot4: np.ndarray) -> np.ndarray:
    """Linear velocity of release site from hinge kinematics (world m/s)."""
    q4 = np.asarray(q4, dtype=float).reshape(4)
    qdot4 = np.asarray(qdot4, dtype=float).reshape(4)
    qy, qs, qe, qw = float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3])
    qdy, qds, qde, qdw = float(qdot4[0]), float(qdot4[1]), float(qdot4[2]), float(qdot4[3])

    Q0 = qs
    Q1 = qs + qe
    Q2 = qs + qe + qw
    Qd0 = qds
    Qd1 = qds + qde
    Qd2 = qds + qde + qdw

    def seg_vel(Q: float, Qd: float) -> np.ndarray:
        return _dunit_x_d_yaw(qy, Q) * qdy + _dunit_x_d_pitch(qy, Q) * Qd

    v = (
        UPPER_ARM_LENGTH_M * seg_vel(Q0, Qd0)
        + FOREARM_LENGTH_M * seg_vel(Q1, Qd1)
        + HAND_LENGTH_M * seg_vel(Q2, Qd2)
    )
    return v


def release_site_position_jacobian_wrt_q_SPEC(q4: np.ndarray) -> np.ndarray:
    """
    ∂ release_site_xyz / ∂ q  (3×4) for analytic linearization (iLQR / Jacobians).

    Columns: [yaw, shoulder, elbow, wrist].
    """
    q4 = np.asarray(q4, dtype=float).reshape(4)
    qy, qs, qe, qw = float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3])
    Q0 = qs
    Q1 = qs + qe
    Q2 = qs + qe + qw

    # d u / d cumulative pitch = dunit_x_d_pitch for all segments sharing that pitch rate
    du_dQ0 = _dunit_x_d_pitch(qy, Q0)
    du_dQ1 = _dunit_x_d_pitch(qy, Q1)
    du_dQ2 = _dunit_x_d_pitch(qy, Q2)

    du_dy0 = _dunit_x_d_yaw(qy, Q0)
    du_dy1 = _dunit_x_d_yaw(qy, Q1)
    du_dy2 = _dunit_x_d_yaw(qy, Q2)

    # ∂p/∂q_yaw
    col_yaw = (
        UPPER_ARM_LENGTH_M * du_dy0
        + FOREARM_LENGTH_M * du_dy1
        + HAND_LENGTH_M * du_dy2
    )
    # ∂p/∂q_shoulder: all three segments
    col_sh = (
        UPPER_ARM_LENGTH_M * du_dQ0
        + FOREARM_LENGTH_M * du_dQ1
        + HAND_LENGTH_M * du_dQ2
    )
    # ∂p/∂q_elbow: last two
    col_el = FOREARM_LENGTH_M * du_dQ1 + HAND_LENGTH_M * du_dQ2
    # ∂p/∂q_wrist: hand only
    col_wr = HAND_LENGTH_M * du_dQ2
    return np.column_stack([col_yaw, col_sh, col_el, col_wr])


def cubic_spline_q_des_SPEC(
    t_s: float, q_start4: np.ndarray, knots12: np.ndarray, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    **PDF A3:** Cubic spline through **3 knot values per joint** over `duration_s`.

    Parameterization: policy vector `knots12` is length 12, ordered as
    [j0_k0..k2, j1_k0..k2, j2_k0..k2, j3_k0..k2] for the four joints.
    """
    q_start4 = np.asarray(q_start4, dtype=float).reshape(4)
    k = np.asarray(knots12, dtype=float).reshape(12)
    T = max(duration_s, 1e-6)
    u = np.clip(t_s / T, 0.0, 1.0)

    q_des = np.zeros(4)
    qd_des = np.zeros(4)
    for j in range(4):
        p0 = q_start4[j]
        p1 = k[j * 3 + 0]
        p2 = k[j * 3 + 1]
        p3 = k[j * 3 + 2]
        q_des[j] = (
            (1 - u) ** 3 * p0
            + 3 * (1 - u) ** 2 * u * p1
            + 3 * (1 - u) * u**2 * p2
            + u**3 * p3
        )
        qd_des[j] = (
            (-3 * (1 - u) ** 2 * p0 + 3 * (1 - u) * (1 - 3 * u) * p1 + 3 * u * (2 - 3 * u) * p2 + 3 * u**2 * p3)
            / T
        )
    return q_des, qd_des


def cubic_spline_qdd_des_SPEC(
    t_s: float, q_start4: np.ndarray, knots12: np.ndarray, duration_s: float
) -> np.ndarray:
    """Desired joint acceleration from the cubic Bezier parameterization used in A3."""
    q_start4 = np.asarray(q_start4, dtype=float).reshape(4)
    k = np.asarray(knots12, dtype=float).reshape(12)
    T = max(duration_s, 1e-6)
    u = np.clip(t_s / T, 0.0, 1.0)

    qdd_des = np.zeros(4)
    for j in range(4):
        p0 = q_start4[j]
        p1 = k[j * 3 + 0]
        p2 = k[j * 3 + 1]
        p3 = k[j * 3 + 2]
        bdd = 6 * (1 - u) * (p2 - 2 * p1 + p0) + 6 * u * (p3 - 2 * p2 + p1)
        qdd_des[j] = bdd / (T**2)
    return qdd_des


def pd_torque_SPEC(
    q: np.ndarray, qdot: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray, kp=None, kd=None
) -> np.ndarray:
    """**PDF A3:** τ = Kp (q* - q) + Kd (q̇* - q̇)."""
    kp = PD_KP_DEFAULT if kp is None else kp
    kd = PD_KD_DEFAULT if kd is None else kd
    return kp * (q_des - q) + kd * (qd_des - qdot)


def feedforward_pd_torque_SPEC(
    q: np.ndarray,
    qdot: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    qdd_des: np.ndarray,
    kp=None,
    kd=None,
    inertia_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Lightweight computed-torque style controller: tau_ff + PD correction."""
    if inertia_diag is None:
        inertia_diag = np.array([1.2, 1.0, 0.8, 0.3], dtype=float)
    M_diag = np.asarray(inertia_diag, dtype=float).reshape(4)
    tau_ff = M_diag * np.asarray(qdd_des, dtype=float).reshape(4)
    return tau_ff + pd_torque_SPEC(q, qdot, q_des, qd_des, kp=kp, kd=kd)


def perturb_torque_SPEC(
    tau: np.ndarray, rng: np.random.Generator, sigma_add=0.5, sigma_mult=0.02
) -> np.ndarray:
    """
    **PDF A3 / Lawrence:** additive Gaussian + multiplicative noise ~ |τ|.
    """
    tau = np.asarray(tau, dtype=float).reshape(4)
    noise = sigma_add * rng.standard_normal(4) + sigma_mult * np.abs(tau) * rng.standard_normal(4)
    return tau + noise


_XML_PATH = mjcf_path_SPEC("arm_4dof_tall.xml")

# Tuned MuJoCo defaults for a forward board-facing throw.
DEFAULT_MUJOCO_THROW_KNOTS_SPEC = np.radians(
    np.array(
        [
            2.0,
            5.0,
            8.0,  # shoulder_yaw
            -12.0,
            -3.0,
            8.0,  # shoulder
            -115.0,
            -35.0,
            -4.0,  # elbow
            0.0,
            20.0,
            30.0,  # wrist
        ],
        dtype=float,
    )
)
DEFAULT_MUJOCO_PD_KP_SPEC = 350.0
DEFAULT_MUJOCO_PD_KD_SPEC = 1.5

DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC = np.radians(
    np.array(
        [
            -6.0,
            -4.0,
            -2.0,  # shoulder_yaw
            -32.0,
            -26.0,
            -7.0,  # shoulder
            -125.0,
            -40.0,
            -4.0,  # elbow
            -10.0,
            1.0,
            19.0,  # wrist
        ],
        dtype=float,
    )
)
DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC = 349.0
DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC = 1.89
DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC = 0.10


def simulate_throw_mujoco_SPEC(
    knots12: np.ndarray,
    q_start4: np.ndarray | None = None,
    xml_path: "str | Path | None" = None,
    duration_s: float | None = None,
    release_time_s: float | None = None,
    rng: "np.random.Generator | None" = None,
    torque_noise: bool = False,
    torque_noise_sigma_add: float = 0.5,
    torque_noise_sigma_mult: float = 0.02,
    kp: float | None = None,
    kd: float | None = None,
    use_feedforward: bool = False,
    inertia_ff_diag: np.ndarray | None = None,
    enforce_joint_limits: bool = True,
) -> dict:
    """
    Full MuJoCo **4-DOF** arm simulation using `motion_mujoco_mjcf_4dof_arm_*.xml`.
    """
    import mujoco  # optional: only imported here to keep mujoco non-required

    xml_path = Path(xml_path) if xml_path else _XML_PATH
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    rng = rng or np.random.default_rng()
    duration_s = THROW_DURATION_S if duration_s is None else duration_s
    if release_time_s is None:
        t_nom = 0.85 * duration_s
        t_r = max(m.opt.timestep, t_nom + RELEASE_TIME_SIGMA_S * rng.standard_normal())
    else:
        t_r = max(m.opt.timestep, float(release_time_s))

    if q_start4 is None:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qvel[:] = 0.0
        q_start4 = d.qpos[:4].copy()
        mujoco.mj_forward(m, d)
    else:
        q_start4 = np.asarray(q_start4, dtype=float).reshape(4)
        d.qpos[:4] = q_start4
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    release_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "release_site")
    jacp = np.zeros((3, m.nv))

    ts: list[float] = []
    qs: list[np.ndarray] = []
    taus: list[np.ndarray] = []
    release_state6: np.ndarray | None = None
    t = 0.0

    while t <= duration_s + 3 * m.opt.timestep:
        q = d.qpos[:4].copy()
        qd = d.qvel[:4].copy()

        q_des, qd_des = cubic_spline_q_des_SPEC(t, q_start4, knots12, duration_s)
        if use_feedforward:
            qdd_des = cubic_spline_qdd_des_SPEC(t, q_start4, knots12, duration_s)
            tau = feedforward_pd_torque_SPEC(
                q,
                qd,
                q_des,
                qd_des,
                qdd_des,
                kp=DEFAULT_MUJOCO_PD_KP_SPEC if kp is None else kp,
                kd=DEFAULT_MUJOCO_PD_KD_SPEC if kd is None else kd,
                inertia_diag=inertia_ff_diag,
            )
        else:
            tau = pd_torque_SPEC(
                q,
                qd,
                q_des,
                qd_des,
                kp=DEFAULT_MUJOCO_PD_KP_SPEC if kp is None else kp,
                kd=DEFAULT_MUJOCO_PD_KD_SPEC if kd is None else kd,
            )
        if torque_noise:
            tau = perturb_torque_SPEC(
                tau,
                rng,
                sigma_add=torque_noise_sigma_add,
                sigma_mult=torque_noise_sigma_mult,
            )

        ts.append(t)
        qs.append(q.copy())
        taus.append(tau.copy())

        if release_state6 is None and t >= t_r:
            mujoco.mj_jacSite(m, d, jacp, None, release_id)
            pos = d.site_xpos[release_id].copy()
            vel = jacp @ d.qvel
            release_state6 = np.concatenate([pos, vel])

        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if enforce_joint_limits:
            q = d.qpos[:4]
            qd = d.qvel[:4]
            for j, (lo, hi) in enumerate(m.jnt_range[:4]):
                if not m.jnt_limited[j]:
                    continue
                if q[j] < lo:
                    q[j] = lo
                    qd[j] = 0.0
                elif q[j] > hi:
                    q[j] = hi
                    qd[j] = 0.0
            d.qpos[:4] = q
            d.qvel[:4] = qd
            mujoco.mj_forward(m, d)

        t += m.opt.timestep

        if t > duration_s + 0.05 and release_state6 is not None:
            break

    return {
        "times": np.array(ts),
        "q": np.array(qs),
        "tau": np.array(taus),
        "release_time_s": t_r,
        "release_state6": release_state6,
    }


def simulate_throw_pd_SPEC(
    knots12: np.ndarray,
    q_start4: np.ndarray,
    dt: float = 0.001,
    duration_s: float = None,
    release_time_s: float = None,
    rng: np.random.Generator | None = None,
    torque_noise: bool = False,
):
    """
    Integrate **joint dynamics** with **velocity-level** PD (simple diagonal inertia ≈1 baseline).

    Intended signal flow for RL (12 spline parameters → torques → release site state).
    """
    rng = rng or np.random.default_rng()
    duration_s = duration_s if duration_s is not None else THROW_DURATION_S
    t_nom = release_time_s if release_time_s is not None else 0.85 * duration_s
    t_r = max(dt, t_nom + RELEASE_TIME_SIGMA_S * rng.standard_normal())

    q = np.asarray(q_start4, dtype=float).reshape(4).copy()
    qd = np.zeros(4)
    invM = np.array([1.0 / 1.2, 1.0 / 1.0, 1.0 / 0.8, 1.0 / 0.3], dtype=float)

    ts, qs, taus = [], [], []
    release_state6 = None
    t = 0.0
    while t <= duration_s + 3 * dt:
        q_des, qd_des = cubic_spline_q_des_SPEC(t, q_start4, knots12, duration_s)
        tau = pd_torque_SPEC(q, qd, q_des, qd_des)
        if torque_noise:
            tau = perturb_torque_SPEC(tau, rng)
        tau[0] = np.clip(tau[0], -TORQUE_LIMIT_SHOULDER_YAW_NM, TORQUE_LIMIT_SHOULDER_YAW_NM)
        tau[1] = np.clip(tau[1], -TORQUE_LIMIT_SHOULDER_NM, TORQUE_LIMIT_SHOULDER_NM)
        tau[2] = np.clip(tau[2], -TORQUE_LIMIT_ELBOW_NM, TORQUE_LIMIT_ELBOW_NM)
        tau[3] = np.clip(tau[3], -TORQUE_LIMIT_WRIST_NM, TORQUE_LIMIT_WRIST_NM)
        qdd = invM * tau
        qd = qd + qdd * dt
        q = clamp_joint_vector_SPEC(q + qd * dt)
        ts.append(t)
        qs.append(q.copy())
        taus.append(tau.copy())
        if release_state6 is None and t >= t_r:
            pos, _ = release_site_fk_SPEC(q)
            vel = release_site_fk_vel_SPEC(q, qd)
            release_state6 = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]], dtype=float)
        t += dt
        if t > duration_s + 0.05 and release_state6 is not None:
            break
    return {
        "times": np.array(ts),
        "q": np.array(qs),
        "tau": np.array(taus),
        "release_time_s": t_r,
        "release_state6": release_state6,
    }
