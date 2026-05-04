"""
A3_cubic_spline_and_pd_controller_SPEC.py
=========================================
**PDF section:** Track A — A3 (Trajectory and controller)

**Components (PDF):**
1. **Cubic spline** per joint: **3 knots per joint** → **9 policy parameters** total.
   Starting posture is **fixed** (cocked keyframe); knots define the forward throw over ~**0.2 s**.
2. **PD control:** τ = Kp (q* − q) + Kd (q̇* − q̇) with default **Kp=100**, **Kd=10** per joint.
3. **Motor noise (PDF / Lawrence):** perturb τ with **additive Gaussian** + **multiplicative**
   noise proportional to |τ|.
4. **Release:** at time **t_r** (Gaussian jitter σ=0.01 s per PDF), read **release_site**
   position and velocity → **6D initial condition** for projectile (**Track B / C1**).

**Coordinate convention:** Matches **B1/B2** world frame (x toward board, y left, z up).

**FK convention:** Three hinges about **world y** (parallel), link lengths along **local x** after
each cumulative rotation — consistent with `A2_mujoco_mjcf_3link_arm_SPEC.xml` layout
(geom along +x in each body frame at zero joint angle).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    FOREARM_LENGTH_M,
    HAND_LENGTH_M,
    PD_KP_DEFAULT,
    PD_KD_DEFAULT,
    RELEASE_TIME_SIGMA_S,
    SHOULDER_MOUNT_X_M,
    SHOULDER_MOUNT_Y_M,
    SHOULDER_MOUNT_Z_M,
    THROW_DURATION_S,
    UPPER_ARM_LENGTH_M,
)
from track_A_arm_SPEC.A1_link_geometry_and_inertia_SPEC import clamp_joint_vector_SPEC


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
    q_start3: np.ndarray,
    q_goal3: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimum-jerk reference (q, qdot, qddot) between two 3D joint postures.
    """
    T = max(float(duration_s), 1e-6)
    u = np.clip(float(t_s) / T, 0.0, 1.0)
    q_start3 = np.asarray(q_start3, dtype=float).reshape(3)
    q_goal3 = np.asarray(q_goal3, dtype=float).reshape(3)
    dq = q_goal3 - q_start3

    s = minimum_jerk_progress_SPEC(u)
    s_u = minimum_jerk_progress_dot_SPEC(u)
    s_uu = minimum_jerk_progress_ddot_SPEC(u)

    q = q_start3 + s * dq
    qd = (s_u / T) * dq
    qdd = (s_uu / (T**2)) * dq
    return q, qd, qdd


def nominal_minimum_jerk_knots_SPEC(
    q_start3: np.ndarray,
    q_goal3: np.ndarray,
) -> np.ndarray:
    """
    Build the 9D knot vector from a minimum-jerk profile sampled at 1/3, 2/3, and 1.
    """
    q_start3 = np.asarray(q_start3, dtype=float).reshape(3)
    q_goal3 = np.asarray(q_goal3, dtype=float).reshape(3)
    knots = []
    for joint_idx in range(3):
        q0 = q_start3[joint_idx]
        q1 = q_goal3[joint_idx]
        dq = q1 - q0
        for u in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            knots.append(q0 + minimum_jerk_progress_SPEC(u) * dq)
    return np.asarray(knots, dtype=float)


def plan_nominal_throw_knots_min_jerk_SPEC(
    q_start3: np.ndarray,
    q_goal3: np.ndarray | None = None,
    alpha_goal: float = 0.25,
) -> dict:
    """
    Create a minimum-jerk nominal throw in knot space for warm starts / OC scaffolding.

    The default goal posture is a blended extension toward the current overhand preset
    so this planner stays consistent with the tuned working motion in this repository.
    """
    q_start3 = np.asarray(q_start3, dtype=float).reshape(3)
    if q_goal3 is None:
        preset_goal = DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC.reshape(3, 3)[:, -1]
        q_goal3 = (1.0 - alpha_goal) * q_start3 + alpha_goal * preset_goal
    else:
        q_goal3 = np.asarray(q_goal3, dtype=float).reshape(3)
    knots9 = nominal_minimum_jerk_knots_SPEC(q_start3, q_goal3)
    return {
        "knots9": knots9,
        "q_start3": q_start3.copy(),
        "q_goal3": q_goal3.copy(),
    }


def rot_y_unit_x_SPEC(cumulative_angle_rad: float, length_m: float) -> np.ndarray:
    """Vector from rotating (length,0,0) about +y by cumulative_angle_rad."""
    c = math.cos(cumulative_angle_rad)
    s = math.sin(cumulative_angle_rad)
    return np.array([length_m * c, 0.0, -length_m * s], dtype=float)


def release_site_fk_SPEC(q012: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics for fingertip **release_site** (world m, m/s).

    q012 = [q0,q1,q2] shoulder, elbow, wrist joint angles (rad), **absolute** rotations
    about **y** for each link segment in the sense: segment i direction angle = q0+...+qi.

    Returns
    -------
    pos : (3,) world position of release site
    vel : (3,) world velocity **assuming q̇ passed separately** — this function returns
          only geometric Jacobian * qdot in `release_site_fk_vel_SPEC`.
    """
    q012 = np.asarray(q012, dtype=float).reshape(3)
    q0, q1, q2 = q012
    o = np.array([SHOULDER_MOUNT_X_M, SHOULDER_MOUNT_Y_M, SHOULDER_MOUNT_Z_M], dtype=float)
    p = (
        o
        + rot_y_unit_x_SPEC(q0, UPPER_ARM_LENGTH_M)
        + rot_y_unit_x_SPEC(q0 + q1, FOREARM_LENGTH_M)
        + rot_y_unit_x_SPEC(q0 + q1 + q2, HAND_LENGTH_M)
    )
    return p, np.zeros(3)


def release_site_fk_vel_SPEC(q012: np.ndarray, qdot012: np.ndarray) -> np.ndarray:
    """Linear velocity of release site from hinge kinematics (world m/s)."""
    q012 = np.asarray(q012, dtype=float).reshape(3)
    qdot012 = np.asarray(qdot012, dtype=float).reshape(3)
    q0, q1, q2 = q012
    qd0, qd1, qd2 = qdot012
    # d/dt R_y(Q)*(L,0,0) = Qdot * (-L sin Q, 0, -L cos Q) for Q = q0 etc.
    def d_segment(Q, Qdot, L):
        return np.array([-L * math.sin(Q) * Qdot, 0.0, -L * math.cos(Q) * Qdot], dtype=float)

    v = (
        d_segment(q0, qd0, UPPER_ARM_LENGTH_M)
        + d_segment(q0 + q1, qd0 + qd1, FOREARM_LENGTH_M)
        + d_segment(q0 + q1 + q2, qd0 + qd1 + qd2, HAND_LENGTH_M)
    )
    return v


def cubic_spline_q_des_SPEC(t_s: float, q_start3: np.ndarray, knots9: np.ndarray, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    **PDF A3:** Cubic spline through **3 knot values per joint** over `duration_s`.

    Parameterization: policy vector `knots9` is length 9, ordered as
    [j0_k0, j0_k1, j0_k2, j1_k0, j1_k1, j1_k2, j2_k0, j2_k1, j2_k2].

    Boundary: q(0)=q_start, q(duration)=last knot per joint (knot2), interior knots at 1/3 and 2/3 time.

    This is a **minimal** piecewise cubic: we map uniform times 0, T/3, 2T/3, T to
    q_start, knot0, knot1, knot2 for each joint independently.
    """
    q_start3 = np.asarray(q_start3, dtype=float).reshape(3)
    k = np.asarray(knots9, dtype=float).reshape(9)
    T = max(duration_s, 1e-6)
    u = np.clip(t_s / T, 0.0, 1.0)

    q_des = np.zeros(3)
    qd_des = np.zeros(3)
    for j in range(3):
        p0 = q_start3[j]
        p1 = k[j * 3 + 0]
        p2 = k[j * 3 + 1]
        p3 = k[j * 3 + 2]
        # Cubic Bezier: B(u)=(1-u)^3 p0 + 3(1-u)^2 u p1 + 3(1-u)u^2 p2 + u^3 p3
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


def cubic_spline_qdd_des_SPEC(t_s: float, q_start3: np.ndarray, knots9: np.ndarray, duration_s: float) -> np.ndarray:
    """
    Desired joint acceleration from the cubic Bezier parameterization used in A3.
    """
    q_start3 = np.asarray(q_start3, dtype=float).reshape(3)
    k = np.asarray(knots9, dtype=float).reshape(9)
    T = max(duration_s, 1e-6)
    u = np.clip(t_s / T, 0.0, 1.0)

    qdd_des = np.zeros(3)
    for j in range(3):
        p0 = q_start3[j]
        p1 = k[j * 3 + 0]
        p2 = k[j * 3 + 1]
        p3 = k[j * 3 + 2]
        bdd = 6 * (1 - u) * (p2 - 2 * p1 + p0) + 6 * u * (p3 - 2 * p2 + p1)
        qdd_des[j] = bdd / (T**2)
    return qdd_des


def pd_torque_SPEC(q: np.ndarray, qdot: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray, kp=None, kd=None) -> np.ndarray:
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
    """
    Lightweight computed-torque style controller: tau_ff + PD correction.
    """
    if inertia_diag is None:
        # Matches the simple diagonal inertia magnitudes used in simulate_throw_pd_SPEC.
        inertia_diag = np.array([1.0, 0.8, 0.3], dtype=float)
    M_diag = np.asarray(inertia_diag, dtype=float).reshape(3)
    tau_ff = M_diag * np.asarray(qdd_des, dtype=float).reshape(3)
    return tau_ff + pd_torque_SPEC(q, qdot, q_des, qd_des, kp=kp, kd=kd)


def perturb_torque_SPEC(tau: np.ndarray, rng: np.random.Generator, sigma_add=0.5, sigma_mult=0.02) -> np.ndarray:
    """
    **PDF A3 / Lawrence:** additive Gaussian + multiplicative noise ~ |τ|.
    Magnitudes are placeholders — tune to match Lawrence et al. noise levels.
    """
    tau = np.asarray(tau, dtype=float).reshape(3)
    noise = sigma_add * rng.standard_normal(3) + sigma_mult * np.abs(tau) * rng.standard_normal(3)
    return tau + noise


_XML_PATH = Path(__file__).parent / "A2_mujoco_mjcf_3link_arm_SPEC.xml"

# Tuned MuJoCo defaults for a forward board-facing throw. The PDF's baseline PD values
# remain documented above; these are the current working rollout defaults for the full MJCF sim.
DEFAULT_MUJOCO_THROW_KNOTS_SPEC = np.radians(np.array([
    -12.0,  -3.0,  8.0,  # shoulder: 43° initial error from -55° keyframe drives overhand swing
   -115.0, -35.0, -4.0,  # elbow: aggressive extension toward parallel (0°) without going downward
      0.0,  20.0, 30.0,  # wrist: late forward snap
], dtype=float))
DEFAULT_MUJOCO_PD_KP_SPEC = 350.0
DEFAULT_MUJOCO_PD_KD_SPEC = 1.5

# A4 visual preset: releases higher and keeps the shoulder negative longer so the
# stick figure reads more like an overhand dart throw than the score-first default.
DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC = np.radians(np.array([
    -32.0, -26.0, -7.0,
   -125.0, -40.0, -4.0,
    -10.0,   1.0, 19.0,
], dtype=float))
DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC = 349.0
DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC = 1.89
DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC = 0.10


def simulate_throw_mujoco_SPEC(
    knots9: np.ndarray,
    q_start3: np.ndarray | None = None,
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
    Full MuJoCo arm simulation using A2_mujoco_mjcf_3link_arm_SPEC.xml.

    Replaces the crude diagonal inertia in simulate_throw_pd_SPEC with real capsule
    inertias (inertiafromgeom=true) and joint-limit constraints from the MJCF.
    Interface is identical so callers can swap either function.

    Key steps per timestep:
      1. Read current qpos/qvel from MuJoCo data.
      2. Evaluate cubic-spline desired trajectory at current t.
      3. Compute PD torque; optionally add noise via perturb_torque_SPEC.
      4. Write to d.ctrl (MuJoCo clips to ctrlrange from the XML automatically).
      5. Call mj_step — advances the full rigid-body dynamics.
      6. At t >= t_r: read release_site world position from d.site_xpos and
         translational velocity via mj_jacSite * qvel.

    The default gains used here are intentionally lower-damped than the PDF starter
    values so the elbow can reach a forward release velocity.
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

    if q_start3 is None:
        mujoco.mj_resetDataKeyframe(m, d, 0)  # "cocked" keyframe
        d.qvel[:] = 0.0
        q_start3 = d.qpos[:3].copy()
        mujoco.mj_forward(m, d)
    else:
        q_start3 = np.asarray(q_start3, dtype=float).reshape(3)
        d.qpos[:3] = q_start3
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
        q = d.qpos[:3].copy()
        qd = d.qvel[:3].copy()

        q_des, qd_des = cubic_spline_q_des_SPEC(t, q_start3, knots9, duration_s)
        if use_feedforward:
            qdd_des = cubic_spline_qdd_des_SPEC(t, q_start3, knots9, duration_s)
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
            # Capture release: site position from MuJoCo FK, velocity via geometric Jacobian
            mujoco.mj_jacSite(m, d, jacp, None, release_id)
            pos = d.site_xpos[release_id].copy()
            vel = jacp @ d.qvel
            release_state6 = np.concatenate([pos, vel])

        d.ctrl[:] = tau  # MuJoCo clips to ctrlrange in the XML
        mujoco.mj_step(m, d)

        if enforce_joint_limits:
            # MuJoCo's joint limits are soft constraints; under aggressive control they can
            # be violated slightly. For our dart-throw kinematics we want a hard floor on:
            # - elbow extension (q_elbow >= 0): forearm never hyper-extends past parallel
            # - wrist range: keep within modeled anatomical range
            q = d.qpos[:3]
            qd = d.qvel[:3]
            for j, (lo, hi) in enumerate(m.jnt_range[:3]):
                if not m.jnt_limited[j]:
                    continue
                if q[j] < lo:
                    q[j] = lo
                    qd[j] = 0.0
                elif q[j] > hi:
                    q[j] = hi
                    qd[j] = 0.0
            d.qpos[:3] = q
            d.qvel[:3] = qd
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
    knots9: np.ndarray,
    q_start3: np.ndarray,
    dt: float = 0.001,
    duration_s: float = None,
    release_time_s: float = None,
    rng: np.random.Generator | None = None,
    torque_noise: bool = False,
):
    """
    Integrate **joint dynamics** with **velocity-level** PD (simple diagonal inertia ≈1 for baseline).

    This is a **lightweight stand-in** until full MuJoCo arm dynamics are wired (**C1**).
    Returns time series and **release_state6** if release occurs.

    **PDF note:** Full sim should use MuJoCo `mj_step` + actuators; this function documents the
    intended signal flow for RL (9 spline parameters → torques → release site state).
    """
    rng = rng or np.random.default_rng()
    duration_s = duration_s if duration_s is not None else THROW_DURATION_S
    # Release time with optional jitter (PDF σ=0.01 s)
    t_nom = release_time_s if release_time_s is not None else 0.85 * duration_s
    t_r = max(dt, t_nom + RELEASE_TIME_SIGMA_S * rng.standard_normal())

    q = np.asarray(q_start3, dtype=float).reshape(3).copy()
    qd = np.zeros(3)
    invM = np.array([1.0 / 1.0, 1.0 / 0.8, 1.0 / 0.3])  # crude diagonal inertia (not from PDF)

    ts, qs, taus = [], [], []
    release_state6 = None
    t = 0.0
    while t <= duration_s + 3 * dt:
        q_des, qd_des = cubic_spline_q_des_SPEC(t, q_start3, knots9, duration_s)
        tau = pd_torque_SPEC(q, qd, q_des, qd_des)
        if torque_noise:
            tau = perturb_torque_SPEC(tau, rng)
        tau[0] = np.clip(tau[0], -100.0, 100.0)
        tau[1] = np.clip(tau[1], -70.0, 70.0)
        tau[2] = np.clip(tau[2], -20.0, 20.0)  # PDF A2 torque limits
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
