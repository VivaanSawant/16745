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


def pd_torque_SPEC(q: np.ndarray, qdot: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray, kp=None, kd=None) -> np.ndarray:
    """**PDF A3:** τ = Kp (q* - q) + Kd (q̇* - q̇)."""
    kp = PD_KP_DEFAULT if kp is None else kp
    kd = PD_KD_DEFAULT if kd is None else kd
    return kp * (q_des - q) + kd * (qd_des - qdot)


def perturb_torque_SPEC(tau: np.ndarray, rng: np.random.Generator, sigma_add=0.5, sigma_mult=0.02) -> np.ndarray:
    """
    **PDF A3 / Lawrence:** additive Gaussian + multiplicative noise ~ |τ|.
    Magnitudes are placeholders — tune to match Lawrence et al. noise levels.
    """
    tau = np.asarray(tau, dtype=float).reshape(3)
    noise = sigma_add * rng.standard_normal(3) + sigma_mult * np.abs(tau) * rng.standard_normal(3)
    return tau + noise


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
