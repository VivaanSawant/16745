"""
Simple state estimator for the dart-arm system at release.
Given noisy observations (e.g. joint angles, velocities, last known dart pose),
estimates release state: position and velocity of dart at release.
No full EKF; kept minimal for baseline.
"""
import numpy as np


def estimate_release_from_arm(
    joint_angles,
    joint_velocities,
    link_lengths,
    release_offset=(0.0, 0.0, 0.0),
    noise_std=0.0,
    rng=None,
):
    """
    Simple 2D or 3D arm: joint_angles and joint_velocities are (n,) for n joints.
    link_lengths (n,) gives length of each link. End-effector position and velocity
    in world frame (x forward, y left, z up) with simple forward kinematics.
    Assumes planar arm in x-z for first two joints; third can be y rotation for "wrist".
    release_offset: (dx, dy, dz) from end-effector to dart release point.
    Returns (pos0, vel0) with optional Gaussian noise (noise_std on position and velocity).
    """
    rng = rng or np.random.default_rng()
    angles = np.asarray(joint_angles).ravel()
    vel = np.asarray(joint_velocities).ravel()
    lengths = np.asarray(link_lengths).ravel()
    n = min(len(angles), len(vel), len(lengths))
    angles, vel, lengths = angles[:n], vel[:n], lengths[:n]

    # Simple planar chain: each link i has angle theta_i from previous link
    x, z = 0.0, 0.0
    vx, vz = 0.0, 0.0
    th_sum = 0.0
    for i in range(n):
        th_sum += angles[i]
        L = lengths[i]
        x += L * np.sin(th_sum)
        z += L * np.cos(th_sum)
        vx += L * vel[i] * np.cos(th_sum)
        vz += L * vel[i] * (-np.sin(th_sum))
    y = 0.0
    vy = 0.0
    if n >= 3:
        y = release_offset[1]
        vy = vel[2] * 0.1 if len(vel) > 2 else 0.0

    pos0 = np.array([x, y, z]) + np.asarray(release_offset)
    vel0 = np.array([vx, vy, vz])

    if noise_std > 0:
        pos0 = pos0 + noise_std * rng.standard_normal(3)
        vel0 = vel0 + noise_std * 2.0 * rng.standard_normal(3)
    return pos0, vel0


def estimate_release_direct(measured_pos, measured_vel, noise_std=0.0, rng=None):
    """
    If we have a direct sensor of release position/velocity, add optional noise.
    """
    rng = rng or np.random.default_rng()
    pos0 = np.asarray(measured_pos, dtype=float).ravel()[:3]
    vel0 = np.asarray(measured_vel, dtype=float).ravel()[:3]
    if noise_std > 0:
        pos0 = pos0 + noise_std * rng.standard_normal(3)
        vel0 = vel0 + noise_std * rng.standard_normal(3)
    return pos0, vel0
