"""
Physics simulator demo: 3D throw, dartboard scoring, hit any target, variance/Monte Carlo.
"""
import numpy as np
import matplotlib.pyplot as plt

from physics.dartboard import score_at, get_target_center, BOARD_RADIUS
from physics.projectile import throw_3d, landing_position
from physics.target_solver import aim_point, find_velocity_to_hit, parse_target
from physics.variance import monte_carlo_landings, landing_stats
from physics.state_estimator import estimate_release_from_arm


def demo_3d_throw():
    """Single 3D throw with drag, no wind."""
    pos0 = [0.0, 0.0, 1.7]   # release ~1.7 m height, 2.37 m to board
    vel0 = [4.5, 0.0, 2.0]   # toward +x, slightly up
    t, x, y, z, vx, vy, vz = throw_3d(pos0, vel0, wind=(0, 0, 0))
    land_x, land_y = x[-1], y[-1]
    score = score_at(land_x, land_y)
    print("--- 3D throw (no wind) ---")
    print(f"Landing: ({land_x:.4f}, {land_y:.4f}) m  Score: {score}")
    return t, x, y, z


def demo_hit_any_number():
    """Aim at T20, S19, D5, BULL and compute release velocity + landing."""
    pos0 = [0.0, 0.0, 1.7]
    for target in ["T20", "S19", "D5", "BULL"]:
        aim = aim_point(target)
        v, land, score = find_velocity_to_hit(pos0, target, v_mag=5.5)
        print(f"Target {target}: aim=({aim[0]:.4f}, {aim[1]:.4f}) -> land=({land[0]:.4f}, {land[1]:.4f}) score={score}")


def demo_variance():
    """Monte Carlo: same nominal throw, add release noise -> landing distribution & expected score."""
    pos0 = [0.0, 0.0, 1.7]
    vel0 = [5.0, 0.0, 2.5]
    landings, scores = monte_carlo_landings(pos0, vel0, n_samples=80, vel_std=0.08, wind_std=0.0, seed=42)
    mean_xy, std_xy, expected_score = landing_stats(landings, scores)
    print("--- Variance / Monte Carlo ---")
    print(f"Mean landing: ({mean_xy[0]:.4f}, {mean_xy[1]:.4f}) m")
    print(f"Std landing:  ({std_xy[0]:.4f}, {std_xy[1]:.4f}) m")
    print(f"Expected score: {expected_score:.2f}")
    return landings, scores


def demo_state_estimator():
    """Estimate release state from arm joint angles/velocities."""
    # 2-link arm: shoulder, elbow (planar)
    joint_angles = [0.3, -0.8]
    joint_velocities = [1.2, 2.0]
    link_lengths = [0.35, 0.35]
    pos0, vel0 = estimate_release_from_arm(joint_angles, joint_velocities, link_lengths, noise_std=0.0)
    print("--- State estimator (arm -> release) ---")
    print(f"Release pos: {pos0}, vel: {vel0}")
    return pos0, vel0


def plot_trajectory_and_board(t, x, y, z, landings_xy=None):
    """Plot trajectory in x-z and board with landing(s)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # x-z side view
    axes[0].plot(x, z, "b-")
    axes[0].axhline(0, color="k", ls="--")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("z (m)")
    axes[0].set_title("Trajectory (side)")
    axes[0].set_aspect("equal"); axes[0].grid(True)
    # board top view
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(BOARD_RADIUS*np.cos(theta), BOARD_RADIUS*np.sin(theta), "k-")
    axes[1].plot(0, 0, "ko", markersize=4)
    axes[1].scatter(x[-1], y[-1], c="blue", s=30, label="landing")
    if landings_xy is not None:
        axes[1].scatter(landings_xy[:, 0], landings_xy[:, 1], c="red", s=5, alpha=0.5, label="MC landings")
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
    axes[1].set_title("Board (top)"); axes[1].set_aspect("equal"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("physics_demo.png", dpi=120)
    print("Saved physics_demo.png")
    # plt.show()  # uncomment for interactive


if __name__ == "__main__":
    t, x, y, z = demo_3d_throw()
    demo_hit_any_number()
    landings, scores = demo_variance()
    demo_state_estimator()
    plot_trajectory_and_board(t, x, y, z, landings_xy=landings)
