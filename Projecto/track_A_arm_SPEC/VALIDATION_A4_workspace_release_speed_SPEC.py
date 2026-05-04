"""
VALIDATION_A4_workspace_release_speed_SPEC.py
==============================================
**PDF section:** Track A — A4 (Validation)

**Checklist:**
- Sweep joint angles within limits; record fingertip positions from **A3 FK**.
- Confirm release site can reach **forward** of shoulder with **upward** velocity component
  for some throws (qualitative workspace check).
- **Release speed 5-7 m/s:** for a sample MuJoCo throw from **A3**, print |v| at release.
- **Stick-figure render:** arm posture at 6 evenly-spaced frames during the throw (x-z plane).

Run: `python track_A_arm_SPEC/VALIDATION_A4_workspace_release_speed_SPEC.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_WRIST_DEG,
    SHOULDER_MOUNT_X_M,
    UPPER_ARM_LENGTH_M,
    FOREARM_LENGTH_M,
    HAND_LENGTH_M,
    OCHE_TO_BOARD_X_M,
    BULLSEYE_CENTER_Z_M,
    R_BOARD_MISS_MM,
)
from track_A_arm_SPEC.A1_link_geometry_and_inertia_SPEC import joint_limits_rad_SPEC
import mujoco
import imageio

from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
    DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
    release_site_fk_SPEC,
    rot_y_unit_x_SPEC,
    simulate_throw_mujoco_SPEC,
)

# A4 visualization uses the tall-player model so the elbow can stay above the pivot
# while still producing a dart-like release height.
_XML_PATH = _ROOT / "track_A_arm_SPEC" / "A2_mujoco_mjcf_3link_arm_TALL_SPEC.xml"

_ARTIFACTS = _ROOT / "artifacts_SPEC"
_ARTIFACTS.mkdir(exist_ok=True)

def _shoulder_mount_z_from_model(xml_path: Path) -> float:
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    # The mount body is the first child body in worldbody; use name lookup for safety.
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shoulder_mount")
    return float(m.body_pos[bid][2])


def _arm_segment_xz(q: np.ndarray, shoulder_mount_z_m: float) -> np.ndarray:
    """Return (4, 2) array of [shoulder, elbow, wrist_joint, tip] in the x-z plane."""
    o = np.array([SHOULDER_MOUNT_X_M, shoulder_mount_z_m])
    seg1 = rot_y_unit_x_SPEC(q[0],           UPPER_ARM_LENGTH_M)[[0, 2]]
    seg2 = rot_y_unit_x_SPEC(q[0] + q[1],    FOREARM_LENGTH_M)[[0, 2]]
    seg3 = rot_y_unit_x_SPEC(q[0]+q[1]+q[2], HAND_LENGTH_M)[[0, 2]]
    elbow = o + seg1
    wrist = elbow + seg2
    tip   = wrist + seg3
    return np.array([o, elbow, wrist, tip])


def workspace_grid_SPEC(n_each=5):
    """Coarse grid in joint space; returns max forward x and max z reached."""
    lim = joint_limits_rad_SPEC()
    qs = [np.linspace(lo, hi, n_each) for lo, hi in lim]
    max_x, max_z = -1e9, -1e9
    for q0 in qs[0]:
        for q1 in qs[1]:
            for q2 in qs[2]:
                p, _ = release_site_fk_SPEC(np.array([q0, q1, q2]))
                max_x = max(max_x, p[0])
                max_z = max(max_z, p[2])
    print("--- A4 workspace (coarse grid FK) ---")
    print(f"  Max fingertip x (toward board): {max_x:.3f} m")
    print(f"  Max fingertip z: {max_z:.3f} m")
    print("  PDF expects board-facing positions ~1.5-2.0 m height (z); interpret vs shoulder pivot height")


def sample_throw_speed_SPEC() -> dict:
    """MuJoCo throw with the A4 overhand preset; prints release speed and returns sim result."""
    q0 = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
    out = simulate_throw_mujoco_SPEC(
        DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
        q_start3=q0,
        torque_noise=False,
        kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        rng=np.random.default_rng(0),
        release_time_s=DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        xml_path=_XML_PATH,
        enforce_joint_limits=True,
    )
    s6 = out["release_state6"]
    if s6 is None:
        print("A4: no release captured")
        return out
    spd = float(np.linalg.norm(s6[3:6]))
    assert s6[3] > 0.0, f"A4 regression: expected forward release vx>0, got vx={s6[3]:.3f} m/s"
    print("--- A4 release speed check (MuJoCo dynamics) ---")
    print("  Preset: overhand A4 visualization (higher release, longer negative-shoulder phase)")
    print(f"  |v_release| = {spd:.3f} m/s  (PDF target band 5 to 7 m/s)")
    print(f"  v_release (vx,vy,vz) = {s6[3]:.3f}, {s6[4]:.3f}, {s6[5]:.3f} m/s")
    print(f"  Release pos (x,y,z) = {s6[0]:.3f}, {s6[1]:.3f}, {s6[2]:.3f}")
    return out


def plot_arm_throw_SPEC(sim_result: dict, n_frames: int = 6, save: bool = True):
    """
    Stick-figure plot of the throw in the x-z (side-view) plane.

    Shows n_frames arm postures evenly spaced over the trajectory, coloured light-to-dark.
    The release instant is marked with a red star.
    """
    times = sim_result["times"]
    qs    = sim_result["q"]
    t_r   = sim_result["release_time_s"]
    s6    = sim_result["release_state6"]
    shoulder_mount_z_m = _shoulder_mount_z_from_model(_XML_PATH)

    fig, ax = plt.subplots(figsize=(7, 5))

    frame_idx = np.linspace(0, len(times) - 1, n_frames, dtype=int)
    colors = plt.cm.Blues(np.linspace(0.25, 0.95, n_frames))

    for ci, fi in enumerate(frame_idx):
        pts = _arm_segment_xz(qs[fi], shoulder_mount_z_m)          # (4, 2): shoulder→elbow→wrist→tip
        ax.plot(pts[:, 0], pts[:, 1],
                color=colors[ci], lw=2,
                label=f"t={times[fi]:.3f} s" if ci in (0, n_frames - 1) else None)
        ax.plot(pts[0, 0], pts[0, 1], "o", color=colors[ci], ms=4)

    # Find the frame closest to release time
    r_idx = int(np.argmin(np.abs(times - t_r)))
    r_pts = _arm_segment_xz(qs[r_idx], shoulder_mount_z_m)
    ax.plot(r_pts[:, 0], r_pts[:, 1], color="red", lw=2.5, zorder=5)
    if s6 is not None:
        ax.plot(s6[0], s6[2], "r*", ms=14, zorder=6, label=f"release  |v|={np.linalg.norm(s6[3:6]):.2f} m/s")

    ax.axhline(shoulder_mount_z_m, color="gray", lw=0.7, ls="--", label="shoulder pivot height")

    # --- Board at x = 2.37 m ---
    board_r = R_BOARD_MISS_MM * 1e-3
    ax.axvline(OCHE_TO_BOARD_X_M, color="saddlebrown", lw=1.0, ls=":")
    ax.plot([OCHE_TO_BOARD_X_M, OCHE_TO_BOARD_X_M],
            [BULLSEYE_CENTER_Z_M - board_r, BULLSEYE_CENTER_Z_M + board_r],
            color="saddlebrown", lw=3, solid_capstyle="round", label="dartboard (scoring)")
    ax.plot(OCHE_TO_BOARD_X_M, BULLSEYE_CENTER_Z_M, "o",
            color="saddlebrown", ms=5, label=f"bullseye  z={BULLSEYE_CENTER_Z_M:.2f} m")

    # --- Projectile arc from release to board ---
    if s6 is not None:
        rx, rz = s6[0], s6[2]
        vx, vz = s6[3], s6[5]
        if vx > 0:
            t_board = (OCHE_TO_BOARD_X_M - rx) / vx
            ts = np.linspace(0, t_board, 120)
            xs = rx + vx * ts
            zs = rz + vz * ts - 0.5 * 9.81 * ts ** 2
            ax.plot(xs, zs, "r--", lw=1.2, alpha=0.7, label="dart flight path")
            ax.plot(xs[-1], zs[-1], "r*", ms=10, zorder=7,
                    label=f"board impact  Δz={( zs[-1]-BULLSEYE_CENTER_Z_M)*1e3:+.0f} mm")

    ax.set_xlabel("x  (m toward board)")
    ax.set_ylabel("z  (m above floor)")
    ax.set_title("A4 — arm throw stick figure (x-z plane, MuJoCo sim)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4)

    out_path = _ARTIFACTS / "A4_arm_throw_stick_figure_SPEC.png"
    if save:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def render_throw_video_SPEC(
    sim_result: dict,
    out_path: Path | None = None,
    fps: int = 60,
    width: int = 640,
    height: int = 480,
) -> Path:
    """
    Replay the stored joint-angle trajectory through MuJoCo offscreen renderer
    and write an MP4.  No display required (WSL2-safe).

    The sim runs at ~1000 Hz internally; we downsample to `fps` by replaying
    every frame in the stored `q` array (already at 1 kHz) and writing each
    frame to the video.  At fps=60 the 0.2 s throw becomes ~3.3 s of video.

    Returns the path of the written file.
    """
    if out_path is None:
        out_path = _ARTIFACTS / "A4_arm_throw_video_SPEC.mp4"

    q_traj = sim_result["q"]          # (T, 3) joint angles
    times  = sim_result["times"]      # (T,)

    m = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    d = mujoco.MjData(m)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.0, 1.25]
    cam.distance  = 1.4
    cam.azimuth   = 90.0    # side view (x-z plane)
    cam.elevation = -10.0

    renderer = mujoco.Renderer(m, height=height, width=width)

    # Write every stored frame; the sim runs at ~1 kHz so ~200 frames for a 0.2 s throw.
    # At fps=60 that gives ~3 s of slow-motion playback of a 0.2 s real throw.
    frame_indices = list(range(len(times)))

    frames = []
    for fi in frame_indices:
        d.qpos[:3] = q_traj[fi]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())

    renderer.close()

    imageio.mimwrite(str(out_path), frames, fps=fps)
    print(f"  Saved video ({len(frames)} frames, {fps} fps): {out_path}")
    return out_path


if __name__ == "__main__":
    workspace_grid_SPEC()
    result = sample_throw_speed_SPEC()
    print("--- A4 stick-figure render ---")
    plot_arm_throw_SPEC(result)
    print("--- A4 video render ---")
    render_throw_video_SPEC(result)
