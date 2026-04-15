"""
VALIDATION_A4_workspace_release_speed_SPEC.py
==============================================
**PDF section:** Track A — A4 (Validation)

**Checklist:**
- Sweep joint angles within limits; record fingertip positions from **A3 FK**.
- Confirm release site can reach **forward** of shoulder with **upward** velocity component
  for some throws (qualitative workspace check).
- **Release speed 5-7 m/s:** for a sample PD throw from **A3**, print |v| at release.

Run: `python track_A_arm_SPEC/VALIDATION_A4_workspace_release_speed_SPEC.py`
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import KEYFRAME_ELBOW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_WRIST_DEG
from track_A_arm_SPEC.A1_link_geometry_and_inertia_SPEC import joint_limits_rad_SPEC
from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    release_site_fk_SPEC,
    release_site_fk_vel_SPEC,
    simulate_throw_pd_SPEC,
)


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
    print("  PDF expects board-facing positions ~1.5-2.0 m height (z); interpret vs shoulder at z=1.50")


def sample_throw_speed_SPEC():
    """Use cocked-ish start and simple knot vector to get a forward throw."""
    q0 = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
    # Knots: push shoulder forward, extend elbow slightly (radians)
    knots = np.array(
        [
            math.radians(10),
            math.radians(40),
            math.radians(70),
            math.radians(5),
            math.radians(30),
            math.radians(55),
            math.radians(0),
            math.radians(-5),
            math.radians(5),
        ]
    )
    out = simulate_throw_pd_SPEC(knots, q0, torque_noise=False)
    s6 = out["release_state6"]
    if s6 is None:
        print("A4: no release captured")
        return
    spd = float(np.linalg.norm(s6[3:6]))
    print("--- A4 release speed check ---")
    print(f"  |v_release| = {spd:.3f} m/s  (PDF target band 5 to 7 m/s)")
    print(f"  Release pos (x,y,z) = {s6[0]:.3f}, {s6[1]:.3f}, {s6[2]:.3f}")


if __name__ == "__main__":
    workspace_grid_SPEC()
    sample_throw_speed_SPEC()
