"""
motion_link_geometry_and_inertia_SPEC.py
=====================================
**PDF section:** Track A — A1 (Link geometry and physical parameters)

This file is **documentation + pure numeric helpers** for the **3-link, 4-DOF** arm
(shoulder yaw + shoulder pitch + elbow + wrist).
The **authoritative kinematics for simulation** are duplicated consistently in:
- `motion_mujoco_mjcf_4dof_arm_SPEC.xml` (MuJoCo model)
- `motion_controller_cubic_spline_pd_SPEC.py` (Python FK for spline/PD without MuJoCo)

**Design rationale (PDF intro):** 3-link arm (Lawrence et al. UAI 2003) with sagittal hinges
plus an added **shoulder yaw** for frontal-plane aim; shoulder mount fixed at **oche**.

**Link table (PDF A1):**
| Link     | Length (m) | Mass (kg) | Notes                          |
|----------|------------|-----------|--------------------------------|
| Upper arm| 0.30       | 2.1       | capsule/cylinder geom          |
| Forearm  | 0.26       | 1.2       |                                |
| Hand     | 0.08       | 0.4       | includes dart ~22 g            |

**Joint limits (PDF A1 + 4-DOF extension, degrees):**
- Shoulder yaw (world z): -55° to 55°
- Shoulder (flex/ext): -60° to 180°
- Elbow (flex): 0° to 145°
- Wrist (flex/ext): -70° to 70°

**Shoulder mount (PDF A1):** fixed in world at **(0, 0, 1.50) m**, facing board at **x = 2.37 m**.

**Inertia (PDF A1):** uniform capsule about center: I = (1/12) m L².
"""

from __future__ import annotations

import math

from dartrobot.constants import (
    FOREARM_LENGTH_M,
    FOREARM_MASS_KG,
    HAND_LENGTH_M,
    HAND_MASS_KG,
    SHOULDER_YAW_LIMIT_RAD,
    SHOULDER_LIMIT_RAD,
    ELBOW_LIMIT_RAD,
    WRIST_LIMIT_RAD,
    SHOULDER_MOUNT_X_M,
    SHOULDER_MOUNT_Y_M,
    SHOULDER_MOUNT_Z_M,
    UPPER_ARM_LENGTH_M,
    UPPER_ARM_MASS_KG,
    capsule_inertia_about_center,
)


def link_table_SPEC():
    """Return a list of dicts describing each link (for logging / tools)."""
    return [
        {"name": "upper_arm", "L_m": UPPER_ARM_LENGTH_M, "m_kg": UPPER_ARM_MASS_KG},
        {"name": "forearm", "L_m": FOREARM_LENGTH_M, "m_kg": FOREARM_MASS_KG},
        {"name": "hand", "L_m": HAND_LENGTH_M, "m_kg": HAND_MASS_KG},
    ]


def shoulder_mount_xyz_SPEC():
    """World-frame shoulder mount (m). **PDF A1.**"""
    return (SHOULDER_MOUNT_X_M, SHOULDER_MOUNT_Y_M, SHOULDER_MOUNT_Z_M)


def joint_limits_rad_SPEC():
    """Tuple of (low, high) radians per joint (yaw, shoulder, elbow, wrist)."""
    return (SHOULDER_YAW_LIMIT_RAD, SHOULDER_LIMIT_RAD, ELBOW_LIMIT_RAD, WRIST_LIMIT_RAD)


def link_inertias_SPEC():
    """(I_upper, I_fore, I_hand) about each link COM, kg·m², capsule formula **PDF A1.**"""
    return (
        capsule_inertia_about_center(UPPER_ARM_MASS_KG, UPPER_ARM_LENGTH_M),
        capsule_inertia_about_center(FOREARM_MASS_KG, FOREARM_LENGTH_M),
        capsule_inertia_about_center(HAND_MASS_KG, HAND_LENGTH_M),
    )


def clamp_joint_vector_SPEC(q):
    """Clamp (q_yaw, q_shoulder, q_elbow, q_wrist) to joint limits (radians)."""
    import numpy as np

    q = np.asarray(q, dtype=float).reshape(4)
    lo = np.array(
        [
            SHOULDER_YAW_LIMIT_RAD[0],
            SHOULDER_LIMIT_RAD[0],
            ELBOW_LIMIT_RAD[0],
            WRIST_LIMIT_RAD[0],
        ],
        dtype=float,
    )
    hi = np.array(
        [
            SHOULDER_YAW_LIMIT_RAD[1],
            SHOULDER_LIMIT_RAD[1],
            ELBOW_LIMIT_RAD[1],
            WRIST_LIMIT_RAD[1],
        ],
        dtype=float,
    )
    return np.clip(q, lo, hi)
