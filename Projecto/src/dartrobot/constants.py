"""
SPEC_QUICK_REFERENCE_constants.py
===================================
Single source of truth for numeric constants from **dart_robot_tasklist.pdf**
(CMU 16-745 Optimal Control and RL — Dart-Throwing Robot).

**PDF section:** "Quick reference: physical constants and parameters" (page 4)
plus Track A geometry (pages 1–2) where noted. **4-DOF extension:** shoulder yaw limits,
keyframe yaw, yaw torque limit, and `N_POLICY_PARAMETERS = 12` (4 joints × 3 knots).

Import this module in Track A / B / C code so parameters are not duplicated.
"""

# --- World / board (PDF: Integration + B2) ---
OCHE_TO_BOARD_X_M = 2.37  # m — board plane x coordinate
BULLSEYE_CENTER_X_M = OCHE_TO_BOARD_X_M
BULLSEYE_CENTER_Y_M = 0.0
BULLSEYE_CENTER_Z_M = 1.73  # m — center of board from floor (bull height)

# --- Shoulder mount (PDF Track A1) ---
SHOULDER_MOUNT_X_M = 0.0
SHOULDER_MOUNT_Y_M = 0.0
SHOULDER_MOUNT_Z_M = 1.50  # m — slightly below bull (1.73)

# --- Track A1: link geometry (meters, kg) ---
UPPER_ARM_LENGTH_M = 0.30
UPPER_ARM_MASS_KG = 2.1
FOREARM_LENGTH_M = 0.26
FOREARM_MASS_KG = 1.2
HAND_LENGTH_M = 0.08
HAND_MASS_KG = 0.4  # includes dart ~22 g per PDF

# --- Track A1: joint limits (degrees → radians used in code) ---
import math

SHOULDER_LIMIT_DEG = (-60.0, 180.0)
# Elbow convention in this repo: q_elbow=0 means forearm is parallel to upper arm.
# Negative values correspond to a "raised" forearm (wrist above elbow), which matches
# a human dart throw better than allowing the forearm to rotate downward past parallel.
ELBOW_LIMIT_DEG = (-145.0, 0.0)
WRIST_LIMIT_DEG = (-70.0, 70.0)
# 4-DOF extension: shoulder yaw about world +z (frontal-plane aim), same 3 links.
SHOULDER_YAW_LIMIT_DEG = (-55.0, 55.0)

def _deg2rad(a, b):
    return (math.radians(a), math.radians(b))

SHOULDER_LIMIT_RAD = _deg2rad(*SHOULDER_LIMIT_DEG)
ELBOW_LIMIT_RAD = _deg2rad(*ELBOW_LIMIT_DEG)
WRIST_LIMIT_RAD = _deg2rad(*WRIST_LIMIT_DEG)
SHOULDER_YAW_LIMIT_RAD = _deg2rad(*SHOULDER_YAW_LIMIT_DEG)

# --- Track A1: capsule inertia about center (PDF): (1/12) * m * L^2 ---
def capsule_inertia_about_center(m_kg, length_m):
    return (1.0 / 12.0) * m_kg * length_m * length_m

# --- Track A2: actuators (Nm) ---
TORQUE_LIMIT_SHOULDER_YAW_NM = 55.0
TORQUE_LIMIT_SHOULDER_NM = 100.0
TORQUE_LIMIT_ELBOW_NM = 70.0
TORQUE_LIMIT_WRIST_NM = 20.0

# --- Track A2: simulation ---
MUJOCO_TIMESTEP_S = 0.001  # PDF: <option timestep="0.001">

# --- Track A3: PD + trajectory ---
PD_KP_DEFAULT = 100.0  # per joint (start value; tune in experiments)
PD_KD_DEFAULT = 10.0
THROW_DURATION_S = 0.2  # PDF: spline over ~0.2 s
N_SPLINE_KNOTS_PER_JOINT = 3  # PDF: 3 knots per joint
N_ARM_JOINTS = 4  # shoulder yaw + shoulder + elbow + wrist (3 links)
N_POLICY_PARAMETERS = N_ARM_JOINTS * N_SPLINE_KNOTS_PER_JOINT  # 12

# --- Track A3: release time noise (PDF Lawrence et al.) ---
RELEASE_TIME_SIGMA_S = 0.01

# --- Track A3: keyframe "cocked" pose (degrees, PDF A2) ---
# Overhand-throw starting posture: shoulder raised near its -60° limit so the
# arm clearly swings from above the bullseye height (z≈1.75) down to release.
KEYFRAME_SHOULDER_YAW_DEG = 0.0
KEYFRAME_SHOULDER_DEG = -55.0
KEYFRAME_ELBOW_DEG = -90.0
KEYFRAME_WRIST_DEG = -20.0

# --- Track B1: dart point mass ---
DART_MASS_KG = 0.022  # 22 g standard competition

# --- Track B1: aerodynamics ---
AIR_DENSITY_KG_M3 = 1.225  # ρ sea level ~15°C
DRAG_COEFFICIENT_CD = 0.47  # PDF: conservative sphere-like
CROSS_SECTION_AREA_M2 = 3.0e-4  # ~20 mm barrel

# --- Track B1 / B4: validation example initial velocity (m/s) ---
VALIDATION_INITIAL_VX_MPS = 5.5
VALIDATION_INITIAL_VY_MPS = 0.0
VALIDATION_INITIAL_VZ_MPS = 1.5
VALIDATION_RELEASE_XYZ_M = (0.0, 0.0, 1.73)  # PDF B4

# --- Track B3 / quick ref: dartboard radii in mm (from PDF) ---
R_INNER_BULL_MM = 6.35
R_OUTER_BULL_MM = 15.9
R_INNER_SINGLE_MAX_MM = 99.0  # inner single region inside treble inner radius
R_TRIPLE_INNER_MM = 99.0
R_TRIPLE_OUTER_MM = 107.0
R_OUTER_SINGLE_INNER_MM = 107.0  # same boundary; outer single runs to double inner
R_DOUBLE_INNER_MM = 162.0
R_DOUBLE_OUTER_MM = 170.0
R_BOARD_MISS_MM = 170.0

# --- Track C2 / quick ref ---
MC_DEFAULT_N_ROLLOUTS = 200

# --- PDF "Segment order clockwise from top" (B3) ---
SEGMENT_ORDER_CLOCKWISE_FROM_TOP = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
)
