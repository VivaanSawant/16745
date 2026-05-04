"""
integration_target_board_regions_SPEC.py
========================================
Board targeting helpers for **sectors 20, 3, 6, 11** with a **dual-stage** curriculum:
**inner-single** radial centers first, then **treble-ring** centers.

Uses the same radial wedge convention as **B3** (`flight_dartboard_scoring_radial_angular_SPEC`).
"""

from __future__ import annotations

import math

import numpy as np

from dartrobot.constants import (
    R_OUTER_BULL_MM,
    R_TRIPLE_INNER_MM,
    R_TRIPLE_OUTER_MM,
    SEGMENT_ORDER_CLOCKWISE_FROM_TOP,
)

ILQR_TARGET_SECTORS_SPEC = (20, 3, 6, 11)


def segment_list_index_for_sector_value_SPEC(sector: int) -> int:
    """Index 0..19 in `SEGMENT_ORDER_CLOCKWISE_FROM_TOP` for a printed segment number."""
    return int(SEGMENT_ORDER_CLOCKWISE_FROM_TOP.index(int(sector)))


def wedge_center_angle_rad_SPEC(sector: int) -> float:
    """
    Center polar angle θ (radians) with **PDF B3** convention θ = atan2(Δy, Δz).

    Segment 20 is centered at θ = 0; indices advance clockwise when viewed from the thrower.
    """
    k = segment_list_index_for_sector_value_SPEC(sector)
    return float(k) * (2.0 * math.pi / 20.0)


def board_deltas_m_at_radius_SPEC(sector: int, radial_distance_m: float) -> tuple[float, float]:
    """(Δy, Δz) in metres relative to bull for `sector` at the given bull-plane radius."""
    theta = wedge_center_angle_rad_SPEC(sector)
    r = float(radial_distance_m)
    dy = r * math.sin(theta)
    dz = r * math.cos(theta)
    return dy, dz


def inner_single_center_radius_m_SPEC() -> float:
    """Geometric mid-radius of the **inner single** annulus (outside outer bull, inside treble)."""
    return 0.5 * (float(R_OUTER_BULL_MM) + float(R_TRIPLE_INNER_MM)) / 1000.0


def treble_ring_center_radius_m_SPEC() -> float:
    """Mid-radius of the **treble** scoring band."""
    return 0.5 * (float(R_TRIPLE_INNER_MM) + float(R_TRIPLE_OUTER_MM)) / 1000.0


def stage_targets_single_then_treble_SPEC(
    sectors: tuple[int, ...] = ILQR_TARGET_SECTORS_SPEC,
) -> dict:
    """
    Return structured targets for stage-1 (single/inner-single centers) and stage-2 (treble centers).

    Each sector maps to `(dy_m, dz_m)` landing deltas relative to the bull.
    """
    r_single = inner_single_center_radius_m_SPEC()
    r_treble = treble_ring_center_radius_m_SPEC()
    out_single = {}
    out_treble = {}
    for s in sectors:
        out_single[int(s)] = board_deltas_m_at_radius_SPEC(int(s), r_single)
        out_treble[int(s)] = board_deltas_m_at_radius_SPEC(int(s), r_treble)
    return {
        "sectors": tuple(int(s) for s in sectors),
        "inner_single_radius_m": r_single,
        "treble_radius_m": r_treble,
        "stage_single_deltas_m": out_single,
        "stage_treble_deltas_m": out_treble,
    }


def stage_progression_gate_passed_SPEC(
    hit_rate: float,
    mean_radial_error_mm: float,
    *,
    min_hit_rate: float = 0.22,
    max_mean_radial_mm: float = 38.0,
) -> bool:
    """
    Promote from **single-stage** to **treble-stage** when dispersion is modest.

    Thresholds are soft engineering defaults (tunable for curriculum / RL gates).
    """
    if not math.isfinite(mean_radial_error_mm):
        return False
    return bool(hit_rate >= min_hit_rate and mean_radial_error_mm <= max_mean_radial_mm)


def workspace_goal_xyz_from_board_deltas_SPEC(
    delta_y_m: float,
    delta_z_m: float,
    shoulder_mount_y_m: float = 0.0,
    shoulder_mount_z_m: float = 1.50,
    shoulder_mount_x_m: float = 0.0,
) -> np.ndarray:
    """
    Map desired board-plane landing offsets (Δy, Δz) to a **soft** fingertip workspace goal.

    Used by the analytic iLQR scaffold as a proxy for terminal end-effector positioning.
    """
    scale_y = 0.52
    scale_z = 0.62
    scale_x = 0.38
    r = float(math.hypot(delta_y_m, delta_z_m))
    x_g = shoulder_mount_x_m + 0.22 + scale_x * r
    y_g = shoulder_mount_y_m + scale_y * float(delta_y_m)
    z_g = shoulder_mount_z_m + scale_z * float(delta_z_m)
    return np.array([x_g, y_g, z_g], dtype=float)
