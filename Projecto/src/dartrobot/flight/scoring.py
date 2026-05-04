"""
flight_dartboard_scoring_radial_angular_SPEC.py
============================================
**PDF section:** Track B — B3 (Dartboard scoring geometry)

**Inputs:** Landing relative to bullseye in **meters**:
- Δy = y_impact (m)
- Δz = z_impact - 1.73 (m)  (see **B2**)

**Internal:** convert to mm, compute r = √(Δy² + Δz²), classify **rings** then **wedge** (18° each).

**PDF segment order (clockwise from top):**
20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5

**Angle (PDF):** θ = atan2(Δy, Δz) with segment **20** centered at **12 o'clock** (θ = 0).
Boundaries: 20|5 at θ = -9°, 20|1 at θ = +9°.

**Scores:** double bull 50, single bull 25, treble 3×n, double 2×n, single n, miss 0.
"""

from __future__ import annotations

import math

from dartrobot.constants import (
    R_BOARD_MISS_MM,
    R_DOUBLE_INNER_MM,
    R_DOUBLE_OUTER_MM,
    R_INNER_BULL_MM,
    R_OUTER_BULL_MM,
    R_TRIPLE_INNER_MM,
    R_TRIPLE_OUTER_MM,
    SEGMENT_ORDER_CLOCKWISE_FROM_TOP,
)


def _segment_index_from_atan2_SPEC(dy_m: float, dz_m: float) -> int:
    """
    Map (Δy, Δz) to segment array index 0..19 for **SEGMENT_ORDER_CLOCKWISE_FROM_TOP**.

    Uses PDF wedge rule: width π/10 rad, segment 20 centered at atan2=0.
    """
    if abs(dy_m) < 1e-15 and abs(dz_m) < 1e-15:
        return 0  # bull center; ring logic overrides segment value
    theta = math.atan2(dy_m, dz_m)  # PDF: atan2(Δy, Δz)
    # Bin index: floor((theta + pi/20) / (2*pi/20)) wrapped to [0,19]
    step = 2 * math.pi / 20.0
    k = math.floor((theta + math.pi / 20.0) / step)
    k %= 20
    return int(k)


def score_from_deltas_SPEC(delta_y_m: float, delta_z_m: float) -> int:
    """
    Integer dartboard score (0–60) from landing **relative to bull** (meters).

    **PDF B3 ring radii (mm):** double bull ≤6.35, single bull ≤15.9, inner single < treble inner,
    treble 99–107, outer single 107–162, double 162–170, miss >170.

    Implementation detail: between 15.9 and 99 mm is **inner single** (1× segment);
    between 107 and 162 mm is **outer single**; treble and double bands as in PDF.
    """
    dy_mm = delta_y_m * 1000.0
    dz_mm = delta_z_m * 1000.0
    r = math.hypot(dy_mm, dz_mm)

    if r > R_BOARD_MISS_MM:
        return 0

    # Bulls (PDF): inner then outer
    if r <= R_INNER_BULL_MM:
        return 50
    if r <= R_OUTER_BULL_MM:
        return 25

    seg_idx = _segment_index_from_atan2_SPEC(delta_y_m, delta_z_m)
    base = SEGMENT_ORDER_CLOCKWISE_FROM_TOP[seg_idx]

    # Treble ring
    if R_TRIPLE_INNER_MM <= r <= R_TRIPLE_OUTER_MM:
        return 3 * base
    # Double ring
    if R_DOUBLE_INNER_MM <= r <= R_DOUBLE_OUTER_MM:
        return 2 * base
    # Inner single: outside outer bull, strictly inside treble inner radius
    if R_OUTER_BULL_MM < r < R_TRIPLE_INNER_MM:
        return base
    # Outer single: outside treble, inside double inner
    if R_TRIPLE_OUTER_MM < r < R_DOUBLE_INNER_MM:
        return base

    return 0


def score_from_world_impact_SPEC(y_imp_m: float, z_imp_m: float, bullseye_z_m: float = 1.73) -> int:
    """Convenience: pass raw world (y,z) at x=board; subtracts bull height."""
    return score_from_deltas_SPEC(y_imp_m, z_imp_m - bullseye_z_m)
