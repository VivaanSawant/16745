"""
Dartboard geometry and scoring. Standard DRA dimensions (mm).
Supports scoring any segment 1-20, bull, double, triple.
"""
import numpy as np

# DRA dimensions in meters
BOARD_RADIUS = 0.170          # outer edge of double ring (170 mm)
INNER_BULL_RADIUS = 0.00635   # 12.7mm diameter -> 6.35mm radius
OUTER_BULL_RADIUS = 0.0159    # 31.8mm diameter
TRIPLE_INNER = 0.099          # 107mm - 8mm
TRIPLE_OUTER = 0.107
DOUBLE_INNER = 0.162          # 170mm - 8mm
DOUBLE_OUTER = 0.170

# Segment order clockwise from top (20 at top). Index 0 = 20, 1 = 1, ...
SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
N_SEGMENTS = 20
SEGMENT_ANGLE = 2 * np.pi / N_SEGMENTS  # 18 deg in rad


def segment_angle_center(segment_number: int) -> float:
    """Angle (rad) to center of segment. segment_number 1-20. 0 = top (segment 20), clockwise."""
    # Segment 20 is at top (pi/2 in usual math coords with x right, y up)
    idx = SEGMENT_ORDER.index(segment_number)
    return np.pi / 2 - idx * SEGMENT_ANGLE


def get_target_center(segment: int, ring: str = "S") -> tuple[float, float]:
    """
    Get (x, y) aim point in board frame (center = bull, x right, y up). Meters.
    segment: 1-20, or 25 for bull.
    ring: "S" single, "D" double, "T" triple. For bull use segment=25, ring ignored.
    """
    if segment == 25:
        return (0.0, 0.0)
    # Radius to middle of ring
    if ring == "S":
        r = (TRIPLE_OUTER + DOUBLE_INNER) / 2   # middle of single band
    elif ring == "D":
        r = (DOUBLE_INNER + DOUBLE_OUTER) / 2
    elif ring == "T":
        r = (TRIPLE_INNER + TRIPLE_OUTER) / 2
    else:
        r = (TRIPLE_OUTER + DOUBLE_INNER) / 2
    a = segment_angle_center(segment)
    return (r * np.cos(a), r * np.sin(a))


def score_at(x: float, y: float) -> int:
    """
    Score for a dart at (x, y) in board frame (meters). Returns 0 if miss.
    """
    r = np.sqrt(x * x + y * y)
    if r > BOARD_RADIUS:
        return 0
    theta = np.arctan2(y, x)
    # Angle from top clockwise: top = pi/2, so segment angle from top = pi/2 - theta (wrap)
    angle_from_top = (np.pi / 2 - theta + 2 * np.pi) % (2 * np.pi)
    seg_idx = int(round(angle_from_top / SEGMENT_ANGLE)) % N_SEGMENTS
    base_number = SEGMENT_ORDER[seg_idx]

    if r <= INNER_BULL_RADIUS:
        return 50
    if r <= OUTER_BULL_RADIUS:
        return 25
    if TRIPLE_INNER <= r <= TRIPLE_OUTER:
        return 3 * base_number
    if DOUBLE_INNER <= r <= DOUBLE_OUTER:
        return 2 * base_number
    if r < TRIPLE_INNER or (r > TRIPLE_OUTER and r < DOUBLE_INNER):
        return base_number
    return 0


def get_all_targets():
    """Yield (name, x, y) for common targets e.g. ('T20', x, y)."""
    for seg in range(1, 21):
        for ring, name in [("S", f"S{seg}"), ("T", f"T{seg}"), ("D", f"D{seg}")]:
            x, y = get_target_center(seg, ring)
            yield (name, x, y)
    yield ("BULL", 0.0, 0.0)
