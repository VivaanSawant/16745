"""
Hit any given number: from target (e.g. T20, S19, D5, BULL) to aim point,
and simple launch model so we can compute release velocity to aim at a point.
"""
import numpy as np
from . import dartboard
from .projectile import throw_3d, landing_position


def parse_target(name: str) -> tuple[int, str]:
    """
    Parse target string like 'T20', 'S19', 'D5', 'BULL' -> (segment, ring).
    Returns (25, 'S') for BULL; (20, 'T') for T20, etc.
    """
    name = name.strip().upper()
    if name == "BULL" or name == "25" or name == "50":
        return (25, "S")
    if len(name) < 2:
        raise ValueError(f"Invalid target: {name}")
    ring = name[0]
    if ring not in "SDT":
        raise ValueError(f"Unknown ring {ring} in {name}")
    seg = int(name[1:])
    if seg < 1 or seg > 20:
        raise ValueError(f"Segment must be 1-20, got {seg}")
    return (seg, ring)


def aim_point(target_name: str) -> tuple[float, float]:
    """Get (x, y) aim point in board frame for target e.g. 'T20', 'S19', 'BULL'."""
    seg, ring = parse_target(target_name)
    return dartboard.get_target_center(seg, ring)


def release_velocity_to_point(pos0, point_xy, board_z=0.0, gravity=9.81, v_mag=5.0):
    """
    Simple launch: assume flat throw (no spin) so trajectory is in a vertical plane.
    Compute initial velocity direction so that without drag we'd hit point_xy.
    Then scale to v_mag. With drag, landing will differ; use as initial guess for optimization.
    """
    x0, y0, z0 = np.asarray(pos0).ravel()[:3]
    x1, y1 = point_xy[0], point_xy[1]
    dx, dy = x1 - x0, y1 - y0
    dz = board_z - z0
    # Ballistic (no drag): vx = dx/t, vy = dy/t, vz = dz/t - 0.5*g*t so t from vz
    # Simpler: set horizontal direction and choose t so we reach (x1,y1) and z=board_z.
    dist_h = np.sqrt(dx * dx + dy * dy)
    if dist_h < 1e-6:
        return np.array([0.0, 0.0, np.sqrt(2 * gravity * max(dz, 0.01))])
    t_est = np.sqrt(2 * max(dz, 0.01) / gravity) if dz > 0 else dist_h / v_mag
    vx = dx / t_est
    vy = dy / t_est
    vz = (dz + 0.5 * gravity * t_est * t_est) / t_est
    v = np.array([vx, vy, vz])
    n = np.linalg.norm(v)
    if n > 1e-10:
        v = v_mag * (v / n)
    return v


def find_velocity_to_hit(pos0, target_name: str, board_z=0.0, v_mag=5.0, wind=(0, 0, 0), tol=0.005):
    """
    Iterate (simple 2D bisection / scale) to get release velocity so landing is near target.
    Returns (vel0, landing_xy, score). Kept simple: one-shot ballistic guess then 1–2 refinements.
    """
    aim = aim_point(target_name)
    v = release_velocity_to_point(pos0, aim, board_z=board_z, v_mag=v_mag)
    x, y = landing_position(pos0, v, wind=wind, board_z=board_z)
    best_v, best_xy = v.copy(), (x, y)
    # Refine magnitude
    for _ in range(5):
        err = np.sqrt((x - aim[0]) ** 2 + (y - aim[1]) ** 2)
        if err < tol:
            break
        scale = 1.0 + 0.3 * (err / (np.linalg.norm(aim) + 1e-6))
        v = v * scale
        x, y = landing_position(pos0, v, wind=wind, board_z=board_z)
        if np.sqrt((x - aim[0]) ** 2 + (y - aim[1]) ** 2) < np.sqrt((best_xy[0] - aim[0]) ** 2 + (best_xy[1] - aim[1]) ** 2):
            best_v, best_xy = v.copy(), (x, y)
    score = dartboard.score_at(best_xy[0], best_xy[1])
    return best_v, best_xy, score
