"""
3D projectile physics for a dart: point mass with gravity, drag, and wind.
All in SI (m, s, m/s). Board is in the x-y plane at z=0; z is up; gravity -z.
"""
import numpy as np
from scipy.integrate import solve_ivp

# Default constants
G = 9.81
RHO = 1.225
CD = 0.5
AREA = 3e-4  # approximate dart cross-section m^2
MASS = 0.025  # ~25 g


def drag_accel(vx, vy, vz, wind_x=0.0, wind_y=0.0, wind_z=0.0, rho=RHO, cd=CD, area=AREA, mass=MASS):
    """Acceleration due to drag. v is velocity, wind is wind vector."""
    vrx = vx - wind_x
    vry = vy - wind_y
    vrz = vz - wind_z
    vmag = np.sqrt(vrx * vrx + vry * vry + vrz * vrz)
    if vmag < 1e-10:
        return 0.0, 0.0, 0.0
    # F_drag = -0.5 * rho * Cd * A * |v|^2 * v_hat
    scale = -0.5 * rho * cd * area * vmag / mass
    return scale * vrx, scale * vry, scale * vrz


def _deriv(t, state, wind, rho, cd, area, mass):
    x, y, z, vx, vy, vz = state
    ax, ay, az = drag_accel(vx, vy, vz, wind[0], wind[1], wind[2], rho, cd, area, mass)
    return [vx, vy, vz, ax, ay, az - G]


def throw_3d(
    pos0,
    vel0,
    wind=(0.0, 0.0, 0.0),
    board_z=0.0,
    rho=RHO,
    cd=CD,
    area=AREA,
    mass=MASS,
    max_time=2.0,
    dt=1e-3,
):
    """
    Integrate dart trajectory until z hits board plane (z=board_z).
    pos0, vel0: (3,) or (x,y,z), (vx,vy,vz).
    Returns (t, x, y, z, vx, vy, vz) arrays; final (x,y) is landing position.
    """
    state0 = np.concatenate([np.asarray(pos0, dtype=float).ravel(), np.asarray(vel0, dtype=float).ravel()])
    wind = np.asarray(wind, dtype=float).ravel()

    def hit_board(t, state, *args):
        return state[2] - board_z

    hit_board.terminal = True
    hit_board.direction = -1

    sol = solve_ivp(
        _deriv,
        (0, max_time),
        state0,
        args=(wind, rho, cd, area, mass),
        dense_output=True,
        events=hit_board,
        max_step=dt,
    )
    if sol.t_events[0].size == 0:
        t_end = min(max_time, sol.t[-1])
        ts = np.arange(0, t_end + dt, dt)
        state = sol.sol(ts)
    else:
        t_end = float(sol.t_events[0][0])
        ts = np.linspace(0, t_end, max(2, int(t_end / dt) + 1))
        state = sol.sol(ts)
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]
    return ts, x, y, z, vx, vy, vz


def landing_position(pos0, vel0, wind=(0.0, 0.0, 0.0), board_z=0.0, **kwargs):
    """Convenience: return final (x, y) on board."""
    _, x, y, z, _, _, _ = throw_3d(pos0, vel0, wind=wind, board_z=board_z, **kwargs)
    return float(x[-1]), float(y[-1])
