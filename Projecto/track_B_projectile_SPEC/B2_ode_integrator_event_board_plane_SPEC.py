"""
B2_ode_integrator_event_board_plane_SPEC.py
============================================
**PDF section:** Track B — B2 (ODE integrator with event detection)

- **State:** [x, y, z, vx, vy, vz] (PDF)
- **Integration:** `scipy.integrate.solve_ivp` with RK45 (default method)
- **Terminal event:** dart crosses **board plane** at **x = 2.37 m**, forward direction (+1)
  (PDF: `x(t) - 2.37 = 0`, `direction = +1`, `terminal = True`)

**Landing extraction (PDF):** At impact, board center is (2.37, 0, 1.73). Define:
- Δy = y_impact
- Δz = z_impact - 1.73

If the dart **never** reaches x = 2.37 with positive crossing, return a **miss** sentinel
(see `integrate_until_board_SPEC`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import BULLSEYE_CENTER_Z_M, OCHE_TO_BOARD_X_M
from track_B_projectile_SPEC.B1_point_mass_force_model_gravity_drag_wind_SPEC import state_derivative_SPEC


BOARD_PLANE_X_M = OCHE_TO_BOARD_X_M  # alias for readability in this file


def _event_board_plane_SPEC(t, y, wind, drag_enabled):
    """Event function g(t); zero when x - 2.37 == 0 (PDF)."""
    return y[0] - BOARD_PLANE_X_M


def integrate_until_board_SPEC(
    release_state6: np.ndarray,
    wind_xyz_mps=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
    max_time_s: float = 3.0,
    rtol=1e-8,
    atol=1e-10,
):
    """
    Integrate from release until **first forward crossing** of x = 2.37 m.

    Parameters
    ----------
    release_state6 : (6,) array
        [x0, y0, z0, vx0, vy0, vz0] at release (m, m/s).

    Returns
    -------
    result : dict with keys:
        - ``hit``: bool — True if event fired with vx>0 at crossing (PDF forward hit)
        - ``t_hit``: float or None
        - ``state_hit``: (6,) at crossing or last state if miss
        - ``delta_y_m``, ``delta_z_m``: landing relative to bullseye (0,0 if miss)
        - ``sol``: scipy OdeSolution or None
    """
    release_state6 = np.asarray(release_state6, dtype=float).reshape(6)
    wind = tuple(float(w) for w in wind_xyz_mps)

    def rhs(t, y):
        return state_derivative_SPEC(t, y, wind_xyz_mps=wind, drag_enabled=drag_enabled)

    ev = lambda t, y: _event_board_plane_SPEC(t, y, wind, drag_enabled)
    ev.terminal = True
    ev.direction = 1.0  # PDF: crossing in +x direction

    sol = solve_ivp(
        rhs,
        (0.0, max_time_s),
        release_state6,
        events=ev,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        method="RK45",
    )

    miss = {
        "hit": False,
        "t_hit": None,
        "state_hit": sol.y[:, -1] if sol.y.size else release_state6,
        "delta_y_m": 0.0,
        "delta_z_m": 0.0,
        "sol": sol,
    }

    if not sol.t_events or len(sol.t_events[0]) == 0:
        return miss

    t_hit = float(sol.t_events[0][0])
    y_hit = sol.sol(t_hit)
    vx_hit = float(y_hit[3])
    if vx_hit <= 0.0:
        return miss

    y_imp = float(y_hit[1])
    z_imp = float(y_hit[2])
    delta_y_m = y_imp - 0.0  # PDF: Δy = y_impact (origin y=0 at bull)
    delta_z_m = z_imp - BULLSEYE_CENTER_Z_M

    return {
        "hit": True,
        "t_hit": t_hit,
        "state_hit": y_hit,
        "delta_y_m": delta_y_m,
        "delta_z_m": delta_z_m,
        "sol": sol,
    }
