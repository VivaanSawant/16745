"""
flight_point_mass_force_model_gravity_drag_wind_SPEC.py
====================================================
**PDF section:** Track B — B1 (Force model and physical parameters)

Implements the **point-mass dart** with:
- **Gravity:** F_g = [0, 0, -m g]  (PDF)
- **Quadratic drag:** F_drag = -½ ρ C_d A |v_rel| v_rel  with v_rel = v - v_wind (PDF)

**Coordinate convention (PDF):** x = toward board, y = horizontal left, z = up.

This module is **pure force/acceleration** — no integration (that is **B2**).
"""

from __future__ import annotations

import math
import numpy as np

from dartrobot.constants import (
    AIR_DENSITY_KG_M3,
    CROSS_SECTION_AREA_M2,
    DART_MASS_KG,
    DRAG_COEFFICIENT_CD,
)


def acceleration_total_SPEC(
    vx: float,
    vy: float,
    vz: float,
    wind_xyz_mps=(0.0, 0.0, 0.0),
    *,
    mass_kg: float = DART_MASS_KG,
    rho: float = AIR_DENSITY_KG_M3,
    cd: float = DRAG_COEFFICIENT_CD,
    area_m2: float = CROSS_SECTION_AREA_M2,
    gravity_m_s2: float = 9.81,
    drag_enabled: bool = True,
) -> tuple[float, float, float]:
    """
    Total linear acceleration (ax, ay, az) in m/s² for the 6D state ODE (PDF B2).

    Parameters
    ----------
    vx, vy, vz : float
        Dart velocity (m/s), world frame.
    wind_xyz_mps : tuple
        Constant wind (wx, wy, wz) in m/s; drag uses **v_rel = v - w** (PDF B1).
    drag_enabled : bool
        If False, returns only gravity (used in **B4** drag-on vs drag-off comparison).

    **PDF references:** B1 gravity magnitude example (~0.216 N for 22 g); drag ~1.5% of g at 6 m/s.
    """
    vrx = vx - wind_xyz_mps[0]
    vry = vy - wind_xyz_mps[1]
    vrz = vz - wind_xyz_mps[2]
    vmag = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)

    ax = ay = 0.0
    az = -gravity_m_s2

    if drag_enabled and vmag > 1e-12:
        k = -0.5 * rho * cd * area_m2 * vmag / mass_kg
        ax += k * vrx
        ay += k * vry
        az += k * vrz

    return ax, ay, az


def state_derivative_SPEC(
    _t: float,
    state6: np.ndarray,
    wind_xyz_mps=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
) -> np.ndarray:
    """
    **PDF B2:** 6D state s = [x, y, z, vx, vy, vz]; ds/dt = [vx,vy,vz, ax,ay,az].

    Kept as a small callable for `scipy.integrate.solve_ivp` in **B2**.
    """
    x, y, z, vx, vy, vz = state6.tolist()
    ax, ay, az = acceleration_total_SPEC(
        vx, vy, vz, wind_xyz_mps, drag_enabled=drag_enabled
    )
    return np.array([vx, vy, vz, ax, ay, az], dtype=float)
