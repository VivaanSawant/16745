"""
C2_jacobian_covariance_optimizer_SPEC.py
==========================================
**PDF section:** Track C — C2 (Analysis layers)

**Jacobian (PDF):** ∂(landing Δy, Δz) / ∂(release state) with release state the **6 scalars**
(x0, y0, z0, vx0, vy0, vz0) at release — computed here with **finite differences** (ε small).

**Predicted landing covariance (PDF):** Σ_land ≈ J Σ_release Jᵀ (first-order), compare to MC scatter.

**Optimizer (PDF):** `scipy.optimize.minimize` (Nelder–Mead) with objective = **negative expected score**
from Monte Carlo — **stub** provided; warm-start and target generalization are project extensions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from track_B_projectile_SPEC.B2_ode_integrator_event_board_plane_SPEC import integrate_until_board_SPEC


def landing_deltas_for_release_SPEC(s6: np.ndarray, wind_xyz=(0.0, 0.0, 0.0), drag_enabled: bool = True) -> np.ndarray:
    """Return (Δy, Δz) in meters or (0,0) if miss."""
    s6 = np.asarray(s6, dtype=float).reshape(6)
    info = integrate_until_board_SPEC(s6, wind_xyz_mps=wind_xyz, drag_enabled=drag_enabled)
    if not info["hit"]:
        return np.array([0.0, 0.0])
    return np.array([info["delta_y_m"], info["delta_z_m"]], dtype=float)


def jacobian_landing_wrt_release_SPEC(
    s6: np.ndarray,
    eps: float = 1e-4,
    wind_xyz=(0.0, 0.0, 0.0),
    drag_enabled: bool = True,
) -> np.ndarray:
    """
    **PDF C2:** 2×6 Jacobian J where rows are [∂Δy, ∂Δz] and cols are ∂/∂ each release component.
    """
    s6 = np.asarray(s6, dtype=float).reshape(6)
    base = landing_deltas_for_release_SPEC(s6, wind_xyz, drag_enabled)
    J = np.zeros((2, 6))
    for j in range(6):
        sp = s6.copy()
        sp[j] += eps
        d = landing_deltas_for_release_SPEC(sp, wind_xyz, drag_enabled)
        J[:, j] = (d - base) / eps
    return J


def predicted_landing_covariance_SPEC(J: np.ndarray, Sigma_release6: np.ndarray) -> np.ndarray:
    """**PDF C2:** Σ_land = J Σ_release Jᵀ (Σ_release is 6×6)."""
    return J @ Sigma_release6 @ J.T


def minimize_negative_mc_score_stub_SPEC(
    mc_objective_fn,
    x0_9params: np.ndarray,
    maxiter: int = 30,
) -> object:
    """
    **PDF C2 (pattern):** Nelder–Mead on a **scalar** objective returned by `mc_objective_fn(x)`.

    Here `x` is an abstract vector (e.g. 9 spline knots). The caller supplies
    `mc_objective_fn` that runs MC and returns **mean score**; we minimize **negative** mean.
    """
    fun = lambda x: -float(mc_objective_fn(x))
    return minimize(fun, x0_9params, method="Nelder-Mead", options={"maxiter": maxiter, "disp": False})
