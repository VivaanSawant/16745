"""
VALIDATION_flight_drag_comparison_and_plots_SPEC.py
================================================
**PDF section:** Track B — B4 (Validation)

**Checklist:**
- Compare **drag-on** vs **drag-off** for the PDF test initial condition.
- Plot **x vs z** side view with board plane at x=2.37 m.
- Plot **Δy vs Δz** board view with optional rings (foundation for MC scatter).
- **Spot-check** ≥5 coordinates from PDF against `score_from_deltas_SPEC`.

Run as script: `python flight_phase_SPEC/VALIDATION_flight_drag_comparison_and_plots_SPEC.py`
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dartrobot.paths import artifacts_dir_SPEC

from dartrobot.constants import (
    BULLSEYE_CENTER_Z_M,
    OCHE_TO_BOARD_X_M,
    VALIDATION_INITIAL_VX_MPS,
    VALIDATION_INITIAL_VY_MPS,
    VALIDATION_INITIAL_VZ_MPS,
    VALIDATION_RELEASE_XYZ_M,
)
from dartrobot.flight.integrator import integrate_until_board_SPEC
from dartrobot.flight.scoring import score_from_deltas_SPEC


def spot_checks_B3_SPEC():
    """**PDF B4** table (inputs in mm as Δy, Δz)."""
    tests = [
        ((0.0, 0.0), 50),
        ((0.0, 0.103), 60),
        ((0.0, -0.103), 9),
        ((0.103, 0.0), 18),
        ((-0.103, 0.0), 33),
    ]
    print("--- B3 spot checks (delta_y, delta_z in meters) ---")
    for (dy, dz), exp in tests:
        got = score_from_deltas_SPEC(dy, dz)
        ok = "OK" if got == exp else f"FAIL (expected {exp})"
        print(f"  ({dy:+.4f}, {dz:+.4f}) m -> score {got}  {ok}")


def drag_on_off_comparison_SPEC():
    """**PDF B4:** same IC, compare landings; difference should be a few mm, not >1 cm."""
    s6 = np.array(
        [
            VALIDATION_RELEASE_XYZ_M[0],
            VALIDATION_RELEASE_XYZ_M[1],
            VALIDATION_RELEASE_XYZ_M[2],
            VALIDATION_INITIAL_VX_MPS,
            VALIDATION_INITIAL_VY_MPS,
            VALIDATION_INITIAL_VZ_MPS,
        ],
        dtype=float,
    )
    on = integrate_until_board_SPEC(s6, drag_enabled=True)
    off = integrate_until_board_SPEC(s6, drag_enabled=False)
    if not (on["hit"] and off["hit"]):
        print("Drag comparison: miss in one or both runs")
        return
    d = math.hypot(on["delta_y_m"] - off["delta_y_m"], on["delta_z_m"] - off["delta_z_m"]) * 1000.0
    print("--- B4 drag-on vs drag-off ---")
    print(f"  Landing drag-on:  dy={on['delta_y_m']*1000:.3f} mm, dz={on['delta_z_m']*1000:.3f} mm")
    print(f"  Landing drag-off: dy={off['delta_y_m']*1000:.3f} mm, dz={off['delta_z_m']*1000:.3f} mm")
    print(f"  |delta landing| (2D) = {d:.3f} mm  (PDF: expect few mm, not >10 mm unless Cd*A wrong)")
    return {
        "drag_on_delta_y_m": float(on["delta_y_m"]),
        "drag_on_delta_z_m": float(on["delta_z_m"]),
        "drag_off_delta_y_m": float(off["delta_y_m"]),
        "drag_off_delta_z_m": float(off["delta_z_m"]),
        "landing_delta_mm": float(d),
    }


def save_plots_SPEC():
    """Save PNGs under ``artifacts/flight/`` (created on demand)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = artifacts_dir_SPEC("flight")
    out_dir.mkdir(parents=True, exist_ok=True)

    s6 = np.array(
        [
            VALIDATION_RELEASE_XYZ_M[0],
            VALIDATION_RELEASE_XYZ_M[1],
            VALIDATION_RELEASE_XYZ_M[2],
            VALIDATION_INITIAL_VX_MPS,
            VALIDATION_INITIAL_VY_MPS,
            VALIDATION_INITIAL_VZ_MPS,
        ],
        dtype=float,
    )
    sol = integrate_until_board_SPEC(s6, drag_enabled=True)["sol"]
    if sol is None or not sol.success:
        print("Plot: integration failed")
        return
    t = np.linspace(0, sol.t[-1], 200)
    y = sol.sol(t)
    x, z = y[0], y[2]

    fig, ax = plt.subplots()
    ax.plot(x, z, label="trajectory")
    ax.axvline(OCHE_TO_BOARD_X_M, color="k", ls="--", label="board x=2.37")
    ax.scatter([OCHE_TO_BOARD_X_M], [BULLSEYE_CENTER_Z_M], c="r", s=20, label="bull center")
    ax.set_xlabel("x (m) toward board")
    ax.set_ylabel("z (m) up")
    ax.set_title("B4 side view (x vs z) SPEC")
    ax.legend()
    ax.grid(True)
    side_path = out_dir / "flight_side_view_xz_SPEC.png"
    fig.savefig(side_path, dpi=140)
    plt.close(fig)

    hit = integrate_until_board_SPEC(s6, drag_enabled=True)
    fig2, ax2 = plt.subplots()
    th = np.linspace(0, 2 * math.pi, 200)
    r_board = 0.170
    ax2.plot(r_board * np.cos(th), r_board * np.sin(th), "k-")
    ax2.scatter([hit["delta_y_m"]], [hit["delta_z_m"]], c="b", s=40, label="landing (Δy,Δz)")
    ax2.set_aspect("equal")
    ax2.set_xlabel("Δy (m)")
    ax2.set_ylabel("Δz (m)")
    ax2.set_title("B4 board view SPEC")
    ax2.grid(True)
    ax2.legend()
    board_path = out_dir / "flight_board_view_deltas_SPEC.png"
    fig2.savefig(board_path, dpi=140)
    plt.close(fig2)
    print(f"Saved: {side_path}")
    print(f"Saved: {board_path}")
    return {
        "side_view_path": side_path,
        "board_view_path": board_path,
        "landing_delta_y_m": float(hit["delta_y_m"]),
        "landing_delta_z_m": float(hit["delta_z_m"]),
    }


def _write_manifest_SPEC(drag_cmp: dict, plot_info: dict) -> Path:
    out_dir = artifacts_dir_SPEC("flight")
    manifest = {
        "stage": "flight_projectile_validation",
        "metrics": {
            **drag_cmp,
            "landing_delta_y_m": float(plot_info["landing_delta_y_m"]),
            "landing_delta_z_m": float(plot_info["landing_delta_z_m"]),
        },
        "artifacts": [
            {
                "path": str(plot_info["side_view_path"]),
                "caption": "B4 side view trajectory (x-z) with board plane reference.",
            },
            {
                "path": str(plot_info["board_view_path"]),
                "caption": "B4 board-frame landing point (Δy, Δz).",
            },
        ],
    }
    manifest_path = out_dir / "flight_demo_manifest_SPEC.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    spot_checks_B3_SPEC()
    drag = drag_on_off_comparison_SPEC()
    plots = save_plots_SPEC()
    _write_manifest_SPEC(drag, plots)
