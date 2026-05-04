"""
VALIDATION_C5_deterministic_lockdown_SPEC.py
============================================
Phase 1 of pre-RL confidence plan: deterministic correctness lock.

Run:
  python track_C_integration_SPEC/VALIDATION_C5_deterministic_lockdown_SPEC.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_WRIST_DEG,
)
from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
    DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
    simulate_throw_mujoco_SPEC,
)
from track_A_arm_SPEC.VALIDATION_A3_regression_suite_SPEC import (
    fk_position_parity_SPEC,
    fk_velocity_parity_SPEC,
    golden_rollout_and_torque_feasibility_SPEC,
)

_ARTIFACTS = _ROOT / "artifacts_SPEC"
_ARTIFACTS.mkdir(exist_ok=True)
_Q_START = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
_XML_STANDARD = _ROOT / "track_A_arm_SPEC" / "A2_mujoco_mjcf_3link_arm_SPEC.xml"
_XML_TALL = _ROOT / "track_A_arm_SPEC" / "A2_mujoco_mjcf_3link_arm_TALL_SPEC.xml"


def _deterministic_matrix_run_SPEC() -> list[dict]:
    rows = []
    for xml_name, xml_path in (("standard", _XML_STANDARD), ("tall", _XML_TALL)):
        for use_feedforward in (False, True):
            for release_time in (0.0761809061781466, DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC):
                out = simulate_throw_mujoco_SPEC(
                    DEFAULT_MUJOCO_OVERHAND_THROW_KNOTS_SPEC,
                    q_start3=_Q_START.copy(),
                    xml_path=xml_path,
                    rng=np.random.default_rng(123),
                    torque_noise=False,
                    release_time_s=release_time,
                    kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
                    kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
                    use_feedforward=use_feedforward,
                    inertia_ff_diag=np.array([1.0, 0.8, 0.3], dtype=float),
                    enforce_joint_limits=True,
                )
                s6 = out["release_state6"]
                if s6 is None:
                    rows.append({
                        "xml": xml_name,
                        "use_feedforward": bool(use_feedforward),
                        "release_time_s": float(release_time),
                        "pass": False,
                        "reason": "no_release",
                    })
                    continue
                speed = float(np.linalg.norm(s6[3:6]))
                pass_row = bool(s6[3] > 0.0 and 4.5 <= speed <= 7.5)
                rows.append({
                    "xml": xml_name,
                    "use_feedforward": bool(use_feedforward),
                    "release_time_s": float(release_time),
                    "release_speed_mps": speed,
                    "release_vx_mps": float(s6[3]),
                    "pass": pass_row,
                    "reason": "ok" if pass_row else "speed_or_vx_out_of_range",
                })
    return rows


def run_deterministic_lockdown_SPEC() -> dict:
    checks = {}
    checks["fk_position_error_inf_m"] = fk_position_parity_SPEC()
    checks["fk_velocity_error_inf_mps"] = fk_velocity_parity_SPEC()
    golden = golden_rollout_and_torque_feasibility_SPEC()
    checks["golden_rel_release"] = float(golden["rel_release"])
    checks["golden_rel_q"] = float(golden["rel_q"])
    checks["golden_max_sat_ratio"] = float(np.max(golden["sat_ratio"]))
    matrix_rows = _deterministic_matrix_run_SPEC()
    pass_rate = float(np.mean([1.0 if r["pass"] else 0.0 for r in matrix_rows])) if matrix_rows else 0.0

    status = {
        "phase": "deterministic_lockdown",
        "checks": checks,
        "matrix_rows": matrix_rows,
        "matrix_pass_rate": pass_rate,
        "phase_pass": bool(pass_rate >= 1.0),
    }

    out_json = _ARTIFACTS / "C5_deterministic_lockdown_SPEC.json"
    out_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# C5 deterministic lockdown",
        "",
        f"- matrix pass rate: `{pass_rate:.3f}`",
        f"- phase pass: `{status['phase_pass']}`",
        "",
        "## Core deterministic checks",
        "",
        f"- FK position inf error (m): `{checks['fk_position_error_inf_m']:.6f}`",
        f"- FK velocity inf error (m/s): `{checks['fk_velocity_error_inf_mps']:.6f}`",
        f"- Golden release relative drift: `{checks['golden_rel_release']:.6f}`",
        f"- Golden final-q relative drift: `{checks['golden_rel_q']:.6f}`",
        f"- Golden max saturation ratio: `{checks['golden_max_sat_ratio']:.3f}`",
        "",
        "## Environment matrix",
        "",
        "| xml | use_feedforward | release_time_s | release_speed_mps | release_vx_mps | pass | reason |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in matrix_rows:
        lines.append(
            f"| {r['xml']} | {int(r['use_feedforward'])} | {r['release_time_s']:.6f} | "
            f"{r.get('release_speed_mps', float('nan')):.3f} | {r.get('release_vx_mps', float('nan')):.3f} | "
            f"{int(r['pass'])} | {r['reason']} |"
        )
    out_md = _ARTIFACTS / "C5_deterministic_lockdown_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("--- C5 deterministic lockdown ---")
    print(f"Phase pass: {status['phase_pass']} | matrix pass rate={pass_rate:.2%}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return status


if __name__ == "__main__":
    run_deterministic_lockdown_SPEC()

