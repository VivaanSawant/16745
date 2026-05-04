"""
VALIDATION_C9_pre_rl_handoff_SPEC.py
====================================
Phase 5 of pre-RL confidence plan: compile final handoff package.

Run:
  python track_C_integration_SPEC/VALIDATION_C9_pre_rl_handoff_SPEC.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from track_C_integration_SPEC.VALIDATION_C5_deterministic_lockdown_SPEC import run_deterministic_lockdown_SPEC
from track_C_integration_SPEC.VALIDATION_C6_uncertainty_calibration_99ci_SPEC import run_uncertainty_calibration_99ci_SPEC
from track_C_integration_SPEC.VALIDATION_C7_spin_contact_confidence_SPEC import run_spin_contact_confidence_SPEC
from track_C_integration_SPEC.VALIDATION_C8_stress_campaign_gate_SPEC import run_stress_campaign_gate_SPEC

_ARTIFACTS = _ROOT / "artifacts_SPEC"
_ARTIFACTS.mkdir(exist_ok=True)


def run_pre_rl_handoff_SPEC() -> dict:
    c5 = run_deterministic_lockdown_SPEC()
    c6 = run_uncertainty_calibration_99ci_SPEC()
    c7 = run_spin_contact_confidence_SPEC()
    c8 = run_stress_campaign_gate_SPEC()

    recommended_randomization = {
        "torque_noise_sigma_add": [0.4, 0.6],
        "torque_noise_sigma_mult": [0.015, 0.025],
        "wind_x_mps": [-0.25, 0.25],
        "wind_y_mps": [-0.15, 0.15],
        "release_time_jitter_s": [-0.008, 0.008],
        "spin_c_lift": [0.004, 0.008],
        "axial_spin_rad_s": [0.0, 40.0],
    }

    all_phases_pass = bool(c5["phase_pass"] and c6["phase_pass"] and c7["phase_pass"] and c8["global_gate_pass"])
    out = {
        "phase_status": {
            "C5_deterministic_lockdown": bool(c5["phase_pass"]),
            "C6_uncertainty_calibration_99ci": bool(c6["phase_pass"]),
            "C7_spin_contact_confidence": bool(c7["phase_pass"]),
            "C8_stress_campaign_gate": bool(c8["global_gate_pass"]),
        },
        "all_phases_pass": all_phases_pass,
        "recommended_randomization_bounds": recommended_randomization,
        "residual_risks": [
            "release-time proxy channel may be weak in some seeds; keep lower control authority than wrist channel",
            "covariance linearization remains local; monitor off-nominal drift during RL domain randomization",
            "stress gate margins should be re-evaluated after any controller retune",
        ],
        "assumptions_locked": [
            "axial-spin-first model is the production spin model",
            "feedforward+PD and direct-PD are both kept as valid controller baselines",
            "tall-arm MJCF is primary for pre-RL studies",
        ],
    }

    out_json = _ARTIFACTS / "C9_pre_rl_handoff_SPEC.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# C9 pre-RL handoff",
        "",
        "## Phase status",
        "",
        f"- C5 deterministic lockdown: `{out['phase_status']['C5_deterministic_lockdown']}`",
        f"- C6 uncertainty calibration 99CI: `{out['phase_status']['C6_uncertainty_calibration_99ci']}`",
        f"- C7 spin contact confidence: `{out['phase_status']['C7_spin_contact_confidence']}`",
        f"- C8 stress campaign gate: `{out['phase_status']['C8_stress_campaign_gate']}`",
        f"- RL start gate (all phases pass): `{all_phases_pass}`",
        "",
        "## Locked assumptions",
        "",
    ]
    for item in out["assumptions_locked"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended RL randomization bounds", ""])
    for k, v in recommended_randomization.items():
        lines.append(f"- {k}: `{v[0]}` to `{v[1]}`")
    lines.extend(["", "## Residual risks", ""])
    for item in out["residual_risks"]:
        lines.append(f"- {item}")
    out_md = _ARTIFACTS / "C9_pre_rl_handoff_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("--- C9 pre-RL handoff ---")
    print(f"RL start gate: {all_phases_pass}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return out


if __name__ == "__main__":
    run_pre_rl_handoff_SPEC()

