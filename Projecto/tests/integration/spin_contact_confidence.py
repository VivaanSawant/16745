"""
VALIDATION_integration_spin_contact_confidence_SPEC.py
=============================================
Phase 3 of pre-RL confidence plan: spin + contact manipulation confidence.

Run:
  python integration_phase_SPEC/VALIDATION_integration_spin_contact_confidence_SPEC.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dartrobot.paths import artifacts_dir_SPEC, project_root_SPEC

_ROOT = project_root_SPEC()

from dartrobot.spin.ablation import SpinStudyConfig_SPEC, run_spin_relevance_study_SPEC

_ARTIFACTS = artifacts_dir_SPEC("spin")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _ci99_SPEC(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(values, [0.5, 99.5])
    return float(lo), float(hi)


def run_spin_contact_confidence_SPEC() -> dict:
    seeds = [7, 13, 19, 29, 37]
    max_shift = []
    max_score_delta = []
    max_radial_delta = []
    wrist_slope = []
    release_time_slope = []
    wrist_radial_slope = []
    release_time_radial_slope = []

    for seed in seeds:
        cfg = SpinStudyConfig_SPEC(
            seed=seed,
            n_rollouts=120,
            proxy_eval_rollouts=160,
            c_lift=0.006,
        )
        out = run_spin_relevance_study_SPEC(cfg)
        axial_rows = out["axial_rows"]
        max_shift.append(max(float(r["landing_shift_rms_mm"]) for r in axial_rows))
        max_score_delta.append(max(abs(float(r["delta_score_vs_no_spin"])) for r in axial_rows))
        max_radial_delta.append(max(abs(float(r["delta_radial_miss_mm"])) for r in axial_rows))
        wrist_slope.append(float(out["proxy"]["wrist_slope_ci"][0]))
        release_time_slope.append(float(out["proxy"]["time_slope_ci"][0]))
        wrist_radial_slope.append(float(out["proxy"]["wrist_radial_slope_ci"][0]))
        release_time_radial_slope.append(float(out["proxy"]["time_radial_slope_ci"][0]))

    max_shift = np.asarray(max_shift, dtype=float)
    max_score_delta = np.asarray(max_score_delta, dtype=float)
    max_radial_delta = np.asarray(max_radial_delta, dtype=float)
    wrist_slope = np.asarray(wrist_slope, dtype=float)
    release_time_slope = np.asarray(release_time_slope, dtype=float)
    wrist_radial_slope = np.asarray(wrist_radial_slope, dtype=float)
    release_time_radial_slope = np.asarray(release_time_radial_slope, dtype=float)

    shift_ci = _ci99_SPEC(max_shift)
    score_ci = _ci99_SPEC(max_score_delta)
    radial_ci = _ci99_SPEC(max_radial_delta)
    wrist_ci = _ci99_SPEC(wrist_slope)
    rel_time_ci = _ci99_SPEC(release_time_slope)
    wrist_radial_ci = _ci99_SPEC(wrist_radial_slope)
    rel_time_radial_ci = _ci99_SPEC(release_time_radial_slope)

    wrist_significant = not (wrist_ci[0] <= 0.0 <= wrist_ci[1])
    release_time_significant = not (rel_time_ci[0] <= 0.0 <= rel_time_ci[1])
    spin_relevant = bool(shift_ci[0] > 15.0 or score_ci[0] > 2.0 or radial_ci[0] > 12.0)
    phase_pass = bool(spin_relevant and wrist_significant)

    status = {
        "phase": "spin_contact_confidence",
        "seeds": seeds,
        "max_shift_mm_ci99": shift_ci,
        "max_score_delta_ci99": score_ci,
        "max_radial_delta_mm_ci99": radial_ci,
        "wrist_slope_ci99": wrist_ci,
        "release_time_slope_ci99": rel_time_ci,
        "wrist_radial_slope_ci99": wrist_radial_ci,
        "release_time_radial_slope_ci99": rel_time_radial_ci,
        "wrist_significant": wrist_significant,
        "release_time_significant": release_time_significant,
        "spin_relevant": spin_relevant,
        "phase_pass": phase_pass,
    }

    out_json = _ARTIFACTS / "integration_spin_contact_confidence_SPEC.json"
    out_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# Integration spin + contact confidence",
        "",
        f"- seeds: `{seeds}`",
        f"- spin relevant (99% gate): `{spin_relevant}`",
        f"- wrist slope significant: `{wrist_significant}`",
        f"- release-time slope significant: `{release_time_significant}`",
        f"- phase pass: `{phase_pass}`",
        "",
        "## 99% CI summary",
        "",
        f"- max shift mm CI99: `{shift_ci[0]:.2f}` to `{shift_ci[1]:.2f}`",
        f"- max |score delta| CI99: `{score_ci[0]:.2f}` to `{score_ci[1]:.2f}`",
        f"- max |radial delta| mm CI99: `{radial_ci[0]:.2f}` to `{radial_ci[1]:.2f}`",
        f"- wrist slope CI99: `{wrist_ci[0]:.3f}` to `{wrist_ci[1]:.3f}`",
        f"- release-time slope CI99: `{rel_time_ci[0]:.3f}` to `{rel_time_ci[1]:.3f}`",
        f"- wrist radial slope CI99: `{wrist_radial_ci[0]:.3f}` to `{wrist_radial_ci[1]:.3f}`",
        f"- release-time radial slope CI99: `{rel_time_radial_ci[0]:.3f}` to `{rel_time_radial_ci[1]:.3f}`",
        "",
        "Interpretation:",
        "- Wrist channel is accepted if CI99 excludes zero.",
        "- Release-time channel can be down-weighted if CI99 crosses zero.",
    ]
    out_md = _ARTIFACTS / "integration_spin_contact_confidence_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("--- C7 spin + contact confidence ---")
    print(f"Phase pass: {phase_pass} | wrist significant={wrist_significant} | release-time significant={release_time_significant}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return status


if __name__ == "__main__":
    run_spin_contact_confidence_SPEC()

