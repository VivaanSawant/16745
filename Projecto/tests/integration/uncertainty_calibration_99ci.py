"""
VALIDATION_integration_uncertainty_calibration_99ci_SPEC.py
==================================================
Phase 2 of pre-RL confidence plan: uncertainty calibration with 99% CIs.

Run:
  python integration_phase_SPEC/VALIDATION_integration_uncertainty_calibration_99ci_SPEC.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dartrobot.paths import artifacts_dir_SPEC, mjcf_path_SPEC, project_root_SPEC

_ROOT = project_root_SPEC()

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
)
from dartrobot.spin.ablation import paired_spin_effect_metrics_SPEC
from dartrobot.motion.controller import simulate_throw_mujoco_SPEC
from dartrobot.integration.release_to_score import score_from_release_state_SPEC

_ARTIFACTS = artifacts_dir_SPEC("integration")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)
_XML_PATH = mjcf_path_SPEC("arm_4dof_tall.xml")
_Q_START = np.radians(
    [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
)
_TUNED_KNOTS12 = np.radians(
    np.array(
        [
            -1.2,
            -0.6,
            0.0,
            -37.46583288056429,
            -33.196898815597365,
            8.519418496181338,
            -111.8152288749504,
            -76.08164184800785,
            -17.940545379302467,
            -2.162963251122143,
            17.39559530401733,
            20.99761664028883,
        ],
        dtype=float,
    )
)
_TUNED_KP = 318.1900570627695
_TUNED_KD = 0.6954130109241007
_TUNED_RELEASE_TIME_S = 0.0761809061781466


def _ci99_SPEC(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(values, [0.5, 99.5])
    return float(lo), float(hi)


def _collect_release_and_score_samples_SPEC(
    sigma_add: float,
    sigma_mult: float,
    n_rollouts: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    releases = []
    scores = []
    for _ in range(n_rollouts):
        out = simulate_throw_mujoco_SPEC(
            _TUNED_KNOTS12,
            q_start4=_Q_START.copy(),
            xml_path=_XML_PATH,
            rng=rng,
            torque_noise=True,
            torque_noise_sigma_add=sigma_add,
            torque_noise_sigma_mult=sigma_mult,
            release_time_s=_TUNED_RELEASE_TIME_S,
            kp=_TUNED_KP,
            kd=_TUNED_KD,
            enforce_joint_limits=True,
        )
        s6 = out["release_state6"]
        if s6 is None:
            continue
        releases.append(s6)
        scores.append(score_from_release_state_SPEC(s6)["score"])
    releases = np.asarray(releases, dtype=float)
    scores = np.asarray(scores, dtype=float)
    trace = float(np.trace(np.cov(releases.T))) if releases.shape[0] > 1 else 0.0
    return {
        "releases": releases,
        "scores": scores,
        "trace_sigma_release": trace,
    }


def run_uncertainty_calibration_99ci_SPEC() -> dict:
    # Target baseline from tuned nominal noise setting.
    baseline = _collect_release_and_score_samples_SPEC(0.5, 0.02, n_rollouts=120, seed=101)
    target_trace = baseline["trace_sigma_release"]
    target_mean_score = float(np.mean(baseline["scores"])) if baseline["scores"].size else 0.0

    candidates = [
        (0.30, 0.010),
        (0.40, 0.015),
        (0.50, 0.020),
        (0.60, 0.025),
        (0.70, 0.030),
    ]
    rows = []
    for sigma_add, sigma_mult in candidates:
        traces = []
        means = []
        fit_errors = []
        for s in (11, 23, 37, 51, 67):
            sample = _collect_release_and_score_samples_SPEC(sigma_add, sigma_mult, n_rollouts=80, seed=s)
            mscore = float(np.mean(sample["scores"])) if sample["scores"].size else 0.0
            tr = float(sample["trace_sigma_release"])
            traces.append(tr)
            means.append(mscore)
            # Joint fit objective: normalized covariance mismatch + score mismatch.
            err = abs(tr - target_trace) / max(target_trace, 1e-9) + abs(mscore - target_mean_score) / max(abs(target_mean_score), 1.0)
            fit_errors.append(err)
        row = {
            "sigma_add": float(sigma_add),
            "sigma_mult": float(sigma_mult),
            "trace_mean": float(np.mean(traces)),
            "trace_ci99": _ci99_SPEC(np.asarray(traces)),
            "score_mean": float(np.mean(means)),
            "score_ci99": _ci99_SPEC(np.asarray(means)),
            "fit_error_mean": float(np.mean(fit_errors)),
            "fit_error_ci99": _ci99_SPEC(np.asarray(fit_errors)),
        }
        rows.append(row)

    best = min(rows, key=lambda r: r["fit_error_mean"])
    trace_err_best = abs(best["trace_mean"] - target_trace) / max(target_trace, 1e-9)

    # Spin-lift uncertainty (paired effect range at fixed axial spin).
    if baseline["releases"].shape[0] > 0:
        release_cloud = baseline["releases"][: min(120, baseline["releases"].shape[0])]
    else:
        release_cloud = np.zeros((1, 6), dtype=float)
    c_lift_candidates = [0.002, 0.004, 0.006, 0.008, 0.010]
    spin_rows = []
    for c_lift in c_lift_candidates:
        metrics = paired_spin_effect_metrics_SPEC(
            release_states6=release_cloud,
            omega_axial_rad_s=40.0,
            c_lift=c_lift,
            off_axis_noise_std=0.0,
            seed=77,
        )
        spin_rows.append({
            "c_lift": float(c_lift),
            "delta_score_vs_no_spin": float(metrics["delta_score_vs_no_spin"]),
            "delta_radial_miss_mm": float(metrics["delta_radial_miss_mm"]),
            "landing_shift_rms_mm": float(metrics["landing_shift_rms_mm"]),
        })

    status = {
        "phase": "uncertainty_calibration_99ci",
        "target_trace_sigma_release": float(target_trace),
        "target_mean_score": float(target_mean_score),
        "candidates": rows,
        "best_candidate": best,
        "spin_lift_rows": spin_rows,
        "trace_relative_error_best": float(trace_err_best),
        "phase_pass": bool(trace_err_best <= 0.10),
    }

    out_json = _ARTIFACTS / "integration_uncertainty_calibration_99ci_SPEC.json"
    out_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# Integration uncertainty calibration (99% CI)",
        "",
        f"- target trace(Sigma_release): `{target_trace:.6f}`",
        f"- target mean score: `{target_mean_score:.3f}`",
        f"- best candidate: sigma_add=`{best['sigma_add']:.3f}`, sigma_mult=`{best['sigma_mult']:.3f}`",
        f"- relative trace error (best): `{trace_err_best:.3f}`",
        f"- phase pass (trace error <= 10%): `{status['phase_pass']}`",
        "",
        "## Torque/release calibration candidates",
        "",
        "| sigma_add | sigma_mult | trace_mean | trace_ci99_lo | trace_ci99_hi | score_mean | score_ci99_lo | score_ci99_hi | fit_error_mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['sigma_add']:.3f} | {r['sigma_mult']:.3f} | {r['trace_mean']:.6f} | "
            f"{r['trace_ci99'][0]:.6f} | {r['trace_ci99'][1]:.6f} | {r['score_mean']:.3f} | "
            f"{r['score_ci99'][0]:.3f} | {r['score_ci99'][1]:.3f} | {r['fit_error_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Spin-lift sensitivity range",
        "",
        "| c_lift | delta_score_vs_no_spin | delta_radial_miss_mm | landing_shift_rms_mm |",
        "|---:|---:|---:|---:|",
    ])
    for r in spin_rows:
        lines.append(
            f"| {r['c_lift']:.3f} | {r['delta_score_vs_no_spin']:.3f} | "
            f"{r['delta_radial_miss_mm']:.3f} | {r['landing_shift_rms_mm']:.3f} |"
        )
    out_md = _ARTIFACTS / "integration_uncertainty_calibration_99ci_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("--- C6 uncertainty calibration (99% CI) ---")
    print(f"Phase pass: {status['phase_pass']} | best sigma_add={best['sigma_add']:.3f}, sigma_mult={best['sigma_mult']:.3f}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return status


if __name__ == "__main__":
    run_uncertainty_calibration_99ci_SPEC()

