"""
VALIDATION_integration_control_robustness_sweep_SPEC.py
==============================================
Short robustness sweep across control formulations and policy parameterizations.

Run: `python integration_phase_SPEC/VALIDATION_integration_control_robustness_sweep_SPEC.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dartrobot.paths import artifacts_dir_SPEC, project_root_SPEC

_ROOT = project_root_SPEC()

from dartrobot.integration.jacobian_covariance import (
    summarize_release_robustness_SPEC,
)
from dartrobot.integration.rl_env_scaffold import (
    DartThrowEnvConfig_SPEC,
    evaluate_action_mc_SPEC,
)
from dartrobot.spin.ablation import (
    paired_spin_effect_metrics_SPEC,
)

_ARTIFACTS = artifacts_dir_SPEC("integration")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)

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
_TUNED_RELEASE_TIME_S = 0.0761809061781466


def _evaluate_case(name: str, config: DartThrowEnvConfig_SPEC, action: np.ndarray, n_rollouts: int = 60, seed: int = 7) -> dict:
    out = evaluate_action_mc_SPEC(action, n_rollouts=n_rollouts, seed=seed, config=config)
    release_states = np.asarray(out["release_states6"], dtype=float)
    landings = np.asarray(out["landings_m"], dtype=float)
    scores = np.asarray(out["scores"], dtype=float)
    if release_states.shape[0] > 1:
        nominal = np.nanmean(release_states, axis=0)
    else:
        nominal = np.zeros(6, dtype=float)
    robust = summarize_release_robustness_SPEC(
        nominal_release6=nominal,
        release_states6=release_states if release_states.shape[0] > 0 else np.zeros((1, 6)),
        landings_m=landings if landings.shape[0] > 0 else np.zeros((1, 2)),
        scores=scores if scores.shape[0] > 0 else np.zeros(1),
    )
    spin_metrics = paired_spin_effect_metrics_SPEC(
        release_states6=release_states if release_states.shape[0] > 0 else np.zeros((1, 6)),
        omega_axial_rad_s=40.0,
        c_lift=0.006,
        off_axis_noise_std=0.0,
        seed=seed + 99,
    )
    return {
        "name": name,
        "mean_score": float(out["mean_score"]),
        "std_score": float(np.std(scores)) if scores.size else 0.0,
        "hit_rate": float(out["hit_rate"]),
        "trace_sigma_release": float(np.trace(robust["Sigma_release_6x6"])),
        "trace_sigma_land_emp": float(np.trace(robust["Sigma_land_empirical_2x2"])),
        "trace_sigma_land_pred": float(np.trace(robust["Sigma_land_predicted_2x2"])),
        "top_sensitivity": robust["sensitivity_ranking"][0]["label"] if robust["sensitivity_ranking"] else "n/a",
        "spin_delta_score_ax40": float(spin_metrics["delta_score_vs_no_spin"]),
        "spin_delta_hit_rate_ax40": float(spin_metrics["delta_hit_rate"]),
        "spin_shift_rms_mm_ax40": float(spin_metrics["landing_shift_rms_mm"]),
    }


def run_control_robustness_sweep_SPEC(n_rollouts: int = 60, seed: int = 7) -> list[dict]:
    """
    Compare direct knots vs residual formulations and PD/feedforward variants.
    """
    cases = []

    cases.append((
        "direct_knots_pd",
        DartThrowEnvConfig_SPEC(
            torque_noise=True,
            kp=318.19,
            kd=0.695,
            use_residual_action=False,
            action_includes_release_time=True,
            use_minimum_jerk_warm_start=False,
        ),
        np.concatenate([_TUNED_KNOTS12.copy(), np.array([_TUNED_RELEASE_TIME_S], dtype=float)]),
    ))
    cases.append((
        "residual_minjerk_pd_zero_residual",
        DartThrowEnvConfig_SPEC(
            torque_noise=True,
            kp=318.19,
            kd=0.695,
            use_residual_action=True,
            residual_action_scale=1.0,
            action_includes_release_time=True,
            use_minimum_jerk_warm_start=True,
            reward_release_weighted_l2_coeff=0.05,
            target_release_state6=(0.2, 0.0, 1.9, 5.5, 0.0, 0.0),
        ),
        np.concatenate([np.zeros(12, dtype=float), np.array([_TUNED_RELEASE_TIME_S - 0.85 * 0.2], dtype=float)]),
    ))
    cases.append((
        "residual_minjerk_plus_release_time",
        DartThrowEnvConfig_SPEC(
            torque_noise=True,
            kp=318.19,
            kd=0.695,
            use_residual_action=True,
            residual_action_scale=0.25,
            action_includes_release_time=True,
            use_minimum_jerk_warm_start=True,
            reward_release_weighted_l2_coeff=0.05,
            target_release_state6=(0.2, 0.0, 1.9, 5.5, 0.0, 0.0),
        ),
        np.concatenate([np.zeros(12, dtype=float), np.array([0.05], dtype=float)]),
    ))
    cases.append((
        "direct_knots_feedforward_pd",
        DartThrowEnvConfig_SPEC(
            torque_noise=True,
            kp=318.19,
            kd=0.695,
            use_residual_action=False,
            action_includes_release_time=True,
            use_feedforward_controller=True,
            inertia_ff_diag=(1.2, 1.0, 0.8, 0.3),
        ),
        np.concatenate([_TUNED_KNOTS12.copy(), np.array([_TUNED_RELEASE_TIME_S], dtype=float)]),
    ))

    rows = []
    for name, cfg, action in cases:
        row = _evaluate_case(name, cfg, action, n_rollouts=n_rollouts, seed=seed)
        rows.append(row)

    print("--- C3 control robustness sweep ---")
    for row in rows:
        print(
            f"{row['name']}: mean_score={row['mean_score']:.2f}, std={row['std_score']:.2f}, "
            f"hit_rate={row['hit_rate']:.2%}, trace(Sigma_release)={row['trace_sigma_release']:.4f}, "
            f"top_sensitivity={row['top_sensitivity']}, spin_dscore@40={row['spin_delta_score_ax40']:.2f}"
        )

    lines = [
        "# C3 control robustness sweep",
        "",
        "| case | mean_score | std_score | hit_rate | tr(Sigma_release) | tr(Sigma_land_emp) | tr(Sigma_land_pred) | top_sensitivity | spin_delta_score_ax40 | spin_delta_hit_rate_ax40 | spin_shift_rms_mm_ax40 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['mean_score']:.3f} | {row['std_score']:.3f} | {row['hit_rate']:.3f} | "
            f"{row['trace_sigma_release']:.6f} | {row['trace_sigma_land_emp']:.6f} | "
            f"{row['trace_sigma_land_pred']:.6f} | {row['top_sensitivity']} | "
            f"{row['spin_delta_score_ax40']:.3f} | {row['spin_delta_hit_rate_ax40']:.3f} | "
            f"{row['spin_shift_rms_mm_ax40']:.2f} |"
        )
    out_md = _ARTIFACTS / "integration_control_robustness_sweep_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_md}")
    return rows


if __name__ == "__main__":
    run_control_robustness_sweep_SPEC()

