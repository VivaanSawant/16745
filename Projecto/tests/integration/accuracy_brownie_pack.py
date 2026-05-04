"""
VALIDATION_integration_accuracy_brownie_pack_SPEC.py
===========================================
Accuracy and credibility extras:
1) projectile-parameter sensitivity,
2) torque-noise calibration against release covariance,
3) risk-objective ablation in release-space optimization.

Run: `python integration_phase_SPEC/VALIDATION_integration_accuracy_brownie_pack_SPEC.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from dartrobot.paths import artifacts_dir_SPEC, mjcf_path_SPEC, project_root_SPEC

_ROOT = project_root_SPEC()

from dartrobot.constants import (
    AIR_DENSITY_KG_M3,
    BULLSEYE_CENTER_Z_M,
    CROSS_SECTION_AREA_M2,
    DART_MASS_KG,
    DRAG_COEFFICIENT_CD,
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
    OCHE_TO_BOARD_X_M,
)
from dartrobot.motion.controller import (
    simulate_throw_mujoco_SPEC,
)
from dartrobot.flight.forces import (
    acceleration_total_SPEC,
)
from dartrobot.flight.scoring import (
    score_from_deltas_SPEC,
)
from dartrobot.integration.jacobian_covariance import (
    optimize_release_state_robust_score_SPEC,
)

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


def _integrate_with_custom_projectile_params_SPEC(
    release_state6: np.ndarray,
    *,
    rho: float,
    cd: float,
    mass_kg: float,
    area_m2: float,
    wind_xyz=(0.0, 0.0, 0.0),
    max_time_s: float = 3.0,
) -> dict:
    """Local B2-style integration with explicit aerodynamic parameters."""
    s0 = np.asarray(release_state6, dtype=float).reshape(6)

    def rhs(_t, y):
        ax, ay, az = acceleration_total_SPEC(
            y[3],
            y[4],
            y[5],
            wind_xyz_mps=wind_xyz,
            rho=rho,
            cd=cd,
            mass_kg=mass_kg,
            area_m2=area_m2,
            drag_enabled=True,
        )
        return np.array([y[3], y[4], y[5], ax, ay, az], dtype=float)

    def event_board(_t, y):
        return y[0] - OCHE_TO_BOARD_X_M

    event_board.terminal = True
    event_board.direction = 1.0

    sol = solve_ivp(rhs, (0.0, max_time_s), s0, events=event_board, dense_output=True, rtol=1e-8, atol=1e-10)
    if not sol.t_events or len(sol.t_events[0]) == 0:
        return {"hit": False, "delta_y_m": np.nan, "delta_z_m": np.nan, "score": 0.0}
    t_hit = float(sol.t_events[0][0])
    y_hit = sol.sol(t_hit)
    if float(y_hit[3]) <= 0.0:
        return {"hit": False, "delta_y_m": np.nan, "delta_z_m": np.nan, "score": 0.0}
    dy = float(y_hit[1])
    dz = float(y_hit[2] - BULLSEYE_CENTER_Z_M)
    return {"hit": True, "delta_y_m": dy, "delta_z_m": dz, "score": float(score_from_deltas_SPEC(dy, dz))}


def projectile_parameter_sensitivity_SPEC(release_state6: np.ndarray) -> list[dict]:
    """One-at-a-time sensitivity sweep for cd, rho, and mass."""
    rows = []
    sweep = [
        ("cd", [0.85, 1.00, 1.15]),
        ("rho", [0.90, 1.00, 1.10]),
        ("mass", [0.90, 1.00, 1.10]),
    ]
    for name, multipliers in sweep:
        for mult in multipliers:
            rho = AIR_DENSITY_KG_M3
            cd = DRAG_COEFFICIENT_CD
            mass = DART_MASS_KG
            if name == "cd":
                cd *= mult
            elif name == "rho":
                rho *= mult
            else:
                mass *= mult
            out = _integrate_with_custom_projectile_params_SPEC(
                release_state6,
                rho=rho,
                cd=cd,
                mass_kg=mass,
                area_m2=CROSS_SECTION_AREA_M2,
            )
            rows.append({
                "parameter": name,
                "multiplier": float(mult),
                "delta_y_mm": float(out["delta_y_m"] * 1e3) if np.isfinite(out["delta_y_m"]) else np.nan,
                "delta_z_mm": float(out["delta_z_m"] * 1e3) if np.isfinite(out["delta_z_m"]) else np.nan,
                "score": float(out["score"]),
                "hit": bool(out["hit"]),
            })
    return rows


def calibrate_torque_noise_to_release_covariance_SPEC(
    target_trace: float,
    n_rollouts: int = 50,
    seed: int = 9,
) -> dict:
    """
    Grid-search (sigma_add, sigma_mult) to match a target release covariance trace.
    """
    rng = np.random.default_rng(seed)
    candidates = [
        (0.30, 0.010),
        (0.40, 0.015),
        (0.50, 0.020),
        (0.60, 0.025),
        (0.70, 0.030),
    ]
    best = None
    for sigma_add, sigma_mult in candidates:
        releases = []
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
            if out["release_state6"] is not None:
                releases.append(out["release_state6"])
        rel = np.asarray(releases, dtype=float)
        if rel.shape[0] < 2:
            continue
        trace = float(np.trace(np.cov(rel.T)))
        err = abs(trace - target_trace)
        row = {
            "sigma_add": float(sigma_add),
            "sigma_mult": float(sigma_mult),
            "trace_sigma_release": trace,
            "abs_error_to_target": err,
        }
        if best is None or row["abs_error_to_target"] < best["abs_error_to_target"]:
            best = row
    return best if best is not None else {
        "sigma_add": np.nan,
        "sigma_mult": np.nan,
        "trace_sigma_release": np.nan,
        "abs_error_to_target": np.nan,
    }


def _estimate_release_cov_trace_SPEC(
    sigma_add: float,
    sigma_mult: float,
    n_rollouts: int = 60,
    seed: int = 5,
) -> float:
    """Empirical tr(Sigma_release) for one torque-noise setting."""
    rng = np.random.default_rng(seed)
    releases = []
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
        if out["release_state6"] is not None:
            releases.append(out["release_state6"])
    rel = np.asarray(releases, dtype=float)
    if rel.shape[0] <= 1:
        return 0.0
    return float(np.trace(np.cov(rel.T)))


def risk_objective_ablation_SPEC(release_state6: np.ndarray) -> list[dict]:
    """
    Compare risk_lambda choices in release-space robust optimization.
    """
    sigma_release = np.diag([5e-5, 5e-5, 5e-5, 2e-2, 2e-2, 2e-2])
    rows = []
    for risk_lambda in (0.0, 0.25, 0.75):
        out = optimize_release_state_robust_score_SPEC(
            nominal_release6=release_state6,
            Sigma_release6=sigma_release,
            n_mc_samples=40,
            seed=3,
            risk_lambda=risk_lambda,
        )
        rows.append({
            "risk_lambda": float(risk_lambda),
            "mean_score_mc": float(out["mean_score_mc"]),
            "std_score_mc": float(out["std_score_mc"]),
            "hit_rate_mc": float(out["hit_rate_mc"]),
        })
    return rows


def run_accuracy_brownie_pack_SPEC() -> dict:
    nom = simulate_throw_mujoco_SPEC(
        _TUNED_KNOTS12,
        q_start4=_Q_START.copy(),
        xml_path=_XML_PATH,
        torque_noise=False,
        rng=np.random.default_rng(1),
        release_time_s=_TUNED_RELEASE_TIME_S,
        kp=_TUNED_KP,
        kd=_TUNED_KD,
        enforce_joint_limits=True,
    )
    release_state6 = np.asarray(nom["release_state6"], dtype=float)

    param_rows = projectile_parameter_sensitivity_SPEC(release_state6)

    target_trace = _estimate_release_cov_trace_SPEC(sigma_add=0.5, sigma_mult=0.02, n_rollouts=60, seed=5)
    baseline = calibrate_torque_noise_to_release_covariance_SPEC(target_trace=target_trace, n_rollouts=60, seed=11)
    risk_rows = risk_objective_ablation_SPEC(release_state6)

    print("--- C4 accuracy brownie pack ---")
    print(f"Nominal release state: {release_state6}")
    print("Best torque-noise calibration candidate:")
    print(
        f"  target tr(Sigma_release)={target_trace:.4f} | "
        f"sigma_add={baseline['sigma_add']:.3f}, sigma_mult={baseline['sigma_mult']:.3f}, "
        f"tr(Sigma_release)={baseline['trace_sigma_release']:.4f}, "
        f"|error|={baseline['abs_error_to_target']:.4f}"
    )

    md_lines = [
        "# C4 accuracy brownie pack",
        "",
        "## Projectile parameter sensitivity",
        "",
        "| parameter | multiplier | hit | score | delta_y_mm | delta_z_mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in param_rows:
        md_lines.append(
            f"| {row['parameter']} | {row['multiplier']:.2f} | {int(row['hit'])} | {row['score']:.1f} | "
            f"{row['delta_y_mm']:.1f} | {row['delta_z_mm']:.1f} |"
        )
    md_lines.extend([
        "",
        "## Torque noise calibration candidate",
        "",
        f"- target tr(Sigma_release): `{target_trace:.4f}`",
        f"- best sigma_add: `{baseline['sigma_add']:.3f}`",
        f"- best sigma_mult: `{baseline['sigma_mult']:.3f}`",
        f"- achieved tr(Sigma_release): `{baseline['trace_sigma_release']:.4f}`",
        "",
        "## Risk-objective ablation",
        "",
        "| risk_lambda | mean_score_mc | std_score_mc | hit_rate_mc |",
        "|---:|---:|---:|---:|",
    ])
    for row in risk_rows:
        md_lines.append(
            f"| {row['risk_lambda']:.2f} | {row['mean_score_mc']:.3f} | {row['std_score_mc']:.3f} | {row['hit_rate_mc']:.3f} |"
        )

    out_md = _ARTIFACTS / "integration_accuracy_brownie_pack_SPEC.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_md}")

    return {
        "nominal_release6": release_state6,
        "projectile_sensitivity": param_rows,
        "noise_calibration": baseline,
        "risk_ablation": risk_rows,
    }


if __name__ == "__main__":
    run_accuracy_brownie_pack_SPEC()

