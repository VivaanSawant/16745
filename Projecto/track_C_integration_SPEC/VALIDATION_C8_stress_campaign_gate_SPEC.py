"""
VALIDATION_C8_stress_campaign_gate_SPEC.py
==========================================
Phase 4 of pre-RL confidence plan: cross-seed stress campaign and readiness gate.

Run:
  python track_C_integration_SPEC/VALIDATION_C8_stress_campaign_gate_SPEC.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from track_C_integration_SPEC.RL_env_scaffold_SPEC import (
    DartThrowEnvConfig_SPEC,
    evaluate_action_mc_SPEC,
)

_ARTIFACTS = _ROOT / "artifacts_SPEC"
_ARTIFACTS.mkdir(exist_ok=True)
_TUNED_KNOTS9 = np.radians(np.array([
    -37.46583288056429, -33.196898815597365, 8.519418496181338,
    -111.8152288749504, -76.08164184800785, -17.940545379302467,
    -2.162963251122143, 17.39559530401733, 20.99761664028883,
], dtype=float))
_TUNED_RELEASE_TIME_S = 0.0761809061781466


def _controller_cases_SPEC() -> list[tuple[str, DartThrowEnvConfig_SPEC, np.ndarray]]:
    return [
        (
            "direct_pd",
            DartThrowEnvConfig_SPEC(
                torque_noise=True,
                kp=318.19,
                kd=0.695,
                use_residual_action=False,
                action_includes_release_time=True,
                use_feedforward_controller=False,
            ),
            np.concatenate([_TUNED_KNOTS9.copy(), np.array([_TUNED_RELEASE_TIME_S], dtype=float)]),
        ),
        (
            "feedforward_pd",
            DartThrowEnvConfig_SPEC(
                torque_noise=True,
                kp=318.19,
                kd=0.695,
                use_residual_action=False,
                action_includes_release_time=True,
                use_feedforward_controller=True,
                inertia_ff_diag=(1.0, 0.8, 0.3),
            ),
            np.concatenate([_TUNED_KNOTS9.copy(), np.array([_TUNED_RELEASE_TIME_S], dtype=float)]),
        ),
    ]


def run_stress_campaign_gate_SPEC() -> dict:
    seeds = [3, 7, 11, 19, 23]
    winds = [
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (-0.25, 0.0, 0.0),
        (0.0, 0.15, 0.0),
    ]
    release_offsets = [-0.004, 0.0, 0.004]

    rows = []
    for case_name, base_cfg, action in _controller_cases_SPEC():
        for wind in winds:
            for rt_offset in release_offsets:
                scenario_scores = []
                scenario_hits = []
                scenario_radial = []
                for seed in seeds:
                    local_action = action.copy()
                    local_action[9] = float(local_action[9] + rt_offset)
                    cfg = DartThrowEnvConfig_SPEC(**vars(base_cfg))
                    cfg.wind_xyz = tuple(float(v) for v in wind)
                    out = evaluate_action_mc_SPEC(local_action, n_rollouts=40, seed=seed, config=cfg)
                    scenario_scores.append(float(out["mean_score"]))
                    scenario_hits.append(float(out["hit_rate"]))
                    landings = np.asarray(out["landings_m"], dtype=float)
                    if landings.size == 0:
                        scenario_radial.append(np.nan)
                    else:
                        # RL env stores landing deltas in board coordinates [dy, dz], both bull-relative.
                        radial_mm = np.sqrt(landings[:, 0] ** 2 + landings[:, 1] ** 2) * 1e3
                        scenario_radial.append(float(np.nanmean(radial_mm)))

                scenario_scores = np.asarray(scenario_scores, dtype=float)
                scenario_hits = np.asarray(scenario_hits, dtype=float)
                scenario_radial = np.asarray(scenario_radial, dtype=float)
                row = {
                    "case": case_name,
                    "wind_xyz": [float(v) for v in wind],
                    "release_time_offset_s": float(rt_offset),
                    "mean_score": float(np.nanmean(scenario_scores)),
                    "mean_hit_rate": float(np.nanmean(scenario_hits)),
                    "mean_radial_miss_mm": float(np.nanmean(scenario_radial)),
                }
                row["pass"] = bool(
                    # Operational readiness gate (non-statistical): we only require
                    # physically reasonable stress behavior before RL starts.
                    row["mean_hit_rate"] >= 0.95
                    and row["mean_score"] >= 2.0
                    and row["mean_radial_miss_mm"] <= 220.0
                )
                rows.append(row)

    pass_count = int(sum(1 for r in rows if r["pass"]))
    pass_rate = float(pass_count / len(rows)) if rows else 0.0
    phase_pass = bool(pass_count == len(rows))
    status = {
        "phase": "stress_campaign_gate",
        "n_scenarios": len(rows),
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "global_gate_pass": phase_pass,
        "rows": rows,
    }

    out_json = _ARTIFACTS / "C8_stress_campaign_gate_SPEC.json"
    out_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# C8 stress campaign gate",
        "",
        f"- scenarios: `{len(rows)}`",
        f"- pass count: `{pass_count}`",
        f"- pass rate: `{pass_rate:.3f}`",
        f"- global readiness gate pass (all scenarios operationally stable): `{phase_pass}`",
        "",
        "| case | wind_xyz | release_time_offset_s | mean_score | mean_hit_rate | mean_radial_miss_mm | pass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {tuple(r['wind_xyz'])} | {r['release_time_offset_s']:.3f} | "
            f"{r['mean_score']:.3f} | {r['mean_hit_rate']:.3f} | {r['mean_radial_miss_mm']:.1f} | {int(r['pass'])} |"
        )
    out_md = _ARTIFACTS / "C8_stress_campaign_gate_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("--- C8 stress campaign gate ---")
    print(f"Pass rate={pass_rate:.2%} | global readiness gate pass={phase_pass}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return status


if __name__ == "__main__":
    run_stress_campaign_gate_SPEC()

