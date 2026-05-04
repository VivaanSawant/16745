"""
VALIDATION_integration_ilqr_target_regions_SPEC.py
=================================================
Region-wise validation for **4-DOF iLQR targeting** (sectors 20, 3, 6, 11):

1. **Stage 1:** inner-single radial centers via `integration_ilqr_motion_targeting_SPEC`.
2. **Monte Carlo** under torque noise per sector.
3. **Stage gate** (`stage_progression_gate_passed_SPEC`) → optional **Stage 2:** treble centers.

Run:
  python integration_phase_SPEC/VALIDATION_integration_ilqr_target_regions_SPEC.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dartrobot.paths import artifacts_dir_SPEC, project_root_SPEC

_ROOT = project_root_SPEC()

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
)
from dartrobot.motion.controller import (
    DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
    DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
)
from dartrobot.integration.target_board_regions import (
    stage_progression_gate_passed_SPEC,
    stage_targets_single_then_treble_SPEC,
)
from dartrobot.integration.ilqr_motion_targeting import (
    knots12_with_ilqr_warmstart_for_targets_SPEC,
)
from dartrobot.integration.release_to_score import (
    evaluate_ilqr_target_pack_mc_SPEC,
)

_ARTIFACTS = artifacts_dir_SPEC("integration")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)


def run_ilqr_target_region_validation_SPEC(
    *,
    n_mc: int = 20,
    seed: int = 11,
    torque_noise: bool = True,
) -> dict:
    q_start4 = np.radians(
        [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
    )
    pack = stage_targets_single_then_treble_SPEC()

    single_ilqr = knots12_with_ilqr_warmstart_for_targets_SPEC(q_start4, pack, stage="single")
    mc_single = evaluate_ilqr_target_pack_mc_SPEC(
        {s: single_ilqr["per_sector"][s] for s in single_ilqr["per_sector"]},
        q_start4=q_start4,
        n_rollouts=n_mc,
        seed=seed,
        torque_noise=torque_noise,
        kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        release_time_s=DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    )

    agg_hits = []
    agg_rad = []
    for row in mc_single["per_sector"].values():
        agg_hits.append(float(row["hit_rate"]))
        if np.isfinite(row["mean_radial_error_mm"]):
            agg_rad.append(float(row["mean_radial_error_mm"]))
    mean_hit = float(np.mean(agg_hits)) if agg_hits else 0.0
    mean_rad = float(np.mean(agg_rad)) if agg_rad else float("nan")
    gate = stage_progression_gate_passed_SPEC(mean_hit, mean_rad)

    treble_ilqr = None
    mc_treble = None
    if gate:
        treble_ilqr = knots12_with_ilqr_warmstart_for_targets_SPEC(q_start4, pack, stage="treble")
        mc_treble = evaluate_ilqr_target_pack_mc_SPEC(
            {s: treble_ilqr["per_sector"][s] for s in treble_ilqr["per_sector"]},
            q_start4=q_start4,
            n_rollouts=n_mc,
            seed=seed + 97,
            torque_noise=torque_noise,
            kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            release_time_s=DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        )

    status = {
        "stage_targets_pack": {
            "sectors": list(pack["sectors"]),
            "inner_single_radius_m": float(pack["inner_single_radius_m"]),
            "treble_radius_m": float(pack["treble_radius_m"]),
        },
        "single_stage": {
            "ilqr": {
                str(k): {
                    kk: (vv.tolist() if isinstance(vv, np.ndarray) else float(vv) if np.isscalar(vv) else vv)
                    for kk, vv in v.items()
                    if kk not in ("u_sequence_nm", "lqr_gain_K")
                }
                for k, v in single_ilqr["per_sector"].items()
            },
            "mc_summary": {str(k): {kk: float(vv) if np.isscalar(vv) else vv for kk, vv in v.items() if kk != "scores"} for k, v in mc_single["per_sector"].items()},
            "aggregate_hit_rate": mean_hit,
            "aggregate_mean_radial_error_mm": mean_rad,
            "stage_gate_passed": bool(gate),
        },
        "treble_stage": None
        if mc_treble is None
        else {
            "ilqr": {
                str(k): {
                    kk: (vv.tolist() if isinstance(vv, np.ndarray) else float(vv) if np.isscalar(vv) else vv)
                    for kk, vv in v.items()
                    if kk not in ("u_sequence_nm", "lqr_gain_K")
                }
                for k, v in treble_ilqr["per_sector"].items()
            },
            "mc_summary": {str(k): {kk: float(vv) if np.isscalar(vv) else vv for kk, vv in v.items() if kk != "scores"} for k, v in mc_treble["per_sector"].items()},
        },
    }

    out_path = _ARTIFACTS / "integration_ilqr_target_regions_SPEC.json"
    out_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return status


if __name__ == "__main__":
    run_ilqr_target_region_validation_SPEC()
