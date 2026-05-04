#!/usr/bin/env python3
"""
run_debug_throw_pipeline_SPEC.py
================================
Deterministic **stage-by-stage** diagnostics for the throw pipeline (iLQR motion → knot optimizer).

For each target preset, measures four stages:

- **S1** ``solve_ilqr_*`` with ``refine_with_mujoco=False``, ``landing_retarget_iters=0``
- **S2** + MuJoCo ``vx`` refine, ``landing_retarget_iters=0``
- **S3** defaults (full retarget)
- **S4** ``optimize_knots_for_board_target_SPEC`` from S3 knots

Writes ``artifacts/debug/throw_pipeline_diagnostics.{json,csv}`` and prints a table + summary hints.

Run from repo root::

    dartrobot debug --preset-set trebles_singles_bulls --maxiter 80 --n-restarts 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from dartrobot.paths import artifacts_dir_SPEC, project_root_SPEC

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    if isinstance(obj, (np.floating, float)) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def iter_target_specs_SPEC(preset_set: str, custom_csv: str | None) -> Iterator[tuple[str, Any, int | None]]:
    """
    Yield ``(label, ring, sector)`` where ``ring`` is a ``RingName_SPEC`` and ``sector`` is int or None.
    """
    from dartrobot.integration.target_score_aim import (
        board_deltas_for_ring_sector_SPEC,
        resolve_preset_SPEC,
    )

    ps = str(preset_set).strip().lower()
    if ps == "custom":
        if not custom_csv or not str(custom_csv).strip():
            raise ValueError("--presets required when --preset-set=custom")
        for raw in str(custom_csv).split(","):
            p = raw.strip()
            if not p:
                continue
            ring, sector = resolve_preset_SPEC(p)
            yield p, ring, sector
        return

    if ps in ("trebles", "trebles_singles_bulls"):
        for s in range(1, 21):
            yield f"t{s}", "treble", s

    if ps in ("singles", "trebles_singles_bulls"):
        for s in range(1, 21):
            yield f"single_{s}", "single", s

    if ps in ("bulls", "trebles_singles_bulls"):
        yield "BULL", "bull_inner", None
        yield "SBULL", "bull_outer", None

    if ps == "s20":
        yield "S20", "single", 20

    if ps not in (
        "trebles",
        "singles",
        "bulls",
        "s20",
        "trebles_singles_bulls",
        "custom",
    ):
        raise ValueError(f"unknown --preset-set {preset_set!r}")


def _resolve_deltas_SPEC(_label: str, ring: Any, sector: int | None) -> tuple[float, float]:
    from dartrobot.integration.target_score_aim import (
        board_deltas_for_ring_sector_SPEC,
    )

    return board_deltas_for_ring_sector_SPEC(ring, sector)


def _collect_row_SPEC(
    *,
    preset_label: str,
    stage_id: str,
    stage_idx: int,
    dy_m: float,
    dz_m: float,
    q_start4: np.ndarray,
    ilqr_out: dict[str, Any],
    diag: dict[str, Any],
    sim: dict[str, Any],
    opt_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    s6 = sim.get("release_state6")
    if s6 is None:
        vx = vy = vz = vnorm = pitch_deg = float("nan")
        px = py = pz = float("nan")
    else:
        s6a = np.asarray(s6, dtype=float).reshape(6)
        px, py, pz = float(s6a[0]), float(s6a[1]), float(s6a[2])
        vx, vy, vz = float(s6a[3]), float(s6a[4]), float(s6a[5])
        vnorm = float(np.linalg.norm(s6a[3:6]))
        hor = math.hypot(vx, vy)
        pitch_deg = float(math.degrees(math.atan2(vz, hor))) if hor > 1e-9 else float("nan")

    land_dy = float(diag.get("delta_y_m", float("nan")))
    land_dz = float(diag.get("delta_z_m", float("nan")))
    if np.isfinite(land_dy) and np.isfinite(land_dz):
        ey_mm = (land_dy - dy_m) * 1000.0
        ez_mm = (land_dz - dz_m) * 1000.0
    else:
        ey_mm = float("nan")
        ez_mm = float("nan")

    row: dict[str, Any] = {
        "preset": preset_label,
        "stage": stage_id,
        "stage_idx": int(stage_idx),
        "target_dy_m": float(dy_m),
        "target_dz_m": float(dz_m),
        "plane_hit": bool(diag.get("plane_hit")),
        "dartboard_face_hit": bool(diag.get("dartboard_face_hit")),
        "score": int(diag.get("score", 0)),
        "r_mm": float(diag.get("r_mm", float("nan"))),
        "ey_mm": ey_mm,
        "ez_mm": ez_mm,
        "l2_err_m2": float(diag.get("l2_err_m2", float("nan"))),
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "vnorm_mps": vnorm,
        "pitch_deg": pitch_deg,
        "release_px_m": px,
        "release_py_m": py,
        "release_pz_m": pz,
        "release_time_s": float(sim.get("release_time_s", float("nan"))),
        "blend_ik_with_overhand_used": float(ilqr_out.get("blend_ik_with_overhand_used", float("nan"))),
        "workspace_goal_xyz_m": np.asarray(ilqr_out.get("workspace_goal_xyz_m", []), dtype=float).tolist()
        if "workspace_goal_xyz_m" in ilqr_out
        else [],
        "workspace_goal_xyz_after_retarget_m": np.asarray(
            ilqr_out.get("workspace_goal_xyz_after_retarget_m", []), dtype=float
        ).tolist()
        if "workspace_goal_xyz_after_retarget_m" in ilqr_out
        else [],
        "p_release_fk_xyz_m": np.asarray(ilqr_out.get("p_release_fk_xyz_m", []), dtype=float).tolist()
        if "p_release_fk_xyz_m" in ilqr_out
        else [],
        "knots12": np.asarray(ilqr_out.get("knots12"), dtype=float).reshape(12).tolist(),
    }
    if opt_meta:
        row["opt_fun"] = float(opt_meta.get("fun", float("nan")))
        row["opt_nfev"] = int(opt_meta.get("nfev", 0))
        row["opt_success"] = bool(opt_meta.get("success", False))
    else:
        row["opt_fun"] = None
        row["opt_nfev"] = None
        row["opt_success"] = None
    return row


def _simulate_and_diag_SPEC(
    knots12: np.ndarray,
    q_start4: np.ndarray,
    dy_m: float,
    dz_m: float,
    mj_kw: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from dartrobot.integration.knot_landing_optimizer import (
        evaluate_knots_landing_diagnostics_SPEC,
    )
    from dartrobot.motion.controller import simulate_throw_mujoco_SPEC

    k = np.asarray(knots12, dtype=float).reshape(12)
    diag = evaluate_knots_landing_diagnostics_SPEC(
        k, q_start4, float(dy_m), float(dz_m), mujoco_kwargs=mj_kw
    )
    sim = simulate_throw_mujoco_SPEC(
        k,
        q_start4=np.asarray(q_start4, dtype=float).reshape(4),
        torque_noise=False,
        **mj_kw,
    )
    return diag, sim


def main(argv: list[str] | None = None) -> int:
    _ROOT = project_root_SPEC()
    try:
        from dartrobot.integration.ilqr_motion_targeting import (
            solve_ilqr_motion_knots_for_board_deltas_SPEC,
        )
        from dartrobot.integration.knot_landing_optimizer import (
            optimize_knots_for_board_target_SPEC,
        )
        from dartrobot.motion.controller import (
            DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        )
    except ImportError as e:
        print("Import failed (is MuJoCo installed?):", e, file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Stage-by-stage throw pipeline diagnostics.")
    p.add_argument(
        "--preset-set",
        type=str,
        default="trebles_singles_bulls",
        choices=("trebles", "singles", "bulls", "s20", "trebles_singles_bulls", "custom"),
    )
    p.add_argument("--presets", type=str, default=None, help="comma-separated when --preset-set=custom")
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--opt-restart-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=artifacts_dir_SPEC("debug"))
    args = p.parse_args(argv)

    q_start4 = np.radians(
        [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
    )
    mj_kw: dict[str, Any] = {
        "kp": DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        "kd": DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        "release_time_s": DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    }

    rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []

    try:
        target_iter = list(iter_target_specs_SPEC(str(args.preset_set), args.presets))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Deduplicate by (ring, sector): same aim as e.g. single_20 vs S20
    seen_key: set[tuple[Any, int | None]] = set()
    deduped: list[tuple[str, Any, int | None]] = []
    for item in target_iter:
        _lab, ring, sec = item
        key = (ring, sec)
        if key in seen_key:
            continue
        seen_key.add(key)
        deduped.append(item)

    STAGES = [
        ("S1_minjerk_no_refine", 1, dict(refine_with_mujoco=False, landing_retarget_iters=0)),
        ("S2_plus_vx_refine", 2, dict(refine_with_mujoco=True, landing_retarget_iters=0)),
        ("S3_plus_retarget", 3, dict(refine_with_mujoco=True, landing_retarget_iters=6)),
    ]

    for preset_label, ring, sector in deduped:
        dy_m, dz_m = _resolve_deltas_SPEC(preset_label, ring, sector)

        knots_s3: np.ndarray | None = None
        ilqr_s3: dict[str, Any] = {}

        for stage_id, stage_idx, ilqr_kw in STAGES:
            ilqr_out = solve_ilqr_motion_knots_for_board_deltas_SPEC(
                q_start4,
                float(dy_m),
                float(dz_m),
                **ilqr_kw,
            )
            knots12 = np.asarray(ilqr_out["knots12"], dtype=float).reshape(12)
            if stage_idx == 3:
                knots_s3 = knots12.copy()
                ilqr_s3 = dict(ilqr_out)

            diag, sim = _simulate_and_diag_SPEC(knots12, q_start4, dy_m, dz_m, mj_kw)
            row = _collect_row_SPEC(
                preset_label=preset_label,
                stage_id=stage_id,
                stage_idx=stage_idx,
                dy_m=dy_m,
                dz_m=dz_m,
                q_start4=q_start4,
                ilqr_out={**ilqr_out, "knots12": knots12},
                diag=diag,
                sim=sim,
                opt_meta=None,
            )
            rows.append(row)
            json_rows.append(dict(row))

        if knots_s3 is None:
            continue

        opt = optimize_knots_for_board_target_SPEC(
            knots_s3,
            float(dy_m),
            float(dz_m),
            q_start4,
            maxiter=int(args.maxiter),
            n_restarts=int(args.n_restarts),
            restart_seed=int(args.opt_restart_seed),
            mujoco_kwargs=mj_kw,
        )
        knots_opt = np.asarray(opt["knots12"], dtype=float).reshape(12)
        opt_meta = {
            "fun": float(opt.get("fun", float("nan"))),
            "nfev": int(opt.get("nfev", 0)),
            "success": bool(opt.get("success", False)),
        }
        diag4, sim4 = _simulate_and_diag_SPEC(knots_opt, q_start4, dy_m, dz_m, mj_kw)
        row4 = _collect_row_SPEC(
            preset_label=preset_label,
            stage_id="S4_knot_opt",
            stage_idx=4,
            dy_m=dy_m,
            dz_m=dz_m,
            q_start4=q_start4,
            ilqr_out={**ilqr_s3, "knots12": knots_opt},
            diag=diag4,
            sim=sim4,
            opt_meta=opt_meta,
        )
        rows.append(row4)
        json_rows.append({k: v for k, v in row4.items()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "throw_pipeline_diagnostics.json"
    csv_path = out_dir / "throw_pipeline_diagnostics.csv"

    json_path.write_text(json.dumps(_json_safe({"preset_set": args.preset_set, "rows": json_rows}), indent=2), encoding="utf-8")

    csv_fields = [
        "preset",
        "stage",
        "stage_idx",
        "r_mm",
        "ey_mm",
        "ez_mm",
        "plane_hit",
        "dartboard_face_hit",
        "score",
        "vx",
        "vy",
        "vz",
        "vnorm_mps",
        "pitch_deg",
        "release_time_s",
        "blend_ik_with_overhand_used",
        "opt_fun",
        "opt_nfev",
        "opt_success",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_fields})

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print()
    hdr = f"{'preset':<14} {'stage':<22} {'r_mm':>10} {'ey_mm':>9} {'ez_mm':>9} {'vx':>7} {'vz':>7} {'|v|':>7} face"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rmm = r["r_mm"]
        rmm_s = f"{rmm:10.1f}" if np.isfinite(rmm) else f"{'nan':>10}"
        ey = r["ey_mm"]
        ez = r["ez_mm"]
        ey_s = f"{ey:9.1f}" if np.isfinite(ey) else f"{'nan':>9}"
        ez_s = f"{ez:9.1f}" if np.isfinite(ez) else f"{'nan':>9}"
        vn = r["vnorm_mps"]
        vn_s = f"{vn:7.2f}" if np.isfinite(vn) else f"{'nan':>7}"
        vx_s = f"{r['vx']:7.2f}" if np.isfinite(r["vx"]) else f"{'nan':>7}"
        vz_s = f"{r['vz']:7.2f}" if np.isfinite(r["vz"]) else f"{'nan':>7}"
        fh = "Y" if r["dartboard_face_hit"] else "N"
        print(
            f"{str(r['preset']):<14} {str(r['stage']):<22} {rmm_s} {ey_s} {ez_s} {vx_s} {vz_s} {vn_s} {fh:>4}"
        )

    # --- Summary + pattern hints (diag_summary) ---
    print()
    print("=== Summary (nanmedian r_mm per stage, face_hit count) ===")
    by_stage: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    face_by_stage: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    ez_all: list[float] = []
    ez_by_stage: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    fun_s4: list[float] = []
    r_s4: list[float] = []

    for r in rows:
        si = int(r["stage_idx"])
        rmm = float(r["r_mm"])
        if np.isfinite(rmm):
            by_stage[si].append(rmm)
        if r.get("dartboard_face_hit"):
            face_by_stage[si] += 1
        ez = float(r["ez_mm"])
        if np.isfinite(ez):
            ez_all.append(ez)
            ez_by_stage[si].append(ez)
        if si == 4:
            fn = r.get("opt_fun")
            if fn is not None and np.isfinite(float(fn)):
                fun_s4.append(float(fn))
            if np.isfinite(rmm):
                r_s4.append(rmm)

    for si in (1, 2, 3, 4):
        arr = np.asarray(by_stage[si], dtype=float)
        med = float(np.nanmedian(arr)) if arr.size else float("nan")
        print(f"  Stage{si}: nanmedian r_mm = {med:.1f}  face_hits = {face_by_stage[si]}/{len(deduped)}")

    m1 = float(np.nanmedian(np.asarray(by_stage[1], dtype=float))) if by_stage[1] else float("nan")
    m2 = float(np.nanmedian(np.asarray(by_stage[2], dtype=float))) if by_stage[2] else float("nan")
    m4 = float(np.nanmedian(np.asarray(by_stage[4], dtype=float))) if by_stage[4] else float("nan")
    mean_fun4 = float(np.mean(fun_s4)) if fun_s4 else float("nan")

    print()
    print("=== Pattern hints ===")
    med_ez_global = float(np.median(np.asarray(ez_all, dtype=float))) if ez_all else float("nan")
    if np.isfinite(med_ez_global) and med_ez_global < -200.0:
        print(
            f"- Median ez_mm across rows is {med_ez_global:.0f} mm (negative ⇒ landings often BELOW aim in Δz). "
            "Often consistent with too-low release, shallow pitch, or late release vs spline."
        )
    else:
        print(f"- Median ez_mm across all rows: {med_ez_global:.1f} mm (check sign vs aim bias).")

    if np.isfinite(m1) and np.isfinite(m2) and m2 > m1 * 1.05:
        print(
            f"- Stage2 median r_mm ({m2:.0f}) > Stage1 ({m1:.0f}): MuJoCo vx-refine may be pulling blend toward overhand nominal and hurting aim."
        )
    else:
        print("- Stage2 vs Stage1: no strong evidence vx-refine worsens median r_mm (or insufficient data).")

    if fun_s4 and np.isfinite(mean_fun4) and np.isfinite(m4) and mean_fun4 < 1000.0 and m4 > 200.0:
        print(
            f"- Stage4 mean opt_fun={mean_fun4:.1f} but nanmedian r_mm={m4:.0f}: loss may be dominated by off-face penalty shape / local minima, not raw L2 to target."
        )
    else:
        print("- Stage4 fun vs r_mm: no single-line 'small fun + huge r' signature from medians alone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
