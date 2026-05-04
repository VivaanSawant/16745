#!/usr/bin/env python3
"""
run_strengthen_baseline_SPEC.py
===============================
Populate **strong classical baseline** cache: joint knots + ``release_time_s`` grid search per
target, atomic pickle writes, CSV/JSON summary and deterministic **r_mm** gate.

Example::

    python3 run_strengthen_baseline_SPEC.py --target-set trebles_singles_bulls --n-restarts 6 --maxiter 80
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    from dartrobot.paths import artifacts_dir_SPEC

    from dartrobot.constants import (
        KEYFRAME_ELBOW_DEG,
        KEYFRAME_SHOULDER_DEG,
        KEYFRAME_SHOULDER_YAW_DEG,
        KEYFRAME_WRIST_DEG,
    )
    from dartrobot.integration.strong_classical_baseline import (
        load_strong_baseline_cache_SPEC,
        save_strong_baseline_entry_SPEC,
        solve_strong_baseline_for_target_SPEC,
        strong_baseline_cache_key_SPEC,
    )
    from dartrobot.rl.env_residual_throw import build_labeled_target_pool_SPEC

    p = argparse.ArgumentParser(description="Build strong classical baseline cache (knots + release time).")
    p.add_argument(
        "--target-set",
        type=str,
        required=True,
        choices=("trebles", "singles", "bulls", "trebles_singles_bulls", "full", "custom", "trebles_bulls", "singles_bulls"),
        help="target pool (same semantics as build_labeled_target_pool_SPEC / build_target_pool_SPEC)",
    )
    p.add_argument(
        "--presets",
        type=str,
        default=None,
        help="comma-separated presets when --target-set=custom (e.g. t10,BULL)",
    )
    p.add_argument(
        "--release-time-grid",
        type=str,
        default="0.05,0.075,0.10,0.125,0.15",
        help="comma-separated release_time_s values",
    )
    p.add_argument("--n-restarts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--restart-seed", type=int, default=0)
    p.add_argument(
        "--cache-path",
        type=Path,
        default=artifacts_dir_SPEC("baseline", "strong_baseline_cache.pkl"),
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=artifacts_dir_SPEC("baseline", "strong_baseline_summary"),
        help="path prefix; writes .csv and .json",
    )
    p.add_argument("--gate-mm", type=float, default=300.0)
    p.add_argument("--force", action="store_true", help="re-solve even if key already in cache")
    args = p.parse_args(argv)

    custom_list: list[str] | None = None
    if args.target_set == "custom":
        if not args.presets:
            print("--presets required when --target-set=custom", file=sys.stderr)
            return 1
        custom_list = [x.strip() for x in str(args.presets).split(",") if x.strip()]

    try:
        rt_grid = tuple(float(x.strip()) for x in str(args.release_time_grid).split(",") if x.strip())
    except ValueError:
        print("Invalid --release-time-grid", file=sys.stderr)
        return 1
    if not rt_grid:
        print("--release-time-grid must contain at least one float", file=sys.stderr)
        return 1

    labeled = build_labeled_target_pool_SPEC(args.target_set, custom_presets=custom_list)
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    q_start4 = np.radians(
        [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG],
        dtype=float,
    ).reshape(4)

    summary_rows: list[dict[str, object]] = []
    gate_mm = float(args.gate_mm)

    for preset, dy_m, dz_m in labeled:
        key = strong_baseline_cache_key_SPEC(dy_m, dz_m)
        if not args.force:
            existing = load_strong_baseline_cache_SPEC(cache_path).get(key)
            if existing is not None:
                print(f"[skip] {preset} key={key} (already in cache; use --force to re-solve)")
                try:
                    r_mm = float(existing["r_mm"])
                except (KeyError, TypeError, ValueError):
                    r_mm = float("nan")
                summary_rows.append(
                    {
                        "preset": preset,
                        "dy_m": dy_m,
                        "dz_m": dz_m,
                        "r_mm": r_mm,
                        "release_time_s": float(existing.get("release_time_s", float("nan"))),
                        "face_hit": bool(existing.get("face_hit", False)),
                        "score": int(existing.get("score", 0)),
                        "fun": float(existing.get("fun", float("nan"))),
                        "cached": True,
                    }
                )
                continue

        print(f"[solve] {preset} (dy,dz)=({dy_m:.6f},{dz_m:.6f}) …")
        soln = solve_strong_baseline_for_target_SPEC(
            q_start4,
            float(dy_m),
            float(dz_m),
            release_time_grid=rt_grid,
            maxiter=int(args.maxiter),
            n_restarts=int(args.n_restarts),
            restart_seed=int(args.restart_seed),
        )
        save_strong_baseline_entry_SPEC(cache_path, float(dy_m), float(dz_m), soln)
        summary_rows.append(
            {
                "preset": preset,
                "dy_m": dy_m,
                "dz_m": dz_m,
                "r_mm": float(soln.r_mm),
                "release_time_s": float(soln.release_time_s),
                "face_hit": bool(soln.face_hit),
                "score": int(soln.score),
                "fun": float(soln.fun),
                "cached": False,
            }
        )
        print(
            f"  → r_mm={soln.r_mm:.2f} release_time_s={soln.release_time_s:.4g} "
            f"face_hit={soln.face_hit} fun={soln.fun:.6g}"
        )

    summary_prefix = Path(args.summary_out)
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    if summary_prefix.suffix.lower() == ".csv":
        csv_path = summary_prefix
        json_path = summary_prefix.with_suffix(".json")
    elif summary_prefix.suffix.lower() == ".json":
        json_path = summary_prefix
        csv_path = summary_prefix.with_suffix(".csv")
    else:
        csv_path = summary_prefix.with_suffix(".csv")
        json_path = summary_prefix.with_suffix(".json")

    fieldnames = ("preset", "dy_m", "dz_m", "r_mm", "release_time_s", "face_hit", "score", "fun", "cached")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    within = sum(1 for r in summary_rows if np.isfinite(float(r["r_mm"])) and float(r["r_mm"]) <= gate_mm)
    ntot = len(summary_rows)
    frac = (within / ntot) if ntot else 0.0
    report = {
        "target_set": args.target_set,
        "n_targets": ntot,
        "gate_mm": gate_mm,
        "n_within_gate": within,
        "fraction_within_gate": frac,
        "release_time_grid": list(rt_grid),
        "cache_path": str(cache_path),
        "rows": summary_rows,
    }
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"gate: {within} / {ntot} targets with deterministic r_mm <= {gate_mm:.0f} mm ({100.0 * frac:.1f}%)")

    if ntot == 0:
        print("No targets processed.", file=sys.stderr)
        return 1
    if within == 0:
        print("gate count is zero: exiting with status 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
