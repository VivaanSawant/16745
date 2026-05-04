#!/usr/bin/env python3
"""
run_target_score_monte_carlo_SPEC.py
====================================
CLI: fix an aim point from **ring + sector** (or a **preset**), build **12-knot** motion via
``solve_ilqr_motion_knots_for_board_deltas_SPEC``, run **N** noisy MuJoCo throws, print mean/std
score, save a **board-view scatter** PNG + JSON manifest.

Optional ``--darts-per-episode D``: group throws into episodes of length ``D`` and report
mean/std of **per-episode total score** (requires ``N % D == 0``).

**Metrics note:** In ``evaluate_knots12_throw_once_SPEC``, ``hit`` means the projectile **crossed the
infinite vertical plane** ``x = 2.37`` m with ``vx > 0`` (B2), not that the dart landed on the
**dartboard face** (``r ≤ 170`` mm). This script reports ``board_plane_crossing_rate`` and
``dartboard_face_hit_rate`` so a cluster far from the board is not misread as “always on target.”

Examples::

    python3 run_target_score_monte_carlo_SPEC.py --preset t20 --n 200 --seed 0 --target-score 60
    python3 run_target_score_monte_carlo_SPEC.py --preset d15 --n 200 --seed 0 --target-score 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Headless / CI: avoid Qt backend when saving PNG only.
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path

import numpy as np

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
    R_BOARD_MISS_MM,
)
from dartrobot.mc.scatter_plot import plot_mc_board_SPEC
from dartrobot.paths import artifacts_dir_SPEC, project_root_SPEC


def main(argv: list[str] | None = None) -> int:
    try:
        from dartrobot.integration.ilqr_motion_targeting import (
            solve_ilqr_motion_knots_for_board_deltas_SPEC,
        )
        from dartrobot.integration.release_to_score import (
            evaluate_knots12_throw_once_SPEC,
        )
        from dartrobot.integration.target_score_aim import (
            assert_target_score_matches_aim_SPEC,
            board_deltas_for_ring_sector_SPEC,
            parse_ring_SPEC,
            resolve_preset_SPEC,
            score_at_aim_SPEC,
        )
        from dartrobot.motion.controller import (
            DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        )
        from dartrobot.integration.knot_landing_optimizer import (
            evaluate_knots_landing_diagnostics_SPEC,
            optimize_knots_for_board_target_SPEC,
        )
    except ImportError as e:
        print(
            "Failed to import project modules (is MuJoCo installed? try: pip install mujoco):",
            e,
            file=sys.stderr,
        )
        return 1

    _ROOT = project_root_SPEC()

    p = argparse.ArgumentParser(description="Monte Carlo throws toward a resolved board aim.")
    p.add_argument("--n", type=int, default=200, help="number of independent throws (rollouts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        help="treble: t1..t20 (e.g. t10); double: d1..d20 (e.g. d15); also S20, BULL, SBULL (case-insensitive for bulls)",
    )
    p.add_argument("--ring", type=str, default=None, help="treble|double|single|bull_inner|bull_outer")
    p.add_argument("--sector", type=int, default=None, help="1..20 (ignored for bulls)")
    p.add_argument(
        "--target-score",
        type=int,
        default=None,
        help="if set, must match B3 score at the resolved aim point",
    )
    p.add_argument("--output-dir", type=Path, default=artifacts_dir_SPEC("integration"))
    p.add_argument("--no-torque-noise", action="store_true")
    p.add_argument(
        "--darts-per-episode",
        type=int,
        default=1,
        help="if >1, group consecutive throws; N must be divisible by D",
    )
    p.add_argument(
        "--optimize-knots",
        action="store_true",
        help="before MC, run L-BFGS-B on 12 knots to reduce landing error vs aim (deterministic; slower)",
    )
    p.add_argument("--opt-maxiter", type=int, default=120, help="max iterations per L-BFGS-B solve")
    p.add_argument(
        "--opt-n-restarts",
        type=int,
        default=8,
        help="multi-start count (each restart runs L-BFGS-B from perturbed knots)",
    )
    p.add_argument("--opt-restart-seed", type=int, default=0, help="RNG seed for knot optimizer restarts")
    p.add_argument(
        "--use-baseline-cache",
        type=Path,
        default=None,
        help="strong_baseline_cache.pkl: skip iLQR + default knot opt; use cached knots12 and release_time_s in rollouts",
    )
    args = p.parse_args(argv)

    n = int(args.n)
    if n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 1
    dpe = int(args.darts_per_episode)
    if dpe < 1:
        print("--darts-per-episode must be >= 1", file=sys.stderr)
        return 1
    if dpe > 1 and n % dpe != 0:
        print(f"With --darts-per-episode {dpe}, --n must be divisible by D (got n={n})", file=sys.stderr)
        return 1

    try:
        if args.preset:
            ring, sector = resolve_preset_SPEC(str(args.preset))
        else:
            if args.ring is None:
                print("Provide --preset or both --ring and (--sector for non-bull rings)", file=sys.stderr)
                return 1
            ring = parse_ring_SPEC(str(args.ring))
            if ring in ("bull_inner", "bull_outer"):
                sector = None
            else:
                if args.sector is None:
                    print(f"--sector required for ring {ring}", file=sys.stderr)
                    return 1
                sector = int(args.sector)

        dy_m, dz_m = board_deltas_for_ring_sector_SPEC(ring, sector)
        aim_score = score_at_aim_SPEC(dy_m, dz_m)
        if args.target_score is not None:
            assert_target_score_matches_aim_SPEC(int(args.target_score), dy_m, dz_m)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    q_start4 = np.radians(
        [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
    )

    mj_kw = {
        "kp": DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        "kd": DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        "release_time_s": DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    }

    knot_opt_meta: dict = {"used": False, "using_baseline_cache": False}
    used_baseline_cache = False

    if args.use_baseline_cache is not None:
        from dartrobot.integration.strong_classical_baseline import (
            lookup_strong_baseline_SPEC,
        )

        bpath = Path(args.use_baseline_cache)
        hit = lookup_strong_baseline_SPEC(bpath, float(dy_m), float(dz_m))
        if hit is None:
            print(
                f"No strong baseline cache entry for (dy,dz)=({dy_m:.7g},{dz_m:.7g}) in {bpath}",
                file=sys.stderr,
            )
            return 1
        knots12, rt_b, meta_b = hit
        knots12 = np.asarray(knots12, dtype=float).reshape(12)
        mj_kw["release_time_s"] = float(rt_b)
        used_baseline_cache = True
        knot_opt_meta = {
            "used": False,
            "using_baseline_cache": True,
            "release_time_s": float(rt_b),
            "cache_meta": meta_b,
        }
        print(f"using_baseline_cache=true release_time_s={float(rt_b)}")
    else:
        ilqr_out = solve_ilqr_motion_knots_for_board_deltas_SPEC(q_start4, float(dy_m), float(dz_m))
        knots12 = np.asarray(ilqr_out["knots12"], dtype=float).reshape(12)

    if bool(args.optimize_knots):
        pre = evaluate_knots_landing_diagnostics_SPEC(
            knots12, q_start4, float(dy_m), float(dz_m), mujoco_kwargs=mj_kw
        )
        opt = optimize_knots_for_board_target_SPEC(
            knots12,
            float(dy_m),
            float(dz_m),
            q_start4,
            maxiter=int(args.opt_maxiter),
            n_restarts=int(args.opt_n_restarts),
            restart_seed=int(args.opt_restart_seed),
            mujoco_kwargs=mj_kw,
        )
        knots12 = np.asarray(opt["knots12"], dtype=float).reshape(12)
        post = evaluate_knots_landing_diagnostics_SPEC(
            knots12, q_start4, float(dy_m), float(dz_m), mujoco_kwargs=mj_kw
        )
        knot_opt_meta = {
            "used": True,
            "using_baseline_cache": bool(used_baseline_cache),
            "opt_maxiter": int(args.opt_maxiter),
            "opt_n_restarts": int(args.opt_n_restarts),
            "opt_restart_seed": int(args.opt_restart_seed),
            "success": bool(opt.get("success")),
            "fun": float(opt.get("fun", 0.0)),
            "nit": int(opt.get("nit", 0)),
            "nfev": int(opt.get("nfev", 0)),
            "pre_opt": pre,
            "post_opt": post,
        }
        print(
            "knot_opt: "
            f"pre_face={pre.get('dartboard_face_hit')} r_mm={pre.get('r_mm')} "
            f"post_face={post.get('dartboard_face_hit')} r_mm={post.get('r_mm')} "
            f"success={opt.get('success')} fun={opt.get('fun'):.6g} nfev={opt.get('nfev')}"
        )

    rng = np.random.default_rng(int(args.seed))
    torque_noise = not bool(args.no_torque_noise)

    scores = []
    dys = []
    dzs = []
    plane_hits = []
    face_hits = []
    for _ in range(n):
        out = evaluate_knots12_throw_once_SPEC(
            knots12,
            q_start4=q_start4,
            rng=rng,
            torque_noise=torque_noise,
            **mj_kw,
        )
        scores.append(float(out["score"]))
        crossed = bool(out.get("hit"))
        plane_hits.append(crossed)
        dyv = out.get("delta_y_m")
        dzv = out.get("delta_z_m")
        dys.append(float(dyv) if crossed else float("nan"))
        dzs.append(float(dzv) if crossed else float("nan"))
        if crossed and np.isfinite(dyv) and np.isfinite(dzv):
            r_mm = math.hypot(float(dyv) * 1000.0, float(dzv) * 1000.0)
            face_hits.append(r_mm <= float(R_BOARD_MISS_MM))
        else:
            face_hits.append(False)

    scores_a = np.asarray(scores, dtype=float)
    mean_score = float(np.mean(scores_a))
    std_score = float(np.std(scores_a))
    board_plane_crossing_rate = float(np.mean(plane_hits))
    dartboard_face_hit_rate = float(np.mean(face_hits))
    score_for_exact = float(args.target_score) if args.target_score is not None else float(aim_score)
    exact = float(np.mean(scores_a == score_for_exact))

    episode_totals = None
    ep_mean = ep_std = None
    if dpe > 1:
        g = scores_a.reshape(-1, dpe)
        episode_totals = np.sum(g, axis=1)
        ep_mean = float(np.mean(episode_totals))
        ep_std = float(np.std(episode_totals))

    print(f"aim: ring={ring} sector={sector} (Δy,Δz)=({dy_m:.5f},{dz_m:.5f}) m  B3_score_at_aim={aim_score}")
    print(f"n={n} seed={args.seed} torque_noise={torque_noise}")
    print(
        f"mean_score={mean_score:.4f}  std_score={std_score:.4f}  "
        f"board_plane_crossing_rate={board_plane_crossing_rate:.4f}  "
        f"dartboard_face_hit_rate={dartboard_face_hit_rate:.4f}  "
        f"frac_matching_target_score={exact:.4f}"
    )
    if episode_totals is not None:
        print(f"darts_per_episode={dpe}  episodes={len(episode_totals)}")
        print(f"episode_total_mean={ep_mean:.4f}  episode_total_std={ep_std:.4f}")

    if args.preset:
        tag = str(args.preset).strip()
    elif sector is None:
        tag = str(ring)
    else:
        tag = f"{ring}_s{sector}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtag = f"_d{dpe}" if dpe > 1 else ""
    png_path = out_dir / f"integration_target_score_mc_{safe}_n{n}{dtag}_SPEC.png"
    json_path = out_dir / f"integration_target_score_mc_{safe}_n{n}{dtag}_SPEC.json"

    manifest = {
        "preset": args.preset,
        "ring": ring,
        "sector": sector,
        "use_baseline_cache": str(Path(args.use_baseline_cache).resolve())
        if args.use_baseline_cache is not None
        else None,
        "release_time_s_rollout": float(mj_kw["release_time_s"]),
        "optimize_knots": bool(args.optimize_knots),
        "opt_maxiter": int(args.opt_maxiter) if bool(args.optimize_knots) else None,
        "opt_n_restarts": int(args.opt_n_restarts) if bool(args.optimize_knots) else None,
        "opt_restart_seed": int(args.opt_restart_seed) if bool(args.optimize_knots) else None,
        "knot_optimizer": knot_opt_meta,
        "aim_delta_y_m": float(dy_m),
        "aim_delta_z_m": float(dz_m),
        "b3_score_at_aim": int(aim_score),
        "target_score_arg": args.target_score,
        "n_rollouts": n,
        "seed": int(args.seed),
        "torque_noise": bool(torque_noise),
        "darts_per_episode": dpe,
        "mean_score": mean_score,
        "std_score": std_score,
        "board_plane_crossing_rate": board_plane_crossing_rate,
        "dartboard_face_hit_rate": dartboard_face_hit_rate,
        "fraction_matching_target_score": exact,
        "episode_total_mean": ep_mean,
        "episode_total_std": ep_std,
        "png_path": str(png_path.relative_to(_ROOT)),
        "json_path": str(json_path.relative_to(_ROOT)),
    }
    json_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_path}")

    try:
        plot_mc_board_SPEC(
            np.asarray(dys, dtype=float),
            np.asarray(dzs, dtype=float),
            scores_a,
            float(dy_m),
            float(dz_m),
            png_path,
            episode_totals=episode_totals,
            dartboard_face_hit_rate=dartboard_face_hit_rate,
        )
    except ImportError as e:
        print("matplotlib required to save plot:", e, file=sys.stderr)
        return 1
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
