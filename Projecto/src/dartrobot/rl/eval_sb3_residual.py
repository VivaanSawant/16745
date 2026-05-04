#!/usr/bin/env python3
"""
eval_sb3_residual_SPEC.py
=========================
Load a trained **goal-conditioned PPO** (``*.zip``) and run Monte Carlo throws at a **single**
board target without retraining. Writes PNG + JSON under ``artifacts/rl/``.

Requires: ``stable-baselines3``, ``gymnasium``, MuJoCo, project SPEC stack.

Example::

    dartrobot rl-eval \\
        --policy-zip policies/ppo_dart_goalcond.zip \\
        --preset t10 --n 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        print("Install RL deps: pip install -e \".[rl]\"", e, file=sys.stderr)
        return 1

    from dartrobot.integration.target_score_aim import (
        assert_target_score_matches_aim_SPEC,
        board_deltas_for_ring_sector_SPEC,
        parse_ring_SPEC,
        resolve_preset_SPEC,
        score_at_aim_SPEC,
    )
    from dartrobot.mc.scatter_plot import plot_mc_board_SPEC
    from dartrobot.paths import artifacts_dir_SPEC, project_root_SPEC
    from dartrobot.rl.env_residual_throw import (
        STRONG_BASELINE_CACHE_DEFAULT_PATH_SPEC,
        DartResidualThrowEnv_SPEC,
        ensure_warm_knots_cached_for_pool_SPEC,
    )

    _ROOT = project_root_SPEC()

    p = argparse.ArgumentParser(description="Evaluate saved PPO residual policy at one target.")
    p.add_argument("--policy-zip", type=Path, required=True, help="SB3 PPO checkpoint (.zip)")
    p.add_argument("--preset", type=str, default=None, help="e.g. t10, d15, BULL")
    p.add_argument("--ring", type=str, default=None)
    p.add_argument("--sector", type=int, default=None)
    p.add_argument("--dy", type=float, default=None, help="explicit target Δy (m), with --dz")
    p.add_argument("--dz", type=float, default=None, help="explicit target Δz (m), with --dy")
    p.add_argument("--target-score", type=int, default=None, help="if set, must match B3 score at aim")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-torque-noise", action="store_true")
    p.add_argument(
        "--warm-start-cache",
        type=Path,
        default=STRONG_BASELINE_CACHE_DEFAULT_PATH_SPEC,
        help="primary strong baseline cache; legacy warm_start_cache.pkl used as fallback",
    )
    p.add_argument("--opt-n-restarts", type=int, default=2)
    p.add_argument("--opt-maxiter", type=int, default=60)
    p.add_argument("--opt-restart-seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=artifacts_dir_SPEC("rl"))
    args = p.parse_args(argv)

    zip_path = Path(args.policy_zip)
    if not zip_path.is_file():
        alt = zip_path.with_suffix(".zip")
        if alt.is_file():
            zip_path = alt
        else:
            print(f"Policy file not found: {args.policy_zip}", file=sys.stderr)
            return 1

    ring: str
    sector: int | None
    tag = "target"

    if args.dy is not None and args.dz is not None:
        dy_m, dz_m = float(args.dy), float(args.dz)
        ring, sector = "explicit", None
        tag = f"dy{dy_m:.4f}_dz{dz_m:.4f}".replace("-", "m")
    elif args.preset:
        ring_t, sector_t = resolve_preset_SPEC(str(args.preset))
        ring = str(ring_t)
        sector = sector_t
        dy_m, dz_m = board_deltas_for_ring_sector_SPEC(ring_t, sector_t)
        tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(args.preset).strip())
    elif args.ring is not None:
        ring_p = parse_ring_SPEC(str(args.ring))
        if ring_p in ("bull_inner", "bull_outer"):
            sector = None
        else:
            if args.sector is None:
                print("--sector required for non-bull --ring", file=sys.stderr)
                return 1
            sector = int(args.sector)
        dy_m, dz_m = board_deltas_for_ring_sector_SPEC(ring_p, sector)
        ring = str(ring_p)
        tag = f"{ring}_s{sector}" if sector is not None else ring
    else:
        print("Provide --preset, or --ring (+ --sector), or --dy and --dz", file=sys.stderr)
        return 1

    if args.target_score is not None:
        assert_target_score_matches_aim_SPEC(int(args.target_score), dy_m, dz_m)

    aim_score = score_at_aim_SPEC(dy_m, dz_m)
    target_score_for_exact = float(args.target_score) if args.target_score is not None else float(aim_score)

    pool = np.asarray([[dy_m, dz_m]], dtype=float)
    cache_path = Path(args.warm_start_cache)
    ensure_warm_knots_cached_for_pool_SPEC(
        pool,
        cache_path,
        opt_n_restarts=int(args.opt_n_restarts),
        opt_maxiter=int(args.opt_maxiter),
        opt_restart_seed=int(args.opt_restart_seed),
        verbose=True,
    )

    torque_noise = not bool(args.no_torque_noise)
    env = DartResidualThrowEnv_SPEC(
        pool,
        torque_noise=torque_noise,
        torque_noise_warmup_steps=0,
        seed=int(args.seed),
        warm_start_cache_path=cache_path,
        opt_n_restarts=int(args.opt_n_restarts),
        opt_maxiter=int(args.opt_maxiter),
        opt_restart_seed=int(args.opt_restart_seed),
    )

    model = PPO.load(str(zip_path))

    n = int(args.n)
    rng_seed = int(args.seed)
    scores: list[float] = []
    dys: list[float] = []
    dzs: list[float] = []
    plane_hits: list[bool] = []
    face_hits: list[bool] = []

    from dartrobot.constants import R_BOARD_MISS_MM

    for i in range(n):
        obs, _ = env.reset(seed=rng_seed + i, options={"target_dy_m": dy_m, "target_dz_m": dz_m})
        obs_b = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = model.predict(obs_b, deterministic=True)
        _obs2, _r, _term, _trunc, info = env.step(action.reshape(-1))

        sc = float(info.get("score", 0.0))
        scores.append(sc)
        crossed = bool(info.get("plane_hit", False))
        plane_hits.append(crossed)
        dyv = info.get("delta_y_m", float("nan"))
        dzv = info.get("delta_z_m", float("nan"))
        dys.append(float(dyv) if crossed and np.isfinite(dyv) else float("nan"))
        dzs.append(float(dzv) if crossed and np.isfinite(dzv) else float("nan"))
        if crossed and np.isfinite(dyv) and np.isfinite(dzv):
            r_mm = float(np.hypot(float(dyv) * 1000.0, float(dzv) * 1000.0))
            face_hits.append(r_mm <= float(R_BOARD_MISS_MM))
        else:
            face_hits.append(False)

    scores_a = np.asarray(scores, dtype=float)
    mean_score = float(np.mean(scores_a))
    std_score = float(np.std(scores_a))
    board_plane_crossing_rate = float(np.mean(plane_hits))
    dartboard_face_hit_rate = float(np.mean(face_hits))
    exact = float(np.mean(scores_a == target_score_for_exact))

    aim_label = f"ring={ring} sector={sector}" if ring != "explicit" else "ring=explicit sector=None"
    print(f"aim: {aim_label} (Δy,Δz)=({dy_m:.5f},{dz_m:.5f}) m  B3_score_at_aim={aim_score}")
    print(f"n={n} seed={rng_seed} torque_noise={torque_noise}  policy={zip_path.name}")
    print(
        f"mean_score={mean_score:.4f}  std_score={std_score:.4f}  "
        f"board_plane_crossing_rate={board_plane_crossing_rate:.4f}  "
        f"dartboard_face_hit_rate={dartboard_face_hit_rate:.4f}  "
        f"frac_matching_target_score={exact:.4f}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"rl_eval_residual_mc_{tag}_n{n}_SPEC.png"
    json_path = out_dir / f"rl_eval_residual_mc_{tag}_n{n}_SPEC.json"

    manifest = {
        "policy_zip": str(zip_path),
        "preset": args.preset,
        "ring": ring,
        "sector": sector,
        "aim_delta_y_m": float(dy_m),
        "aim_delta_z_m": float(dz_m),
        "b3_score_at_aim": int(aim_score),
        "target_score_arg": args.target_score,
        "n_rollouts": n,
        "seed": rng_seed,
        "torque_noise": torque_noise,
        "mean_score": mean_score,
        "std_score": std_score,
        "board_plane_crossing_rate": board_plane_crossing_rate,
        "dartboard_face_hit_rate": dartboard_face_hit_rate,
        "fraction_matching_target_score": exact,
        "warm_start_cache": str(cache_path),
        "png_path": str(png_path.relative_to(_ROOT)),
        "json_path": str(json_path.relative_to(_ROOT)),
    }
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    try:
        plot_mc_board_SPEC(
            np.asarray(dys, dtype=float),
            np.asarray(dzs, dtype=float),
            scores_a,
            float(dy_m),
            float(dz_m),
            png_path,
            episode_totals=None,
            dartboard_face_hit_rate=dartboard_face_hit_rate,
        )
    except ImportError as e:
        print("matplotlib required to save plot:", e, file=sys.stderr)
        return 1
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
