#!/usr/bin/env python3
"""
train_sb3_residual_SPEC.py
==========================
Train a **goal-conditioned PPO** policy on ``DartResidualThrowEnv_SPEC`` (residual knots, sampled
targets from a pool, cached classical warm starts).

Install deps first::

    pip install -e ".[rl]"

Run from repo root::

    dartrobot rl-train --target-set trebles_bulls --timesteps 300000 --n-envs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        print("Install RL deps: pip install -e \".[rl]\"", e, file=sys.stderr)
        return 1

    from dartrobot.paths import policies_dir_SPEC, project_root_SPEC
    from dartrobot.rl.env_residual_throw import (
        STRONG_BASELINE_CACHE_DEFAULT_PATH_SPEC,
        DartResidualThrowEnv_SPEC,
        build_target_pool_SPEC,
        ensure_warm_knots_cached_for_pool_SPEC,
        eval_holdout_pool_SPEC,
    )

    _ROOT = project_root_SPEC()

    p = argparse.ArgumentParser(description="PPO goal-conditioned residual knot policy (one-step env).")
    p.add_argument(
        "--target-set",
        type=str,
        default="trebles_bulls",
        choices=(
            "full",
            "trebles_bulls",
            "singles_bulls",
            "trebles_singles_bulls",
            "trebles",
            "singles",
            "bulls",
            "custom",
        ),
        help="which (dy,dz) targets to sample each reset",
    )
    p.add_argument(
        "--targets",
        type=str,
        default=None,
        help="comma-separated presets when --target-set=custom (e.g. t10,t20,BULL)",
    )
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument(
        "--save",
        type=Path,
        default=policies_dir_SPEC() / "ppo_dart_goalcond",
        help="path prefix for SB3 save (adds .zip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4, help="parallel envs (Subproc when >1)")
    p.add_argument(
        "--warm-start-cache",
        type=Path,
        default=STRONG_BASELINE_CACHE_DEFAULT_PATH_SPEC,
        help="primary pickle: strong baseline (knots12+release_time); falls back to legacy warm_start_cache.pkl",
    )
    p.add_argument("--opt-n-restarts", type=int, default=2)
    p.add_argument("--opt-maxiter", type=int, default=60)
    p.add_argument("--opt-restart-seed", type=int, default=0)
    p.add_argument("--no-torque-noise", action="store_true")
    p.add_argument(
        "--torque-noise-warmup-steps",
        type=int,
        default=0,
        help="if >0 and torque noise on: train with noise off until this many timesteps, then enable",
    )
    p.add_argument("--checkpoint-freq", type=int, default=10_000, help="0 disables checkpoint callback")
    p.add_argument("--eval-freq", type=int, default=10_000, help="0 disables eval callback")
    p.add_argument("--n-eval-episodes", type=int, default=16)
    args = p.parse_args(argv)

    custom_list: list[str] | None = None
    if args.target_set == "custom":
        if not args.targets:
            print("--targets required when --target-set=custom", file=sys.stderr)
            return 1
        custom_list = [x.strip() for x in str(args.targets).split(",") if x.strip()]

    target_pool = build_target_pool_SPEC(args.target_set, custom_presets=custom_list)
    eval_pool = eval_holdout_pool_SPEC()

    cache_path = Path(args.warm_start_cache)
    print(f"Pre-warming classical knot cache for {target_pool.shape[0]} training targets → {cache_path}")
    ensure_warm_knots_cached_for_pool_SPEC(
        target_pool,
        cache_path,
        opt_n_restarts=int(args.opt_n_restarts),
        opt_maxiter=int(args.opt_maxiter),
        opt_restart_seed=int(args.opt_restart_seed),
        verbose=True,
    )
    print(f"Pre-warming eval holdout pool ({eval_pool.shape[0]} targets)…")
    ensure_warm_knots_cached_for_pool_SPEC(
        eval_pool,
        cache_path,
        opt_n_restarts=int(args.opt_n_restarts),
        opt_maxiter=int(args.opt_maxiter),
        opt_restart_seed=int(args.opt_restart_seed) + 10_000,
        verbose=False,
    )

    torque_noise = not bool(args.no_torque_noise)
    warmup_steps = int(args.torque_noise_warmup_steps) if torque_noise else 0

    def _make_train() -> DartResidualThrowEnv_SPEC:
        return DartResidualThrowEnv_SPEC(
            target_pool,
            torque_noise=torque_noise,
            torque_noise_warmup_steps=warmup_steps,
            seed=int(args.seed),
            warm_start_cache_path=cache_path,
            opt_n_restarts=int(args.opt_n_restarts),
            opt_maxiter=int(args.opt_maxiter),
            opt_restart_seed=int(args.opt_restart_seed),
        )

    def _make_eval() -> DartResidualThrowEnv_SPEC:
        return DartResidualThrowEnv_SPEC(
            eval_pool,
            torque_noise=torque_noise,
            torque_noise_warmup_steps=0,
            seed=int(args.seed) + 999,
            warm_start_cache_path=cache_path,
            opt_n_restarts=int(args.opt_n_restarts),
            opt_maxiter=int(args.opt_maxiter),
            opt_restart_seed=int(args.opt_restart_seed),
        )

    n_envs = max(1, int(args.n_envs))
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        _make_train,
        n_envs=n_envs,
        seed=int(args.seed),
        vec_env_cls=vec_cls,
    )

    class TorqueNoiseWarmupCallback(BaseCallback):
        """After ``warmup_steps`` timesteps, set ``set_torque_noise_runtime(target)`` on all sub-envs."""

        def __init__(self, warmup_steps_inner: int, runtime_after: bool, verbose_inner: int = 0):
            super().__init__(verbose_inner)
            self._warmup = int(max(0, warmup_steps_inner))
            self._runtime_after = bool(runtime_after)
            self._done = False

        def _on_step(self) -> bool:
            if self._done or self._warmup <= 0:
                return True
            if self.num_timesteps >= self._warmup:
                try:
                    self.training_env.env_method("set_torque_noise_runtime", self._runtime_after)
                except AttributeError:
                    pass
                self._done = True
            return True

    callbacks = []
    if warmup_steps > 0 and torque_noise:
        callbacks.append(TorqueNoiseWarmupCallback(warmup_steps, torque_noise, verbose=0))

    save_dir = Path(args.save).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = Path(args.save).name

    if int(args.checkpoint_freq) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=int(args.checkpoint_freq),
                save_path=str(save_dir),
                name_prefix=f"{save_prefix}_ckpt",
                verbose=1,
            )
        )

    eval_env = None
    if int(args.eval_freq) > 0:
        eval_env = make_vec_env(_make_eval, n_envs=1, seed=int(args.seed) + 12345)
        # SB3 appends .zip if missing
        best_path = save_dir / "best_model"
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_path),
                log_path=str(save_dir / "eval_logs"),
                eval_freq=int(args.eval_freq),
                n_eval_episodes=int(args.n_eval_episodes),
                deterministic=True,
                verbose=1,
            )
        )

    cb = CallbackList(callbacks) if callbacks else None

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=int(args.seed),
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=int(args.timesteps), callback=cb)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Saved policy to {save_path}.zip")
    if eval_env is not None:
        eval_env.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
