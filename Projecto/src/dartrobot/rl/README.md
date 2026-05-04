# RL phase (goal-conditioned residual throws)

Optional **reinforcement learning** on top of the SPEC MuJoCo + flight stack. A **goal-conditioned** PPO policy learns a **bounded residual** on 12 spline knots (added to a **classical-optimized warm start**).

## Layout (in package)

| Module | Role |
|--------|------|
| `dartrobot.rl.env_residual_throw` | Gymnasium one-step env: obs `[q, qdot, target_Δy, target_Δz]`; action = residual on knots; warm start from strong baseline cache + legacy pickle fallback. |
| `dartrobot.rl.train_sb3_residual` | SB3 **PPO** training (vectorized envs, checkpoints, eval callback). |
| `dartrobot.rl.eval_sb3_residual` | Load `.zip`, Monte Carlo at one target; saves PNG + JSON under `artifacts/rl/`. |

Checkpoints and legacy warm cache live in repo-root **`policies/`** (not inside `src/`).

## Install

```bash
pip install -e ".[rl]"
```

## Train

```bash
dartrobot rl-train \
  --target-set trebles_bulls \
  --timesteps 300000 \
  --n-envs 4 \
  --save policies/ppo_dart_goalcond
```

- Default `--warm-start-cache` → `artifacts/baseline/strong_baseline_cache.pkl`; falls back to `policies/warm_start_cache.pkl`.
- Pre-warm fills missing keys via classical optimizer when absent from both caches.

## Evaluate

```bash
dartrobot rl-eval \
  --policy-zip policies/ppo_dart_goalcond.zip \
  --preset t10 \
  --n 200 \
  --seed 0
```

## Classical baseline (no RL)

```bash
dartrobot mc --preset t10 --n 200 --optimize-knots --opt-n-restarts 10 --opt-maxiter 80
dartrobot baseline --target-set bulls --n-restarts 6 --maxiter 80
```
