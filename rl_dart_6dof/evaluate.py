"""
Evaluate learned policy + effective degrees of freedom (DOF).

1. Roll episodes: log joint angles and actions over time.
2. PCA (sklearn) on stacked [angles, velocities] along the trajectory timeline.
3. Effective DOF at 95%% cumulative variance.
4. Per-joint torque variance from action trajectories.

Saves PCA and torque-variance plots under eval_outputs/ by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, replace

import numpy as np
import torch
from sklearn.decomposition import PCA

# Ensure sibling imports when run as script from any cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import CFG, Config
from env import DartEnv
from models import PolicyNet
from train import raw_policy_to_env_action
from utils import plot_pca_variance, plot_torque_variance, to_tensor, TrajectoryLogger


def load_actor(path: str, device: str) -> tuple[PolicyNet, Config]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    raw_cfg = ckpt.get("cfg", {})
    if isinstance(raw_cfg, dict) and raw_cfg:
        names = {f.name for f in fields(Config)}
        overrides = {k: v for k, v in raw_cfg.items() if k in names}
        cfg = replace(CFG, **overrides)
    else:
        cfg = CFG
    actor = PolicyNet(cfg.state_dim, cfg.action_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, cfg


def collect_trajectories(
    env: DartEnv,
    actor: PolicyNet,
    device: str,
    n_episodes: int,
) -> list[dict]:
    runs: list[dict] = []
    release_flags: list[float] = []
    landing_dists: list[float] = []
    with torch.no_grad():
        for _ in range(n_episodes):
            log = TrajectoryLogger()
            s = env.reset()
            done = False
            last_info = None
            while not done:
                st = to_tensor(s, device=device).unsqueeze(0)
                raw = actor(st).squeeze(0).cpu().numpy()
                a = raw_policy_to_env_action(raw)
                s2, r, done, _ = env.step(a)
                log.add(s, a, r)
                s = s2
                last_info = _["info"]
            runs.append(log.as_arrays())
            if last_info is not None:
                release_flags.append(1.0 if last_info.released else 0.0)
                if last_info.landing_xy is not None:
                    x, y = last_info.landing_xy
                    d = float(np.sqrt((x - env.target_x) ** 2 + (y - env.target_y) ** 2))
                    landing_dists.append(d)
    if release_flags:
        runs.append(
            {
                "_meta_release_rate": float(np.mean(release_flags)),
                "_meta_mean_dist": float(np.mean(landing_dists)) if landing_dists else float("nan"),
                "_meta_std_dist": float(np.std(landing_dists)) if landing_dists else float("nan"),
            }
        )
    return runs


def effective_dof_95(explained_ratio: np.ndarray) -> int:
    c = np.cumsum(explained_ratio)
    for i, val in enumerate(c):
        if val >= 0.95:
            return i + 1
    return len(explained_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="DOF analysis (PCA + torque variance)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy_*.pt (default: checkpoints/policy_best.pt or policy_final.pt)",
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = _SCRIPT_DIR
    cand = args.checkpoint
    if cand is None:
        for name in ("policy_best.pt", "policy_final.pt"):
            p = os.path.join(base, "checkpoints", name)
            if os.path.isfile(p):
                cand = p
                break
    if cand is None or not os.path.isfile(cand):
        print("No checkpoint found. Train first: python train.py")
        sys.exit(1)

    actor, cfg = load_actor(cand, device)
    env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        seed=args.seed,
    )

    runs = collect_trajectories(env, actor, device, args.episodes)
    meta = runs[-1] if isinstance(runs[-1], dict) and any(k.startswith("_meta_") for k in runs[-1].keys()) else None
    if meta is not None:
        runs = runs[:-1]

    # Concatenate all timesteps: state features for PCA (angles + velocities)
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for d in runs:
        n = d["angles"].shape[0]
        ang = d["angles"]
        vel = d["velocities"]
        state_rows.append(np.concatenate([ang, vel], axis=1))
        action_rows.append(d["actions"])

    X = np.vstack(state_rows).astype(np.float64)
    A = np.vstack(action_rows).astype(np.float64)
    torque = A[:, :6]

    # PCA on centered state trajectory (12-D per row)
    n_comp = min(12, max(1, X.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    explained = pca.explained_variance_ratio_
    dof_95 = effective_dof_95(explained)
    cumulative = np.cumsum(explained)

    print(f"Checkpoint: {cand}")
    print(f"Trajectory samples (rows): {X.shape[0]}")
    print(f"PCA components used: {n_comp}")
    print("Per-component explained variance ratio:")
    for i, r_i in enumerate(explained[:12], start=1):
        print(f"  PC{i}: {r_i:.4f} (cum: {cumulative[i-1]:.4f})")
    print(f"Effective DOF (95% variance): {dof_95}")
    if meta is not None:
        print(f"Release rate: {meta['_meta_release_rate']:.3f}")
        print(f"Mean landing distance to target (m): {meta['_meta_mean_dist']:.4f} (std {meta['_meta_std_dist']:.4f})")

    torque_var = np.var(torque, axis=0)
    print("Per-joint torque variance:")
    for j in range(6):
        print(f"  joint {j+1}: {torque_var[j]:.6f}")

    out_dir = os.path.join(base, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    pca_path = os.path.join(out_dir, "pca_variance.png")
    torque_path = os.path.join(out_dir, "torque_variance.png")
    plot_pca_variance(np.asarray(explained, dtype=np.float32), pca_path)
    plot_torque_variance(np.asarray(torque_var, dtype=np.float32), torque_path)
    print(f"Saved plots:\n  {pca_path}\n  {torque_path}")

    try:
        import matplotlib.pyplot as plt

        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
