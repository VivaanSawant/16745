"""
Actor-critic (DDPG-style) training for DartEnv.

Rollout: state -> policy (+ Gaussian exploration) -> env action -> env.step ->
store (s,a,r,s',done).

Updates:
  critic: MSE(Q(s,a), r + gamma * (1-done) * Q_target(s', mu_target(s')))
  policy: maximize Q(s, mu(s))  -> minimize -mean(Q(...))
  soft-update Q_target toward Q periodically.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import CFG, Config
from env import DartEnv
from models import CriticNet, PolicyNet, make_target
from replay_buffer import ReplayBuffer
from utils import ensure_dir, set_seed, soft_update, to_tensor


def raw_policy_to_env_action(raw: np.ndarray) -> np.ndarray:
    """Map policy output [-1,1]^7 to env torques [-1,1]^6 + release [0,1]."""
    raw = np.asarray(raw, dtype=np.float32).reshape(7,)
    tau = np.clip(raw[:6], -1.0, 1.0)
    rel = float(np.clip((raw[6] + 1.0) / 2.0, 0.0, 1.0))
    return np.concatenate([tau, np.array([rel], dtype=np.float32)])


def exploration_noise(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(0.0, cfg.action_noise_std, size=(6,)).astype(np.float32)
    n = np.clip(n, -cfg.action_noise_clip, cfg.action_noise_clip)
    rn = float(rng.normal(0.0, cfg.release_noise_std))
    rn = float(np.clip(rn, -0.3, 0.3))
    return np.concatenate([n, np.array([rn], dtype=np.float32)])


def env_action_torch(raw_t: torch.Tensor) -> torch.Tensor:
    """Batch: raw [-1,1] -> concatenated env action."""
    tau = torch.clamp(raw_t[:, :6], -1.0, 1.0)
    rel = torch.clamp((raw_t[:, 6:7] + 1.0) / 2.0, 0.0, 1.0)
    return torch.cat([tau, rel], dim=-1)


@torch.no_grad()
def evaluate_policy(
    env: DartEnv,
    actor: PolicyNet,
    device: str,
    n_episodes: int,
) -> dict[str, float]:
    rewards: list[float] = []
    for _ in range(n_episodes):
        s = env.reset()
        # Optional eval seed perturbation keeps env RNG fixed per-eval if desired.
        ep_r = 0.0
        done = False
        while not done:
            st = to_tensor(s, device=device).unsqueeze(0)
            raw = actor(st).squeeze(0).cpu().numpy()
            a = raw_policy_to_env_action(raw)
            s, r, done, _ = env.step(a)
            ep_r += r
        rewards.append(ep_r)
    return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    env = DartEnv(
        dt=cfg.dt,
        max_steps=cfg.max_steps,
        release_threshold=cfg.release_threshold,
        seed=cfg.seed,
    )
    rng = np.random.default_rng(cfg.seed + 999)

    actor = PolicyNet(cfg.state_dim, cfg.action_dim).to(device)
    critic = CriticNet(cfg.state_dim, cfg.action_dim).to(device)
    target_critic = make_target(critic).to(device)

    actor_opt = Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = Adam(critic.parameters(), lr=cfg.critic_lr)

    buf = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size, seed=cfg.seed + 42)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.save_dir)
    ensure_dir(save_dir)

    s = env.reset()
    ep_reward = 0.0
    ep_len = 0
    best_eval = -1e18

    for step in range(1, cfg.total_env_steps + 1):
        # --- action ---
        with torch.no_grad():
            st = to_tensor(s, device=device).unsqueeze(0)
            raw = actor(st).squeeze(0).cpu().numpy()
        if step < cfg.start_steps:
            raw = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
        else:
            raw = np.clip(raw + exploration_noise(cfg, rng), -1.0, 1.0)

        a_env = raw_policy_to_env_action(raw)
        s2, r, done, _ = env.step(a_env)
        buf.add(s, a_env, r, s2, done)

        s = s2 if not done else env.reset()
        ep_reward += r
        ep_len += 1
        if done:
            if step % 1000 == 0:
                print(f"step={step} last_ep_reward={ep_reward:.3f} ep_len={ep_len}")
            ep_reward = 0.0
            ep_len = 0

        # --- updates ---
        if len(buf) >= cfg.batch_size and step >= cfg.start_steps:
            for _ in range(cfg.updates_per_step):
                b = buf.sample(cfg.batch_size)
                bs = to_tensor(b.state, device)
                ba = to_tensor(b.action, device)
                br = to_tensor(b.reward, device)
                bns = to_tensor(b.next_state, device)
                bd = to_tensor(b.done, device)

                with torch.no_grad():
                    next_raw = actor(bns)
                    next_a = env_action_torch(next_raw)
                    target_q = br + cfg.gamma * (1.0 - bd) * target_critic(bns, next_a)

                q = critic(bs, ba)
                critic_loss = F.mse_loss(q, target_q)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                cur_raw = actor(bs)
                cur_a = env_action_torch(cur_raw)
                policy_loss = -critic(bs, cur_a).mean()

                actor_opt.zero_grad()
                policy_loss.backward()
                actor_opt.step()

                soft_update(target_critic, critic, cfg.tau)

        if step % cfg.eval_every_steps == 0 and step >= cfg.start_steps:
            stats = evaluate_policy(env, actor, device, cfg.eval_episodes)
            print(f"[eval @ {step}] {stats}")
            if stats["mean_reward"] > best_eval:
                best_eval = stats["mean_reward"]
                path = os.path.join(save_dir, "policy_best.pt")
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "target_critic": target_critic.state_dict(),
                        "cfg": asdict(cfg),
                        "step": step,
                    },
                    path,
                )

    final_path = os.path.join(save_dir, "policy_final.pt")
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "cfg": asdict(cfg),
            "step": cfg.total_env_steps,
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train DDPG-style agent on DartEnv.")
    p.add_argument("--total-env-steps", type=int, default=None, help="Override CFG.total_env_steps")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = CFG
    if args.total_env_steps is not None:
        te = args.total_env_steps
        cfg = replace(
            cfg,
            total_env_steps=te,
            start_steps=min(cfg.start_steps, max(100, te // 10)),
            eval_every_steps=max(500, te // 5),
        )
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    train(cfg)


if __name__ == "__main__":
    main()
