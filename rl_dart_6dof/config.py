from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Problem dimensions
    state_dim: int = 12
    action_dim: int = 7  # 6 torques + 1 release signal

    # Environment
    max_steps: int = 80
    dt: float = 0.04
    release_threshold: float = 0.75

    # RL
    gamma: float = 0.99
    tau: float = 0.005  # soft update rate for target critic

    # Optimization
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256

    # Replay buffer
    buffer_size: int = 300_000
    start_steps: int = 2_000  # random/exploration steps before training updates

    # Exploration noise (Gaussian on torques, mild on release)
    action_noise_std: float = 0.20
    release_noise_std: float = 0.05
    action_noise_clip: float = 0.50

    # Training loop
    total_env_steps: int = 60_000
    updates_per_step: int = 1
    eval_every_steps: int = 10_000
    eval_episodes: int = 10

    # Misc
    seed: int = 0
    device: str = "cpu"
    save_dir: str = "checkpoints"


CFG = Config()

