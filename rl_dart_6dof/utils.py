from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (state - mean) / (std + 1e-8)


@dataclass
class TrajectoryLogger:
    """Stores a single episode trajectory for analysis/plots."""

    angles: list[np.ndarray] = field(default_factory=list)
    velocities: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.angles.append(state[:6].copy())
        self.velocities.append(state[6:].copy())
        self.actions.append(action.copy())
        self.rewards.append(float(reward))

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "angles": np.asarray(self.angles, dtype=np.float32),
            "velocities": np.asarray(self.velocities, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
        }


def plot_pca_variance(explained: np.ndarray, save_path: str) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    xs = np.arange(1, len(explained) + 1)
    cum = np.cumsum(explained)

    plt.figure(figsize=(7, 4))
    plt.plot(xs, cum, marker="o")
    plt.axhline(0.95, linestyle="--", color="red", linewidth=1, label="95% threshold")
    plt.xticks(xs)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_torque_variance(torque_var: np.ndarray, save_path: str) -> None:
    d = os.path.dirname(save_path)
    if d:
        ensure_dir(d)
    plt.figure(figsize=(7, 4))
    joints = np.arange(1, len(torque_var) + 1)
    plt.bar(joints, torque_var)
    plt.xticks(joints)
    plt.xlabel("Joint index")
    plt.ylabel("Torque variance")
    plt.title("Per-joint torque usage (variance)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def pretty_dict(d: dict[str, Any]) -> str:
    keys = sorted(d.keys())
    parts = []
    for k in keys:
        v = d[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)

