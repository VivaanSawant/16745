from __future__ import annotations

import copy

import torch
import torch.nn as nn


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU, out_activation: type[nn.Module] | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """
    Deterministic policy for DDPG-style training.
    - Input: state (12)
    - Output: action (7) with tanh squash -> [-1, 1]

    Convention:
      - First 6 outputs are torques in [-1, 1]
      - Last output is release signal mapped to [0, 1] via (tanh + 1)/2 at action() time
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = mlp([state_dim, hidden, hidden, action_dim], activation=nn.ReLU, out_activation=nn.Tanh)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CriticNet(nn.Module):
    """
    Q(s,a) critic.
    - Input: concat(state, action) = 12 + 7
    - Output: scalar Q
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = mlp([state_dim + action_dim, hidden, hidden, 1], activation=nn.ReLU, out_activation=None)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


def make_target(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    return target

