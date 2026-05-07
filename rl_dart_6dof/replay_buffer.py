from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        i = self._idx
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = float(reward)
        self.next_state[i] = next_state
        self.done[i] = 1.0 if done else 0.0

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: have {self._size}, need {batch_size}.")
        idx = self.rng.integers(0, self._size, size=(batch_size,))
        return Batch(
            state=self.state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )

