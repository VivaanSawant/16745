from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepInfo:
    released: bool
    landing_xy: tuple[float, float] | None
    score: float


class DartEnv:
    """
    Minimal 6-DOF arm + dart throw environment.

    State (12):
      - 6 joint angles
      - 6 joint velocities

    Action (7):
      - 6 joint torques (continuous, [-1, 1] expected)
      - 1 release signal (continuous, [0, 1] expected)

    Physics is intentionally simple but structured so you can swap in better models later:
      - joint dynamics: v_{t+1} = v_t + dt*(k_tau*tau - damping*v_t), theta_{t+1} = theta_t + dt*v_{t+1}
      - release triggers a ballistic flight from a "release point" with velocity derived from joint state
    """

    def __init__(
        self,
        dt: float = 0.04,
        max_steps: int = 80,
        release_threshold: float = 0.75,
        seed: int = 0,
    ) -> None:
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.release_threshold = float(release_threshold)
        self.rng = np.random.default_rng(seed)

        self.state_dim = 12
        self.action_dim = 7

        # Joint dynamics parameters (kept explicit to allow future tuning)
        self.k_tau = 7.5  # torque-to-acceleration gain
        self.damping = 1.2
        self.max_vel = 12.0
        self.max_angle = np.pi

        # Throw / board parameters
        self.g = 9.81
        self.release_height = 1.5  # meters above ground plane

        # "Dartboard" is on ground plane at y=0, centered at x=target_x
        self.target_x = 3.0
        self.target_y = 0.0

        # Internal state
        self._t = 0
        self._angles = np.zeros(6, dtype=np.float32)
        self._vels = np.zeros(6, dtype=np.float32)

        # Fixed random projection from joint features -> release velocity (structured but simple)
        # Features: [sin(theta), cos(theta), v] (18 dims) -> v_xyz (3 dims)
        w = self.rng.normal(0.0, 1.0, size=(3, 18)).astype(np.float32)
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self._W_vel = w

    def reset(self) -> np.ndarray:
        self._t = 0
        # Small random initial pose and velocity
        self._angles = self.rng.uniform(low=-0.1, high=0.1, size=(6,)).astype(np.float32)
        self._vels = self.rng.uniform(low=-0.05, high=0.05, size=(6,)).astype(np.float32)
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected action shape (7,), got {action.shape}.")

        torques = np.clip(action[:6], -1.0, 1.0)
        release = float(np.clip(action[6], 0.0, 1.0))

        # 1) Apply torques to update state
        acc = self.k_tau * torques - self.damping * self._vels
        self._vels = np.clip(self._vels + self.dt * acc, -self.max_vel, self.max_vel)
        self._angles = np.clip(self._angles + self.dt * self._vels, -self.max_angle, self.max_angle)

        self._t += 1

        # 2) Release -> simulate projectile and score
        done = False
        reward = 0.0
        info = StepInfo(released=False, landing_xy=None, score=0.0)

        if release >= self.release_threshold or self._t >= self.max_steps:
            done = True
            released = release >= self.release_threshold
            landing = self._simulate_landing_xy()
            score = self._dartboard_score(landing[0], landing[1])
            # Reward: score if released, otherwise small penalty for timing out
            reward = float(score if released else 0.1 * score)
            info = StepInfo(released=released, landing_xy=landing, score=float(score))

        return self._get_state(), reward, done, {"info": info}

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self._angles, self._vels], axis=0).astype(np.float32)

    def _release_velocity_xyz(self) -> np.ndarray:
        # A simple "kinematic" map from joint pose/velocity to a 3D release velocity.
        s = np.sin(self._angles)
        c = np.cos(self._angles)
        feats = np.concatenate([s, c, self._vels], axis=0).astype(np.float32)  # (18,)
        v = self._W_vel @ feats  # (3,)

        # Encourage realistic directionality: mostly forward (+x) and mild upward (+z)
        v[0] = abs(v[0]) + 6.0
        v[1] = 0.2 * v[1]
        v[2] = 0.5 * v[2] + 2.0

        # Cap speed to avoid extreme outliers early in training
        speed = float(np.linalg.norm(v))
        if speed > 20.0:
            v *= 20.0 / (speed + 1e-8)
        return v.astype(np.float32)

    def _simulate_landing_xy(self) -> tuple[float, float]:
        # Release from origin at height h; land when z(t)=0.
        v = self._release_velocity_xyz()
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        h = float(self.release_height)
        g = float(self.g)

        # Solve 0 = h + vz*t - 0.5*g*t^2 for positive t.
        # t = (vz + sqrt(vz^2 + 2gh)) / g
        disc = vz * vz + 2.0 * g * h
        t = (vz + float(np.sqrt(max(disc, 0.0)))) / g
        x = vx * t
        y = vy * t
        return (float(x), float(y))

    def _dartboard_score(self, x: float, y: float) -> float:
        """
        Simplified dartboard scoring based on distance to a target center.
        - Bullseye-like center gets high score.
        - Score decays with radial distance.
        - This is intentionally smooth-ish for learning, but still interpretable.
        """
        dx = x - self.target_x
        dy = y - self.target_y
        r = float(np.sqrt(dx * dx + dy * dy))

        # Smooth score with ring bonuses. Units are "meters" here; tune for your sim.
        base = 50.0 * float(np.exp(-2.5 * r * r))
        ring_bonus = 0.0
        if r < 0.05:
            ring_bonus = 50.0
        elif r < 0.15:
            ring_bonus = 25.0
        elif r < 0.30:
            ring_bonus = 10.0
        return float(base + ring_bonus)

