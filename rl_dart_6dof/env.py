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
        self.min_release_steps = 10  # prevents trivial immediate-release solutions

        # "Dartboard" is on ground plane at y=0, centered at x=target_x
        self.target_x = 3.0
        self.target_y = 0.0

        # Internal state
        self._t = 0
        self._angles = np.zeros(6, dtype=np.float32)
        self._vels = np.zeros(6, dtype=np.float32)

        # Reward shaping (accuracy-first)
        # - A dense shaping term uses the distance-to-target you'd get if you released NOW.
        # - Final reward on release is a sharp function of landing distance (dominant signal).
        # - Timeout is strongly negative to avoid "never release".
        self.step_penalty = 0.005
        self.timeout_penalty = 100.0

        # Distance shaping weights
        self.shaping_dist_weight = 0.05  # per-step shaping: -w * predicted_dist
        self.hit_sigma = 0.25  # meters, controls sharpness of hit reward
        self.release_dist_penalty = 5.0  # subtract on release: encourages accuracy (meters)

        # Note: release velocity model is deterministic and structured to be learnable:
        # joint0 -> azimuth (left/right), joint1 -> elevation (up/down),
        # remaining joints/velocities modulate speed.

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

        # 2) Dense shaping: how close we'd land if we released right now
        pred_xy = self._simulate_landing_xy()
        pred_dist = float(np.sqrt((pred_xy[0] - self.target_x) ** 2 + (pred_xy[1] - self.target_y) ** 2))

        # 3) Release -> simulate projectile and score
        done = False
        # Encourage the policy to eventually raise the release signal after it is allowed to release.
        can_release = self._t >= self.min_release_steps
        release_enc = (0.5 * release) if can_release else 0.0
        reward = -self.step_penalty - self.shaping_dist_weight * pred_dist + float(release_enc)
        info = StepInfo(released=False, landing_xy=None, score=0.0)

        if (can_release and release >= self.release_threshold) or self._t >= self.max_steps:
            done = True
            released = bool(can_release and release >= self.release_threshold)
            landing = self._simulate_landing_xy()
            dist = float(np.sqrt((landing[0] - self.target_x) ** 2 + (landing[1] - self.target_y) ** 2))
            score = self._dartboard_score(landing[0], landing[1])

            # Reward:
            # - If released: accuracy-dominant reward based on landing distance, plus a small score term.
            # - If timed out: strong negative penalty.
            if released:
                # Gaussian hit reward in [0, 100], sharply peaked at the target center.
                hit = 100.0 * float(np.exp(-0.5 * (dist / self.hit_sigma) ** 2))
                reward = float(hit + 0.2 * score - self.release_dist_penalty * dist)
            else:
                reward = float(-self.timeout_penalty)
            info = StepInfo(released=released, landing_xy=landing, score=float(score))

        return self._get_state(), reward, done, {"info": info, "t": self._t, "pred_dist": pred_dist}

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self._angles, self._vels], axis=0).astype(np.float32)

    def _release_velocity_xyz(self) -> np.ndarray:
        # Deterministic "throw kinematics" that is easier to learn to aim with.
        # Azimuth (left/right) and elevation (up/down) come from two joints.
        az = float(self._angles[0])          # yaw-like
        el = float(self._angles[1])          # pitch-like

        # Map remaining joint speeds into a positive forward speed.
        # Use a smooth, bounded function so learning is stable.
        vmag = float(np.linalg.norm(self._vels[2:]))  # joints 2..5 contribute
        speed = 6.0 + 6.0 * float(np.tanh(0.25 * vmag))  # in ~[6,12)

        # Clamp angles to plausible throwing ranges.
        az = float(np.clip(az, -0.8, 0.8))
        el = float(np.clip(el, -0.2, 1.0))

        # Convert to Cartesian velocity.
        vx = speed * float(np.cos(el)) * float(np.cos(az))
        vy = speed * float(np.cos(el)) * float(np.sin(az))
        vz = speed * float(np.sin(el)) + 1.5  # baseline upward component

        # Mild caps (safety)
        vx = float(np.clip(vx, 0.5, 20.0))
        vy = float(np.clip(vy, -10.0, 10.0))
        vz = float(np.clip(vz, -2.0, 15.0))
        return np.asarray([vx, vy, vz], dtype=np.float32)

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

