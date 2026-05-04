"""
RL_env_scaffold_SPEC.py
=======================
Minimal RL-facing scaffold for the dart-throwing project.

This module does **not** implement training. It provides a lightweight environment-style
API around the current MuJoCo arm -> release -> projectile -> score pipeline so future
SAC/TD3/PPO code has a stable interface to target.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_WRIST_DEG,
    THROW_DURATION_S,
)
from track_A_arm_SPEC.A1_link_geometry_and_inertia_SPEC import joint_limits_rad_SPEC
from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    DEFAULT_MUJOCO_PD_KD_SPEC,
    DEFAULT_MUJOCO_PD_KP_SPEC,
    DEFAULT_MUJOCO_THROW_KNOTS_SPEC,
    simulate_throw_mujoco_SPEC,
)
from track_C_integration_SPEC.C1_pipeline_arm_release_to_projectile_score_SPEC import (
    score_from_release_state_SPEC,
)
from track_C_integration_SPEC.C2_jacobian_covariance_optimizer_SPEC import (
    landing_info_for_release_SPEC,
)


def default_reset_observation_SPEC(wind_xyz=(0.0, 0.0, 0.0), include_wind: bool = True) -> np.ndarray:
    """Observation at reset: [q0, q1, q2, qd0, qd1, qd2, wind?]."""
    q0 = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
    obs = [*q0.tolist(), 0.0, 0.0, 0.0]
    if include_wind:
        obs.extend(float(w) for w in wind_xyz)
    return np.asarray(obs, dtype=float)


def knots_action_bounds_SPEC() -> tuple[np.ndarray, np.ndarray]:
    """
    Box bounds for the 9 spline-knot action vector.

    Each knot for a given joint is bounded by that joint's PDF limit.
    """
    lim = joint_limits_rad_SPEC()
    low = np.array([lim[0][0]] * 3 + [lim[1][0]] * 3 + [lim[2][0]] * 3, dtype=float)
    high = np.array([lim[0][1]] * 3 + [lim[1][1]] * 3 + [lim[2][1]] * 3, dtype=float)
    return low, high


def clip_knots_action_SPEC(knots9: np.ndarray) -> np.ndarray:
    """Project a candidate 9D action into joint-limit bounds."""
    low, high = knots_action_bounds_SPEC()
    return np.clip(np.asarray(knots9, dtype=float).reshape(9), low, high)


@dataclass
class DartThrowEnvConfig_SPEC:
    """
    Configuration for the one-step throw environment.

    `action_includes_release_time=True` makes the action dimension 10, with the final
    scalar interpreted directly as release time in seconds.
    """

    wind_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drag_enabled: bool = True
    torque_noise: bool = True
    include_wind_in_obs: bool = True
    action_includes_release_time: bool = False
    kp: float = DEFAULT_MUJOCO_PD_KP_SPEC
    kd: float = DEFAULT_MUJOCO_PD_KD_SPEC
    reward_torque_l2_coeff: float = 0.0
    reward_speed_tracking_coeff: float = 0.0
    target_release_speed_mps: float = 5.5


class DartThrowingOneStepEnv_SPEC:
    """
    Minimal Gym-like environment with one throw per episode.

    Reset observation is the cocked arm state. A single action chooses the throw spline
    (and optionally release time), then `step()` returns a terminal transition.
    """

    def __init__(self, config: DartThrowEnvConfig_SPEC | None = None):
        self.config = DartThrowEnvConfig_SPEC() if config is None else config
        self._rng = np.random.default_rng()
        self._terminated = False
        self._reset_obs = default_reset_observation_SPEC(
            wind_xyz=self.config.wind_xyz,
            include_wind=self.config.include_wind_in_obs,
        )
        self.default_action = DEFAULT_MUJOCO_THROW_KNOTS_SPEC.copy()

    @property
    def observation_dim(self) -> int:
        return int(self._reset_obs.size)

    @property
    def action_dim(self) -> int:
        return 10 if self.config.action_includes_release_time else 9

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._terminated = False
        self._reset_obs = default_reset_observation_SPEC(
            wind_xyz=self.config.wind_xyz,
            include_wind=self.config.include_wind_in_obs,
        )
        info = {
            "action_dim": self.action_dim,
            "observation_dim": self.observation_dim,
            "default_action": self.default_action.copy(),
        }
        return self._reset_obs.copy(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("Episode already terminated; call reset() before step().")

        action = np.asarray(action, dtype=float).reshape(-1)
        expected_dim = self.action_dim
        if action.size != expected_dim:
            raise ValueError(f"Expected action dim {expected_dim}, got {action.size}")

        if self.config.action_includes_release_time:
            knots9 = clip_knots_action_SPEC(action[:9])
            release_time_s = float(np.clip(action[9], 0.05 * THROW_DURATION_S, 0.99 * THROW_DURATION_S))
        else:
            knots9 = clip_knots_action_SPEC(action[:9])
            release_time_s = None

        q_start3 = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
        sim = simulate_throw_mujoco_SPEC(
            knots9,
            q_start3=q_start3,
            release_time_s=release_time_s,
            rng=self._rng,
            torque_noise=self.config.torque_noise,
            kp=self.config.kp,
            kd=self.config.kd,
        )
        s6 = sim["release_state6"]
        terminal_obs = np.zeros_like(self._reset_obs)
        self._terminated = True

        if s6 is None:
            info = {
                "score": 0.0,
                "hit": False,
                "release_state6": None,
                "terminated_reason": "no_release_state",
                "terminal_observation": terminal_obs.copy(),
            }
            return terminal_obs, 0.0, True, False, info

        rollout = score_from_release_state_SPEC(
            s6,
            wind_xyz=self.config.wind_xyz,
            drag_enabled=self.config.drag_enabled,
        )
        landing = landing_info_for_release_SPEC(
            s6,
            wind_xyz=self.config.wind_xyz,
            drag_enabled=self.config.drag_enabled,
        )
        reward = float(rollout["score"])
        if self.config.reward_torque_l2_coeff != 0.0:
            reward -= float(self.config.reward_torque_l2_coeff * np.mean(np.sum(sim["tau"] ** 2, axis=1)))
        if self.config.reward_speed_tracking_coeff != 0.0:
            speed = float(np.linalg.norm(s6[3:]))
            reward -= float(self.config.reward_speed_tracking_coeff * abs(speed - self.config.target_release_speed_mps))

        info = {
            "score": float(rollout["score"]),
            "hit": bool(rollout["hit"]),
            "delta_y_m": float(rollout["delta_y_m"]) if rollout["hit"] else np.nan,
            "delta_z_m": float(rollout["delta_z_m"]) if rollout["hit"] else np.nan,
            "release_state6": s6.copy(),
            "landing_m": landing["landing_m"].copy(),
            "release_time_s": float(sim["release_time_s"]),
            "knots9_used": knots9.copy(),
            "terminal_observation": terminal_obs.copy(),
        }
        return terminal_obs, reward, True, False, info


def evaluate_action_mc_SPEC(
    action9_or10: np.ndarray,
    n_rollouts: int = 40,
    seed: int = 0,
    config: DartThrowEnvConfig_SPEC | None = None,
) -> dict:
    """
    Monte Carlo utility for RL-facing experiments.

    Returns expected reward plus release/landing samples for one fixed action.
    """
    env = DartThrowingOneStepEnv_SPEC(config=config)
    rewards = []
    scores = []
    landings = []
    release_states = []
    hits = []
    obs, _ = env.reset(seed=seed)
    del obs
    for idx in range(n_rollouts):
        if idx > 0:
            env.reset()
        _, reward, terminated, truncated, info = env.step(action9_or10)
        assert terminated and not truncated
        rewards.append(float(reward))
        scores.append(float(info["score"]))
        hits.append(bool(info["hit"]))
        if info["release_state6"] is not None:
            release_states.append(info["release_state6"])
        landings.append(info["landing_m"])
    rewards = np.asarray(rewards, dtype=float)
    scores = np.asarray(scores, dtype=float)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_score": float(np.mean(scores)),
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "release_states6": np.asarray(release_states, dtype=float),
        "landings_m": np.asarray(landings, dtype=float),
        "scores": scores,
    }
