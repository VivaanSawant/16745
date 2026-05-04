"""
integration_rl_env_scaffold_SPEC.py
=======================
Minimal RL-facing scaffold for the dart-throwing project.

This module does **not** implement training. It provides a lightweight environment-style
API around the current MuJoCo **4-DOF** arm → release → projectile → score pipeline so future
SAC/TD3/PPO code has a stable interface to target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
    THROW_DURATION_S,
)
from dartrobot.motion.link_geometry import joint_limits_rad_SPEC
from dartrobot.motion.controller import (
    DEFAULT_MUJOCO_PD_KD_SPEC,
    DEFAULT_MUJOCO_PD_KP_SPEC,
    DEFAULT_MUJOCO_THROW_KNOTS_SPEC,
    plan_nominal_throw_knots_min_jerk_SPEC,
    simulate_throw_mujoco_SPEC,
)
from dartrobot.integration.release_to_score import (
    score_from_release_state_SPEC,
)
from dartrobot.integration.jacobian_covariance import (
    landing_info_for_release_SPEC,
)


def default_reset_observation_SPEC(wind_xyz=(0.0, 0.0, 0.0), include_wind: bool = True) -> np.ndarray:
    """Observation at reset: [q_yaw, q_sh, q_el, q_wr, qd×4, wind?]."""
    q0 = np.radians(
        [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
    )
    obs = [*q0.tolist(), 0.0, 0.0, 0.0, 0.0]
    if include_wind:
        obs.extend(float(w) for w in wind_xyz)
    return np.asarray(obs, dtype=float)


def knots_action_bounds_SPEC() -> tuple[np.ndarray, np.ndarray]:
    """
    Box bounds for the **12** spline-knot action vector.

    Each knot for a given joint is bounded by that joint's limit (yaw, shoulder, elbow, wrist).
    """
    lim = joint_limits_rad_SPEC()
    low = np.concatenate([np.full(3, lim[j][0], dtype=float) for j in range(4)])
    high = np.concatenate([np.full(3, lim[j][1], dtype=float) for j in range(4)])
    return low, high


def clip_knots_action_SPEC(knots12: np.ndarray) -> np.ndarray:
    """Project a candidate 12D action into joint-limit bounds."""
    low, high = knots_action_bounds_SPEC()
    return np.clip(np.asarray(knots12, dtype=float).reshape(12), low, high)


@dataclass
class DartThrowEnvConfig_SPEC:
    """
    Configuration for the one-step throw environment.

    `action_includes_release_time=True` makes the action dimension **13**, with the final
    scalar interpreted directly as release time in seconds.
    """

    wind_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drag_enabled: bool = True
    torque_noise: bool = True
    torque_noise_sigma_add: float = 0.5
    torque_noise_sigma_mult: float = 0.02
    include_wind_in_obs: bool = True
    action_includes_release_time: bool = False
    kp: float = DEFAULT_MUJOCO_PD_KP_SPEC
    kd: float = DEFAULT_MUJOCO_PD_KD_SPEC
    reward_torque_l2_coeff: float = 0.0
    reward_speed_tracking_coeff: float = 0.0
    reward_landing_l2_coeff: float = 0.0
    target_release_speed_mps: float = 5.5
    use_residual_action: bool = False
    residual_action_scale: float = 1.0
    use_minimum_jerk_warm_start: bool = False
    use_feedforward_controller: bool = False
    inertia_ff_diag: tuple[float, float, float, float] | None = None
    reward_release_weighted_l2_coeff: float = 0.0
    # C2 sensitivity-informed default ordering: vz > vx > z > x > vy > y.
    release_sensitivity_weights6: tuple[float, float, float, float, float, float] = (
        0.25, 0.10, 0.40, 0.80, 0.15, 1.00
    )
    target_release_state6: tuple[float, float, float, float, float, float] | None = None
    # Optional iLQR / OC warm start (12 knots). When set, overrides the default throw preset
    # unless `use_minimum_jerk_warm_start` builds from the same `q_start4` seed.
    warm_start_knots12: tuple[float, ...] | None = None


def nominal_action_warm_start_SPEC(
    q_start4: np.ndarray,
    use_minimum_jerk: bool,
) -> np.ndarray:
    """Return nominal knot action used to warm-start RL."""
    if not use_minimum_jerk:
        return DEFAULT_MUJOCO_THROW_KNOTS_SPEC.copy()
    out = plan_nominal_throw_knots_min_jerk_SPEC(q_start4=np.asarray(q_start4, dtype=float).reshape(4))
    return out["knots12"].copy()


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
        self._q_start4 = np.radians(
            [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
        )
        self.default_action = nominal_action_warm_start_SPEC(
            q_start4=self._q_start4,
            use_minimum_jerk=self.config.use_minimum_jerk_warm_start,
        )
        if self.config.warm_start_knots12 is not None:
            self.default_action = clip_knots_action_SPEC(np.asarray(self.config.warm_start_knots12, dtype=float))

    @property
    def observation_dim(self) -> int:
        return int(self._reset_obs.size)

    @property
    def action_dim(self) -> int:
        return 13 if self.config.action_includes_release_time else 12

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
            "residual_action_mode": bool(self.config.use_residual_action),
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
            raw_knots = action[:12]
            raw_release_time = float(action[12])
        else:
            raw_knots = action[:12]
            raw_release_time = np.nan

        if self.config.use_residual_action:
            knots12 = clip_knots_action_SPEC(self.default_action + self.config.residual_action_scale * raw_knots)
        else:
            knots12 = clip_knots_action_SPEC(raw_knots)

        if self.config.action_includes_release_time:
            if self.config.use_residual_action:
                base_release = 0.85 * THROW_DURATION_S
                release_time_s = float(
                    np.clip(
                        base_release + self.config.residual_action_scale * raw_release_time,
                        0.05 * THROW_DURATION_S,
                        0.99 * THROW_DURATION_S,
                    )
                )
            else:
                release_time_s = float(np.clip(raw_release_time, 0.05 * THROW_DURATION_S, 0.99 * THROW_DURATION_S))
        else:
            release_time_s = None

        sim = simulate_throw_mujoco_SPEC(
            knots12,
            q_start4=self._q_start4,
            release_time_s=release_time_s,
            rng=self._rng,
            torque_noise=self.config.torque_noise,
            torque_noise_sigma_add=self.config.torque_noise_sigma_add,
            torque_noise_sigma_mult=self.config.torque_noise_sigma_mult,
            kp=self.config.kp,
            kd=self.config.kd,
            use_feedforward=self.config.use_feedforward_controller,
            inertia_ff_diag=None
            if self.config.inertia_ff_diag is None
            else np.asarray(self.config.inertia_ff_diag, dtype=float),
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
        landing_l2_error_m = np.nan
        if self.config.reward_landing_l2_coeff != 0.0:
            if rollout["hit"]:
                landing_l2_error_m = float(np.linalg.norm(landing["landing_m"]))
            else:
                landing_l2_error_m = 1.0
            reward -= float(self.config.reward_landing_l2_coeff * landing_l2_error_m)
        release_weighted_error = np.nan
        if self.config.reward_release_weighted_l2_coeff != 0.0 and self.config.target_release_state6 is not None:
            target = np.asarray(self.config.target_release_state6, dtype=float).reshape(6)
            weights = np.asarray(self.config.release_sensitivity_weights6, dtype=float).reshape(6)
            diff = s6 - target
            release_weighted_error = float(np.sqrt(np.sum(weights * diff * diff)))
            reward -= float(self.config.reward_release_weighted_l2_coeff * release_weighted_error)

        info = {
            "score": float(rollout["score"]),
            "hit": bool(rollout["hit"]),
            "delta_y_m": float(rollout["delta_y_m"]) if rollout["hit"] else np.nan,
            "delta_z_m": float(rollout["delta_z_m"]) if rollout["hit"] else np.nan,
            "release_state6": s6.copy(),
            "landing_m": landing["landing_m"].copy(),
            "landing_l2_error_m": landing_l2_error_m,
            "release_weighted_error": release_weighted_error,
            "release_time_s": float(sim["release_time_s"]),
            "knots12_used": knots12.copy(),
            "default_action_knots12": self.default_action.copy(),
            "terminal_observation": terminal_obs.copy(),
        }
        return terminal_obs, reward, True, False, info


def evaluate_action_mc_SPEC(
    action12_or13: np.ndarray,
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
        _, reward, terminated, truncated, info = env.step(action12_or13)
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
