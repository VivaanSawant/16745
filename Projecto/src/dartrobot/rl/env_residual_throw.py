"""
env_dart_residual_throw_SPEC.py
===============================
Gymnasium **one-step** throw: **residual** on top of a **cached classical-optimizer** warm start
(``optimize_knots_for_board_target_SPEC`` seeded from ``solve_ilqr_motion_knots_for_board_deltas_SPEC``).

**Goal-conditioned:** observation includes ``target_Δy, target_Δz``; each ``reset()`` samples a new
target from a pool (unless ``options`` pins a target). Warm knots are keyed by rounded ``(dy,dz)``
and persisted to a pickle cache (default ``artifacts/baseline/strong_baseline_cache.pkl``,
with fallback to ``policies/warm_start_cache.pkl``) by default.

Requires: ``gymnasium``, MuJoCo, project SPEC stack. See ``src/dartrobot/rl/README.md``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dartrobot.paths import artifacts_dir_SPEC, policies_dir_SPEC

from dartrobot.constants import (
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
    R_BOARD_MISS_MM,
)

# In-process merge of disk cache (Subproc workers load from disk after parent prewarm).
_MEM_THROW_CACHE: dict[str, tuple[np.ndarray, float]] = {}

STRONG_BASELINE_CACHE_DEFAULT_PATH_SPEC = artifacts_dir_SPEC("baseline", "strong_baseline_cache.pkl")
LEGACY_WARM_START_CACHE_PATH_SPEC = policies_dir_SPEC() / "warm_start_cache.pkl"


def warm_start_cache_key_SPEC(dy_m: float, dz_m: float, decimals: int = 6) -> str:
    """Stable dict key for ``(Δy, Δz)`` in metres."""
    return f"{round(float(dy_m), decimals)}_{round(float(dz_m), decimals)}"


def build_labeled_target_pool_SPEC(
    target_set: str,
    *,
    custom_presets: list[str] | None = None,
) -> list[tuple[str, float, float]]:
    """
    Return ``[(label, Δy_m, Δz_m), ...]`` in the same order as ``build_target_pool_SPEC`` rows.

    ``target_set``: ``full`` | ``trebles_bulls`` | ``singles_bulls`` | ``trebles_singles_bulls`` |
    ``trebles`` | ``singles`` | ``bulls`` | ``custom``.
    """
    from dartrobot.integration.target_score_aim import (
        board_deltas_for_ring_sector_SPEC,
        resolve_preset_SPEC,
    )

    ts = str(target_set).strip().lower()
    rows: list[tuple[str, float, float]] = []

    if ts == "custom":
        if not custom_presets:
            raise ValueError("target_set='custom' requires non-empty custom_presets")
        for raw in custom_presets:
            p = str(raw).strip()
            if not p:
                continue
            ring, sector = resolve_preset_SPEC(p)
            dy, dz = board_deltas_for_ring_sector_SPEC(ring, sector)
            rows.append((p, float(dy), float(dz)))
        if not rows:
            raise ValueError("no valid presets in custom_presets")
        return rows

    if ts == "trebles":
        for s in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("treble", s)
            rows.append((f"t{s}", float(dy), float(dz)))
        return rows

    if ts == "singles":
        for s in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("single", s)
            rows.append((f"single_{s}", float(dy), float(dz)))
        return rows

    if ts == "bulls":
        for lab, ring in (("BULL", "bull_inner"), ("SBULL", "bull_outer")):
            dy, dz = board_deltas_for_ring_sector_SPEC(ring, None)
            rows.append((lab, float(dy), float(dz)))
        return rows

    if ts == "trebles_singles_bulls":
        for s in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("treble", s)
            rows.append((f"t{s}", float(dy), float(dz)))
        for s in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("single", s)
            rows.append((f"single_{s}", float(dy), float(dz)))
        for lab, ring in (("BULL", "bull_inner"), ("SBULL", "bull_outer")):
            dy, dz = board_deltas_for_ring_sector_SPEC(ring, None)
            rows.append((lab, float(dy), float(dz)))
        return rows

    if ts == "trebles_bulls":
        for sector in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("treble", sector)
            rows.append((f"t{sector}", float(dy), float(dz)))
        for lab, ring in (("BULL", "bull_inner"), ("SBULL", "bull_outer")):
            dy, dz = board_deltas_for_ring_sector_SPEC(ring, None)
            rows.append((lab, float(dy), float(dz)))
        return rows

    if ts == "singles_bulls":
        for sector in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("single", sector)
            rows.append((f"single_{sector}", float(dy), float(dz)))
        for lab, ring in (("BULL", "bull_inner"), ("SBULL", "bull_outer")):
            dy, dz = board_deltas_for_ring_sector_SPEC(ring, None)
            rows.append((lab, float(dy), float(dz)))
        return rows

    if ts == "full":
        for sector in range(1, 21):
            dy, dz = board_deltas_for_ring_sector_SPEC("treble", sector)
            rows.append((f"t{sector}", float(dy), float(dz)))
            dy, dz = board_deltas_for_ring_sector_SPEC("single", sector)
            rows.append((f"single_{sector}", float(dy), float(dz)))
            dy, dz = board_deltas_for_ring_sector_SPEC("double", sector)
            rows.append((f"d{sector}", float(dy), float(dz)))
        for lab, ring in (("BULL", "bull_inner"), ("SBULL", "bull_outer")):
            dy, dz = board_deltas_for_ring_sector_SPEC(ring, None)
            rows.append((lab, float(dy), float(dz)))
        return rows

    raise ValueError(f"unknown or empty target_set {target_set!r}")


def build_target_pool_SPEC(
    target_set: str,
    *,
    custom_presets: list[str] | None = None,
) -> np.ndarray:
    """
    Return ``(N, 2)`` array of ``(Δy_m, Δz_m)`` targets.

    ``target_set``: ``full`` | ``trebles_bulls`` | ``singles_bulls`` | ``trebles_singles_bulls`` |
    ``trebles`` | ``singles`` | ``bulls`` | ``custom``.
    ``custom_presets``: comma-split preset strings (``t10``, ``BULL``, …) when ``target_set=custom``.
    """
    labeled = build_labeled_target_pool_SPEC(target_set, custom_presets=custom_presets)
    return np.asarray([[dy, dz] for _, dy, dz in labeled], dtype=np.float64)


def _normalize_legacy_cache_value_SPEC(v: Any) -> tuple[np.ndarray, float] | None:
    from dartrobot.motion.controller import (
        DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    )

    if isinstance(v, np.ndarray):
        return np.asarray(v, dtype=np.float64).reshape(12), float(DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC)
    if isinstance(v, dict) and "knots12" in v:
        rt = float(v.get("release_time_s", DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC))
        return np.asarray(v["knots12"], dtype=np.float64).reshape(12), rt
    return None


def _load_legacy_throw_cache_file_SPEC(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def _persist_legacy_throw_cache_SPEC(path: Path, entire: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(entire, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _persist_one_legacy_entry_SPEC(path: Path, dy_m: float, dz_m: float, knots12: np.ndarray, release_time_s: float) -> None:
    k = warm_start_cache_key_SPEC(dy_m, dz_m)
    raw = _load_legacy_throw_cache_file_SPEC(path)
    raw[k] = {
        "knots12": np.asarray(knots12, dtype=np.float64).reshape(12),
        "release_time_s": float(release_time_s),
    }
    _persist_legacy_throw_cache_SPEC(path, raw)


def eval_holdout_pool_SPEC() -> np.ndarray:
    """Small fixed pool for ``EvalCallback`` (subset of presets)."""
    from dartrobot.integration.target_score_aim import (
        board_deltas_for_ring_sector_SPEC,
        resolve_preset_SPEC,
    )

    presets = ("t20", "t10", "t1", "BULL", "d12", "S20")
    rows = []
    for p in presets:
        ring, sector = resolve_preset_SPEC(p)
        rows.append(board_deltas_for_ring_sector_SPEC(ring, sector))
    return np.asarray(rows, dtype=np.float64)


def _default_mujoco_kw_overhand_SPEC() -> dict[str, Any]:
    from dartrobot.motion.controller import (
        DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    )

    return {
        "kp": DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
        "kd": DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        "release_time_s": DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
    }


def get_cached_warm_throw_SPEC(
    dy_m: float,
    dz_m: float,
    q_start4: np.ndarray,
    cache_path_primary: Path,
    *,
    cache_path_legacy: Path | None = None,
    opt_n_restarts: int = 2,
    opt_maxiter: int = 60,
    opt_restart_seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Return ``(knots12, release_time_s)`` for ``(dy_m, dz_m)``.

    Reads **strong baseline** pickle at ``cache_path_primary`` first, then ``cache_path_legacy``
    (defaults to ``LEGACY_WARM_START_CACHE_PATH_SPEC``). On miss, runs iLQR → knot optimizer at
    default overhand release time and persists to the **legacy** pickle only.
    """
    global _MEM_THROW_CACHE
    from dartrobot.integration.strong_classical_baseline import (
        lookup_strong_baseline_SPEC,
    )

    key = warm_start_cache_key_SPEC(dy_m, dz_m)
    legacy = Path(cache_path_legacy) if cache_path_legacy is not None else LEGACY_WARM_START_CACHE_PATH_SPEC
    primary = Path(cache_path_primary)

    if key in _MEM_THROW_CACHE:
        k12, rt = _MEM_THROW_CACHE[key]
        return k12.copy(), float(rt)

    hit = lookup_strong_baseline_SPEC(primary, float(dy_m), float(dz_m))
    if hit is not None:
        k12, rt, _meta = hit
        _MEM_THROW_CACHE[key] = (k12.copy(), float(rt))
        if verbose:
            print(f"[warm_cache] strong_baseline key={key} release_time_s={rt:.4g}")
        return k12.copy(), float(rt)

    leg_raw = _load_legacy_throw_cache_file_SPEC(legacy)
    lv = leg_raw.get(key)
    if lv is not None:
        parsed = _normalize_legacy_cache_value_SPEC(lv)
        if parsed is not None:
            k12, rt = parsed
            _MEM_THROW_CACHE[key] = (k12.copy(), float(rt))
            if verbose:
                print(f"[warm_cache] legacy_disk key={key} release_time_s={rt:.4g}")
            return k12.copy(), float(rt)

    from dartrobot.integration.ilqr_motion_targeting import (
        solve_ilqr_motion_knots_for_board_deltas_SPEC,
    )
    from dartrobot.integration.knot_landing_optimizer import (
        optimize_knots_for_board_target_SPEC,
    )
    from dartrobot.integration.rl_env_scaffold import clip_knots_action_SPEC

    q0 = np.asarray(q_start4, dtype=float).reshape(4)
    mj_kw = _default_mujoco_kw_overhand_SPEC()
    ilqr = solve_ilqr_motion_knots_for_board_deltas_SPEC(q0, float(dy_m), float(dz_m))
    k0 = clip_knots_action_SPEC(np.asarray(ilqr["knots12"], dtype=float).reshape(12))
    opt = optimize_knots_for_board_target_SPEC(
        k0,
        float(dy_m),
        float(dz_m),
        q0,
        maxiter=int(opt_maxiter),
        n_restarts=int(opt_n_restarts),
        restart_seed=int(opt_restart_seed),
        mujoco_kwargs=mj_kw,
    )
    warm12 = clip_knots_action_SPEC(np.asarray(opt["knots12"], dtype=float).reshape(12))
    rt0 = float(mj_kw["release_time_s"])
    _MEM_THROW_CACHE[key] = (warm12.copy(), rt0)
    _persist_one_legacy_entry_SPEC(legacy, float(dy_m), float(dz_m), warm12, rt0)
    if verbose:
        print(f"[warm_cache] computed {key} fun={opt.get('fun'):.6g} success={opt.get('success')}")
    return warm12.copy(), rt0


def get_cached_warm_knots12_SPEC(
    dy_m: float,
    dz_m: float,
    q_start4: np.ndarray,
    cache_path: Path,
    *,
    opt_n_restarts: int = 2,
    opt_maxiter: int = 60,
    opt_restart_seed: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """Backward-compatible: return only ``knots12`` (default release time may differ from strong cache)."""
    k12, _rt = get_cached_warm_throw_SPEC(
        dy_m,
        dz_m,
        q_start4,
        cache_path,
        cache_path_legacy=LEGACY_WARM_START_CACHE_PATH_SPEC,
        opt_n_restarts=opt_n_restarts,
        opt_maxiter=opt_maxiter,
        opt_restart_seed=opt_restart_seed,
        verbose=verbose,
    )
    return k12


def ensure_warm_knots_cached_for_pool_SPEC(
    target_pool: np.ndarray,
    cache_path_primary: Path,
    *,
    cache_path_legacy: Path | None = None,
    opt_n_restarts: int = 2,
    opt_maxiter: int = 60,
    opt_restart_seed: int = 0,
    verbose: bool = False,
) -> None:
    """Pre-compute warm starts for every row in ``target_pool`` (sequential; run in main process)."""
    q0 = np.radians(
        [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
    )
    pool = np.asarray(target_pool, dtype=float).reshape(-1, 2)
    leg = Path(cache_path_legacy) if cache_path_legacy is not None else LEGACY_WARM_START_CACHE_PATH_SPEC
    for i in range(pool.shape[0]):
        dy, dz = float(pool[i, 0]), float(pool[i, 1])
        get_cached_warm_throw_SPEC(
            dy,
            dz,
            q0,
            Path(cache_path_primary),
            cache_path_legacy=leg,
            opt_n_restarts=opt_n_restarts,
            opt_maxiter=opt_maxiter,
            opt_restart_seed=opt_restart_seed + i,
            verbose=verbose,
        )


class DartResidualThrowEnv_SPEC(gym.Env):
    """
    Observation: ``[q(4), qdot(4), target_Δy, target_Δz]`` (10,).

    Action: ``[-1,1]^12`` scaled into a **bounded residual** added to warm-start knots, then clipped.

    ``reset(*, options={"target_dy_m": float, "target_dz_m": float})`` pins the target for that
    episode (used by eval); otherwise a row is sampled uniformly from ``target_pool``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_pool: np.ndarray,
        *,
        residual_span_frac: float = 0.18,
        torque_noise: bool = True,
        torque_noise_sigma_add: float = 0.5,
        torque_noise_sigma_mult: float = 0.02,
        torque_noise_warmup_steps: int = 0,
        seed: int | None = None,
        warm_start_cache_path: Path | str | None = None,
        opt_n_restarts: int = 2,
        opt_maxiter: int = 60,
        opt_restart_seed: int = 0,
    ):
        super().__init__()
        pool = np.asarray(target_pool, dtype=np.float64).reshape(-1, 2)
        if pool.shape[0] < 1:
            raise ValueError("target_pool must have at least one (dy, dz) row")
        self._target_pool = pool.copy()
        self.residual_span_frac = float(residual_span_frac)
        self.torque_noise_sigma_add = float(torque_noise_sigma_add)
        self.torque_noise_sigma_mult = float(torque_noise_sigma_mult)
        self._torque_noise_target = bool(torque_noise)
        self.torque_noise_warmup_steps = int(max(0, torque_noise_warmup_steps))
        if self.torque_noise_warmup_steps > 0 and self._torque_noise_target:
            self.torque_noise_runtime = False
        else:
            self.torque_noise_runtime = bool(torque_noise)

        self.opt_n_restarts = int(opt_n_restarts)
        self.opt_maxiter = int(opt_maxiter)
        self.opt_restart_seed = int(opt_restart_seed)

        if warm_start_cache_path is None:
            self.warm_start_cache_path = STRONG_BASELINE_CACHE_DEFAULT_PATH_SPEC
        else:
            self.warm_start_cache_path = Path(warm_start_cache_path)

        self._rng = np.random.default_rng(seed)
        self._q_start4 = np.radians(
            [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
        )
        self.target_dy = float(self._target_pool[0, 0])
        self.target_dz = float(self._target_pool[0, 1])
        self._warm12: np.ndarray | None = None
        self._release_time_s: float | None = None
        self._terminated = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def set_torque_noise_runtime(self, enabled: bool) -> None:
        """VecEnv ``env_method`` hook: enable/disable torque noise mid-training (warmup)."""
        self.torque_noise_runtime = bool(enabled)

    def _build_observation_SPEC(self) -> np.ndarray:
        q = self._q_start4.astype(np.float64)
        qd = np.zeros(4, dtype=np.float64)
        return np.concatenate([q, qd, [self.target_dy, self.target_dz]]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        opts = options or {}
        if "target_dy_m" in opts and "target_dz_m" in opts:
            self.target_dy = float(opts["target_dy_m"])
            self.target_dz = float(opts["target_dz_m"])
        else:
            idx = int(self._rng.integers(0, self._target_pool.shape[0]))
            self.target_dy = float(self._target_pool[idx, 0])
            self.target_dz = float(self._target_pool[idx, 1])

        self._warm12, self._release_time_s = get_cached_warm_throw_SPEC(
            self.target_dy,
            self.target_dz,
            self._q_start4,
            self.warm_start_cache_path,
            cache_path_legacy=LEGACY_WARM_START_CACHE_PATH_SPEC,
            opt_n_restarts=self.opt_n_restarts,
            opt_maxiter=self.opt_maxiter,
            opt_restart_seed=self.opt_restart_seed,
            verbose=False,
        )
        self._terminated = False
        return self._build_observation_SPEC(), {
            "warm_knots12": self._warm12.copy(),
            "release_time_s": float(self._release_time_s),
            "target_dy_m": self.target_dy,
            "target_dz_m": self.target_dz,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("Call reset() before step().")
        if self._warm12 is None:
            raise RuntimeError("Environment not reset.")

        from dartrobot.integration.release_to_score import (
            evaluate_knots12_throw_once_SPEC,
        )
        from dartrobot.integration.rl_env_scaffold import (
            clip_knots_action_SPEC,
            knots_action_bounds_SPEC,
        )
        from dartrobot.motion.controller import (
            DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC,
        )

        a = np.asarray(action, dtype=np.float64).reshape(12)
        a = np.clip(a, -1.0, 1.0)
        low, high = knots_action_bounds_SPEC()
        span = (high - low) * float(self.residual_span_frac)
        knots12 = clip_knots_action_SPEC(self._warm12 + a * span)

        rt = float(self._release_time_s) if self._release_time_s is not None else float(
            DEFAULT_MUJOCO_OVERHAND_RELEASE_TIME_S_SPEC
        )
        out = evaluate_knots12_throw_once_SPEC(
            knots12,
            q_start4=self._q_start4,
            rng=self._rng,
            torque_noise=self.torque_noise_runtime,
            torque_noise_sigma_add=self.torque_noise_sigma_add,
            torque_noise_sigma_mult=self.torque_noise_sigma_mult,
            kp=DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            kd=DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            release_time_s=rt,
        )

        self._terminated = True
        reward = float(self._reward_from_rollout_SPEC(out, self.target_dy, self.target_dz))
        info = {
            "score": float(out.get("score", 0.0)),
            "plane_hit": bool(out.get("hit")),
            "delta_y_m": float(out.get("delta_y_m", float("nan"))),
            "delta_z_m": float(out.get("delta_z_m", float("nan"))),
            "knots12": knots12.copy(),
        }
        if out.get("hit") and np.isfinite(out.get("delta_y_m")) and np.isfinite(out.get("delta_z_m")):
            dy = float(out["delta_y_m"])
            dz = float(out["delta_z_m"])
            r_mm = float(np.hypot(dy * 1000.0, dz * 1000.0))
            info["r_mm"] = r_mm
            info["dartboard_face_hit"] = bool(r_mm <= float(R_BOARD_MISS_MM))
        else:
            info["r_mm"] = float("nan")
            info["dartboard_face_hit"] = False

        return self._build_observation_SPEC(), reward, True, False, info

    @staticmethod
    def _reward_from_rollout_SPEC(out: dict, target_dy: float, target_dz: float) -> float:
        if not bool(out.get("hit")):
            return -5.0
        dy = float(out.get("delta_y_m", np.nan))
        dz = float(out.get("delta_z_m", np.nan))
        if not (np.isfinite(dy) and np.isfinite(dz)):
            return -5.0
        r_mm = float(np.hypot(dy * 1000.0, dz * 1000.0))
        if r_mm <= float(R_BOARD_MISS_MM):
            return 2.0 + 0.02 * float(out.get("score", 0.0))
        off = max(0.0, r_mm - float(R_BOARD_MISS_MM))
        err_t = float(np.hypot(dy - float(target_dy), dz - float(target_dz)))
        return -0.002 * off - 3.0 * err_t
