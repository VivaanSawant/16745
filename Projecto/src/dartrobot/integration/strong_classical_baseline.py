"""
integration_strong_classical_baseline_SPEC.py
=============================================
**Strong classical baseline:** per-target joint optimization of **12 spline knots** and
**release_time_s** over a small grid, cached for reuse by Monte Carlo and RL warm starts.

See ``dartrobot baseline`` to populate ``artifacts/baseline/strong_baseline_cache.pkl``.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def strong_baseline_cache_key_SPEC(dy_m: float, dz_m: float, decimals: int = 6) -> str:
    return f"{round(float(dy_m), decimals)}_{round(float(dz_m), decimals)}"


_STRONG_CACHE_LOADED: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def _invalidate_strong_cache(path: Path) -> None:
    _STRONG_CACHE_LOADED.pop(str(Path(path).resolve()), None)


@dataclass
class BaselineSolution_SPEC:
    knots12: np.ndarray
    release_time_s: float
    r_mm: float
    delta_y_m: float
    delta_z_m: float
    plane_hit: bool
    face_hit: bool
    score: int
    fun: float
    n_restarts_used: int
    release_time_grid: tuple[float, ...]
    nfev_total: int

    def to_cache_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["knots12"] = np.asarray(self.knots12, dtype=float).reshape(12).tolist()
        d["release_time_grid"] = tuple(float(x) for x in self.release_time_grid)
        return d

    @staticmethod
    def from_cache_dict(d: dict[str, Any]) -> BaselineSolution_SPEC:
        g = d.get("release_time_grid", ())
        if isinstance(g, list):
            g = tuple(float(x) for x in g)
        return BaselineSolution_SPEC(
            knots12=np.asarray(d["knots12"], dtype=float).reshape(12),
            release_time_s=float(d["release_time_s"]),
            r_mm=float(d["r_mm"]),
            delta_y_m=float(d["delta_y_m"]),
            delta_z_m=float(d["delta_z_m"]),
            plane_hit=bool(d["plane_hit"]),
            face_hit=bool(d["face_hit"]),
            score=int(d["score"]),
            fun=float(d["fun"]),
            n_restarts_used=int(d["n_restarts_used"]),
            release_time_grid=tuple(float(x) for x in g),
            nfev_total=int(d.get("nfev_total", 0)),
        )


def load_strong_baseline_cache_SPEC(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    key = str(path.resolve())
    try:
        mtime = path.stat().st_mtime if path.exists() else -1.0
    except OSError:
        mtime = -1.0
    hit = _STRONG_CACHE_LOADED.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]

    if not path.exists():
        data: dict[str, dict[str, Any]] = {}
    else:
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            if not isinstance(raw, dict):
                data = {}
            else:
                data = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, EOFError, pickle.UnpicklingError):
            data = {}
    _STRONG_CACHE_LOADED[key] = (mtime, data)
    return data


def save_strong_baseline_cache_SPEC(path: Path, entire_cache: dict[str, dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(entire_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    _invalidate_strong_cache(path)


def save_strong_baseline_entry_SPEC(path: Path, dy_m: float, dz_m: float, soln: BaselineSolution_SPEC) -> None:
    key = strong_baseline_cache_key_SPEC(dy_m, dz_m)
    cache = dict(load_strong_baseline_cache_SPEC(path))
    cache[key] = soln.to_cache_dict()
    save_strong_baseline_cache_SPEC(path, cache)


def lookup_strong_baseline_SPEC(
    path: Path,
    dy_m: float,
    dz_m: float,
) -> tuple[np.ndarray, float, dict[str, Any]] | None:
    """
    Return ``(knots12, release_time_s, meta)`` if ``path`` contains a row for the rounded key;
    otherwise ``None``.
    """
    key = strong_baseline_cache_key_SPEC(dy_m, dz_m)
    row = load_strong_baseline_cache_SPEC(path).get(key)
    if row is None:
        return None
    try:
        sol = BaselineSolution_SPEC.from_cache_dict(row)
    except (KeyError, TypeError, ValueError):
        return None
    meta = {
        "r_mm": sol.r_mm,
        "fun": sol.fun,
        "plane_hit": sol.plane_hit,
        "face_hit": sol.face_hit,
        "score": sol.score,
        "release_time_grid": sol.release_time_grid,
        "nfev_total": sol.nfev_total,
    }
    return sol.knots12.copy(), float(sol.release_time_s), meta


def solve_strong_baseline_for_target_SPEC(
    q_start4: np.ndarray,
    dy_m: float,
    dz_m: float,
    *,
    release_time_grid: tuple[float, ...] = (0.05, 0.075, 0.10, 0.125, 0.15),
    maxiter: int = 80,
    n_restarts: int = 6,
    restart_seed: int = 0,
    drag_enabled: bool = True,
    wind_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BaselineSolution_SPEC:
    """
    iLQR seed knots, then for each ``release_time_grid`` value run ``optimize_knots_for_board_target_SPEC``;
    pick deterministic solution with smallest ``r_mm`` (tiebreak: lower ``fun``).
    """
    from dartrobot.integration.ilqr_motion_targeting import (
        solve_ilqr_motion_knots_for_board_deltas_SPEC,
    )
    from dartrobot.integration.knot_landing_optimizer import (
        evaluate_knots_landing_diagnostics_SPEC,
        optimize_knots_for_board_target_SPEC,
    )
    from dartrobot.integration.rl_env_scaffold import clip_knots_action_SPEC
    from dartrobot.motion.controller import (
        DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
        DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
    )

    q0 = np.asarray(q_start4, dtype=float).reshape(4)
    ilqr = solve_ilqr_motion_knots_for_board_deltas_SPEC(q0, float(dy_m), float(dz_m))
    k_seed = clip_knots_action_SPEC(np.asarray(ilqr["knots12"], dtype=float).reshape(12))

    best_soln: BaselineSolution_SPEC | None = None
    best_r = float("inf")
    best_fun = float("inf")
    nfev_total = 0
    grid_t = tuple(float(x) for x in release_time_grid)

    for gi, rt in enumerate(grid_t):
        mj_kw: dict[str, Any] = {
            "kp": DEFAULT_MUJOCO_OVERHAND_PD_KP_SPEC,
            "kd": DEFAULT_MUJOCO_OVERHAND_PD_KD_SPEC,
            "release_time_s": rt,
        }
        opt = optimize_knots_for_board_target_SPEC(
            k_seed.copy(),
            float(dy_m),
            float(dz_m),
            q0,
            maxiter=int(maxiter),
            n_restarts=int(n_restarts),
            restart_seed=int(restart_seed) + int(gi) * 17,
            wind_xyz=wind_xyz,
            drag_enabled=drag_enabled,
            mujoco_kwargs=mj_kw,
        )
        nfev_total += int(opt.get("nfev", 0))
        kopt = clip_knots_action_SPEC(np.asarray(opt["knots12"], dtype=float).reshape(12))
        diag = evaluate_knots_landing_diagnostics_SPEC(
            kopt, q0, float(dy_m), float(dz_m), wind_xyz=wind_xyz, drag_enabled=drag_enabled, mujoco_kwargs=mj_kw
        )
        r_mm = float(diag.get("r_mm", float("nan")))
        if not np.isfinite(r_mm):
            r_mm = float("inf")
        fun = float(opt.get("fun", float("inf")))
        if r_mm < best_r - 1e-9 or (abs(r_mm - best_r) < 1e-9 and fun < best_fun):
            best_r = r_mm
            best_fun = fun
            r_out = float(diag["r_mm"]) if np.isfinite(float(diag.get("r_mm", float("nan")))) else float("nan")
            best_soln = BaselineSolution_SPEC(
                knots12=kopt.copy(),
                release_time_s=float(rt),
                r_mm=r_out,
                delta_y_m=float(diag["delta_y_m"]),
                delta_z_m=float(diag["delta_z_m"]),
                plane_hit=bool(diag["plane_hit"]),
                face_hit=bool(diag["dartboard_face_hit"]),
                score=int(diag.get("score", 0)),
                fun=fun,
                n_restarts_used=int(opt.get("n_restarts_used", n_restarts)),
                release_time_grid=grid_t,
                nfev_total=nfev_total,
            )

    if best_soln is None:
        raise RuntimeError("solve_strong_baseline_for_target_SPEC: no grid point produced a solution")
    return BaselineSolution_SPEC(
        knots12=best_soln.knots12,
        release_time_s=best_soln.release_time_s,
        r_mm=best_soln.r_mm,
        delta_y_m=best_soln.delta_y_m,
        delta_z_m=best_soln.delta_z_m,
        plane_hit=best_soln.plane_hit,
        face_hit=best_soln.face_hit,
        score=best_soln.score,
        fun=best_soln.fun,
        n_restarts_used=best_soln.n_restarts_used,
        release_time_grid=grid_t,
        nfev_total=nfev_total,
    )
