"""
spin_ablation_SPEC.py
=====================
Axial-spin-first study with contact-to-spin proxy experiments.

Run: `python spin_relevance_SPEC/spin_ablation_SPEC.py`
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from dartrobot.paths import artifacts_dir_SPEC, mjcf_path_SPEC

from dartrobot.constants import (
    AIR_DENSITY_KG_M3,
    BULLSEYE_CENTER_Z_M,
    CROSS_SECTION_AREA_M2,
    DART_MASS_KG,
    DRAG_COEFFICIENT_CD,
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_WRIST_DEG,
    OCHE_TO_BOARD_X_M,
)
from dartrobot.motion.controller import (
    simulate_throw_mujoco_SPEC,
)
from dartrobot.flight.scoring import (
    score_from_deltas_SPEC,
)

_ARTIFACTS = artifacts_dir_SPEC("spin")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)
_XML_PATH = mjcf_path_SPEC("arm_4dof_tall.xml")

_Q_START = np.radians(
    [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
)
_TUNED_KNOTS12 = np.radians(
    np.array(
        [
            -1.2,
            -0.6,
            0.0,
            -37.46583288056429,
            -33.196898815597365,
            8.519418496181338,
            -111.8152288749504,
            -76.08164184800785,
            -17.940545379302467,
            -2.162963251122143,
            17.39559530401733,
            20.99761664028883,
        ],
        dtype=float,
    )
)
_TUNED_KP = 318.1900570627695
_TUNED_KD = 0.6954130109241007
_TUNED_RELEASE_TIME_S = 0.0761809061781466


@dataclass
class SpinStudyConfig_SPEC:
    c_lift: float = 0.006
    spin_magnitudes_axial_rad_s: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0)
    off_axis_noise_rad_s: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)
    n_rollouts: int = 80
    seed: int = 21
    vel_noise_std_mps: float = 0.10
    shift_threshold_mm: float = 15.0
    score_threshold_pts: float = 2.0
    radial_miss_threshold_mm: float = 12.0
    proxy_eval_rollouts: int = 120
    # Proxy mapping: omega_ax = a0 + a1*|v| + a2*wrist_rate + a3*tau_wrist + a4*(tr-t_ref)
    omega_proxy_coeffs: tuple[float, float, float, float, float] = (-90.0, 18.0, 4.5, 0.25, -150.0)


def _nominal_release_state_SPEC() -> np.ndarray:
    out = simulate_throw_mujoco_SPEC(
        _TUNED_KNOTS12,
        q_start4=_Q_START.copy(),
        xml_path=_XML_PATH,
        torque_noise=False,
        rng=np.random.default_rng(1),
        release_time_s=_TUNED_RELEASE_TIME_S,
        kp=_TUNED_KP,
        kd=_TUNED_KD,
        enforce_joint_limits=True,
    )
    s6 = out["release_state6"]
    if s6 is None:
        raise RuntimeError("No release state captured for nominal throw.")
    return np.asarray(s6, dtype=float).reshape(6)


def _integrate_until_board_with_spin_SPEC(
    release_state6: np.ndarray,
    omega_xyz_rad_s: np.ndarray,
    c_lift: float,
    wind_xyz_mps=(0.0, 0.0, 0.0),
    max_time_s: float = 3.0,
) -> dict:
    """
    B2-style integration with optional Magnus-like lift term.
    """
    s0 = np.asarray(release_state6, dtype=float).reshape(6)
    omega = np.asarray(omega_xyz_rad_s, dtype=float).reshape(3)
    wind = np.asarray(wind_xyz_mps, dtype=float).reshape(3)
    k_drag = 0.5 * AIR_DENSITY_KG_M3 * DRAG_COEFFICIENT_CD * CROSS_SECTION_AREA_M2 / DART_MASS_KG

    def rhs(_t, y):
        vel = y[3:6]
        v_rel = vel - wind
        vmag = float(np.linalg.norm(v_rel))
        a_drag = -k_drag * vmag * v_rel
        a_spin = c_lift * np.cross(omega, v_rel)
        a_grav = np.array([0.0, 0.0, -9.81], dtype=float)
        acc = a_grav + a_drag + a_spin
        return np.array([vel[0], vel[1], vel[2], acc[0], acc[1], acc[2]], dtype=float)

    def event_board(_t, y):
        return y[0] - OCHE_TO_BOARD_X_M

    event_board.terminal = True
    event_board.direction = 1.0
    sol = solve_ivp(
        rhs,
        (0.0, max_time_s),
        s0,
        events=event_board,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
        method="RK45",
    )

    if not sol.t_events or len(sol.t_events[0]) == 0:
        return {"hit": False, "delta_y_m": np.nan, "delta_z_m": np.nan, "score": 0.0}
    t_hit = float(sol.t_events[0][0])
    y_hit = sol.sol(t_hit)
    if float(y_hit[3]) <= 0.0:
        return {"hit": False, "delta_y_m": np.nan, "delta_z_m": np.nan, "score": 0.0}
    dy = float(y_hit[1])
    dz = float(y_hit[2] - BULLSEYE_CENTER_Z_M)
    return {"hit": True, "delta_y_m": dy, "delta_z_m": dz, "score": float(score_from_deltas_SPEC(dy, dz))}


def _orthonormal_basis_from_axis_SPEC(axis3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis3 = np.asarray(axis3, dtype=float).reshape(3)
    ref = np.array([0.0, 0.0, 1.0], dtype=float) if abs(axis3[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = np.cross(axis3, ref)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(axis3, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    return e1, e2


def _spin_vector_from_axial_SPEC(
    release_state6: np.ndarray,
    omega_axial_rad_s: float,
    off_axis_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    vel = np.asarray(release_state6, dtype=float).reshape(6)[3:6]
    speed = float(np.linalg.norm(vel))
    axis = vel / speed if speed > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
    e1, e2 = _orthonormal_basis_from_axis_SPEC(axis)
    off1, off2 = rng.normal(0.0, off_axis_noise_std, size=2)
    return omega_axial_rad_s * axis + off1 * e1 + off2 * e2


def paired_spin_effect_metrics_SPEC(
    release_states6: np.ndarray,
    omega_axial_rad_s: float,
    c_lift: float = 0.006,
    off_axis_noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    """
    Paired no-spin vs spin evaluation for a release-state sample set.
    """
    rng = np.random.default_rng(seed)
    release_states6 = np.asarray(release_states6, dtype=float).reshape(-1, 6)
    n = release_states6.shape[0]
    base_scores = np.zeros(n, dtype=float)
    spin_scores = np.zeros(n, dtype=float)
    base_hits = np.zeros(n, dtype=bool)
    spin_hits = np.zeros(n, dtype=bool)
    base_land = np.full((n, 2), np.nan, dtype=float)
    spin_land = np.full((n, 2), np.nan, dtype=float)

    for i in range(n):
        s6 = release_states6[i]
        out0 = _integrate_until_board_with_spin_SPEC(s6, np.zeros(3), c_lift=0.0)
        omega_vec = _spin_vector_from_axial_SPEC(s6, omega_axial_rad_s, off_axis_noise_std, rng)
        out1 = _integrate_until_board_with_spin_SPEC(s6, omega_vec, c_lift=c_lift)
        base_scores[i] = out0["score"]
        spin_scores[i] = out1["score"]
        base_hits[i] = bool(out0["hit"])
        spin_hits[i] = bool(out1["hit"])
        base_land[i] = [out0["delta_y_m"], out0["delta_z_m"]]
        spin_land[i] = [out1["delta_y_m"], out1["delta_z_m"]]

    valid = base_hits & spin_hits & np.all(np.isfinite(base_land), axis=1) & np.all(np.isfinite(spin_land), axis=1)
    shift_rms_mm = float("nan")
    if np.any(valid):
        diff_mm = (spin_land[valid] - base_land[valid]) * 1e3
        shift_rms_mm = float(np.sqrt(np.mean(np.sum(diff_mm * diff_mm, axis=1))))
    base_radial_mm = np.linalg.norm(base_land, axis=1) * 1e3
    spin_radial_mm = np.linalg.norm(spin_land, axis=1) * 1e3
    if np.any(valid):
        mean_radial_no_spin = float(np.mean(base_radial_mm[valid]))
        mean_radial_spin = float(np.mean(spin_radial_mm[valid]))
    else:
        mean_radial_no_spin = float("nan")
        mean_radial_spin = float("nan")
    return {
        "n_samples": int(n),
        "mean_score_no_spin": float(np.mean(base_scores)),
        "mean_score_spin": float(np.mean(spin_scores)),
        "delta_score_vs_no_spin": float(np.mean(spin_scores) - np.mean(base_scores)),
        "hit_rate_no_spin": float(np.mean(base_hits)),
        "hit_rate_spin": float(np.mean(spin_hits)),
        "delta_hit_rate": float(np.mean(spin_hits) - np.mean(base_hits)),
        "landing_shift_rms_mm": shift_rms_mm,
        "mean_radial_miss_no_spin_mm": mean_radial_no_spin,
        "mean_radial_miss_spin_mm": mean_radial_spin,
        "delta_radial_miss_mm": float(mean_radial_spin - mean_radial_no_spin) if np.isfinite(mean_radial_spin) and np.isfinite(mean_radial_no_spin) else float("nan"),
    }


def _bootstrap_slope_ci_SPEC(x: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2:
        return 0.0, 0.0, 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    slopes = []
    n = x.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-12:
            continue
        slopes.append(float(np.polyfit(xb, yb, 1)[0]))
    if not slopes:
        return slope, slope, slope
    lo, hi = np.percentile(np.asarray(slopes, dtype=float), [2.5, 97.5])
    return slope, float(lo), float(hi)


def _extract_release_features_SPEC(sim: dict) -> dict:
    s6 = np.asarray(sim["release_state6"], dtype=float).reshape(6)
    q = np.asarray(sim["q"], dtype=float)
    tau = np.asarray(sim["tau"], dtype=float)
    times = np.asarray(sim["times"], dtype=float)
    t_r = float(sim["release_time_s"])
    idx_r = int(np.argmin(np.abs(times - t_r)))
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1e-3
    wrist_rate = 0.0
    if q.shape[0] >= 3:
        i0 = max(1, idx_r - 1)
        i1 = min(q.shape[0] - 1, idx_r + 1)
        wrist_rate = float((q[i1, 3] - q[i0 - 1, 3]) / max((i1 - (i0 - 1)) * dt, 1e-9))
    win = slice(max(0, idx_r - 10), min(tau.shape[0], idx_r + 1))
    tau_wrist_mean = float(np.mean(np.abs(tau[win, 3]))) if tau.size else 0.0
    speed = float(np.linalg.norm(s6[3:6]))
    return {
        "release_state6": s6,
        "release_speed": speed,
        "wrist_rate": wrist_rate,
        "tau_wrist_mean": tau_wrist_mean,
        "release_time_s": t_r,
    }


def _sample_release_cloud_SPEC(
    base_release6: np.ndarray,
    n_rollouts: int,
    vel_noise_std_mps: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = np.repeat(np.asarray(base_release6, dtype=float).reshape(1, 6), n_rollouts, axis=0)
    cloud[:, 3:6] += rng.normal(0.0, vel_noise_std_mps, size=(n_rollouts, 3))
    return cloud


def _omega_proxy_from_features_SPEC(features: dict, coeffs: tuple[float, float, float, float, float], t_ref: float) -> float:
    a0, a1, a2, a3, a4 = coeffs
    return float(
        a0
        + a1 * features["release_speed"]
        + a2 * features["wrist_rate"]
        + a3 * features["tau_wrist_mean"]
        + a4 * (features["release_time_s"] - t_ref)
    )


def _contact_spin_proxy_experiments_SPEC(cfg: SpinStudyConfig_SPEC) -> dict:
    wrist_deltas_deg = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=float)
    rel_time_offsets_s = np.array([-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015], dtype=float)
    t_ref = _TUNED_RELEASE_TIME_S
    rows_wrist = []
    rows_time = []

    for ddeg in wrist_deltas_deg:
        knots = _TUNED_KNOTS12.copy()
        knots[11] += np.radians(ddeg)  # last wrist knot (joint index 3)
        sim = simulate_throw_mujoco_SPEC(
            knots,
            q_start4=_Q_START.copy(),
            xml_path=_XML_PATH,
            rng=np.random.default_rng(100 + int(ddeg * 10)),
            torque_noise=False,
            release_time_s=t_ref,
            kp=_TUNED_KP,
            kd=_TUNED_KD,
            enforce_joint_limits=True,
        )
        feats = _extract_release_features_SPEC(sim)
        omega_ax = _omega_proxy_from_features_SPEC(feats, cfg.omega_proxy_coeffs, t_ref)
        release_cloud = _sample_release_cloud_SPEC(
            feats["release_state6"],
            n_rollouts=cfg.proxy_eval_rollouts,
            vel_noise_std_mps=cfg.vel_noise_std_mps,
            seed=1000 + int((ddeg + 15.0) * 10),
        )
        out_spin = paired_spin_effect_metrics_SPEC(
            release_cloud,
            omega_axial_rad_s=omega_ax,
            c_lift=cfg.c_lift,
            off_axis_noise_std=0.0,
            seed=1,
        )
        rows_wrist.append({
            "wrist_delta_deg": float(ddeg),
            "omega_axial": omega_ax,
            "score_delta": out_spin["delta_score_vs_no_spin"],
            "radial_delta_mm": out_spin["delta_radial_miss_mm"],
        })

    for dt in rel_time_offsets_s:
        sim = simulate_throw_mujoco_SPEC(
            _TUNED_KNOTS12,
            q_start4=_Q_START.copy(),
            xml_path=_XML_PATH,
            rng=np.random.default_rng(300 + int(dt * 1e4)),
            torque_noise=False,
            release_time_s=t_ref + float(dt),
            kp=_TUNED_KP,
            kd=_TUNED_KD,
            enforce_joint_limits=True,
        )
        feats = _extract_release_features_SPEC(sim)
        omega_ax = _omega_proxy_from_features_SPEC(feats, cfg.omega_proxy_coeffs, t_ref)
        release_cloud = _sample_release_cloud_SPEC(
            feats["release_state6"],
            n_rollouts=cfg.proxy_eval_rollouts,
            vel_noise_std_mps=cfg.vel_noise_std_mps,
            seed=2000 + int((dt + 0.03) * 1e4),
        )
        out_spin = paired_spin_effect_metrics_SPEC(
            release_cloud,
            omega_axial_rad_s=omega_ax,
            c_lift=cfg.c_lift,
            off_axis_noise_std=0.0,
            seed=2,
        )
        rows_time.append({
            "release_time_offset_s": float(dt),
            "omega_axial": omega_ax,
            "score_delta": out_spin["delta_score_vs_no_spin"],
            "radial_delta_mm": out_spin["delta_radial_miss_mm"],
        })

    xw = np.array([r["wrist_delta_deg"] for r in rows_wrist], dtype=float)
    yw = np.array([r["omega_axial"] for r in rows_wrist], dtype=float)
    xt = np.array([r["release_time_offset_s"] for r in rows_time], dtype=float)
    yt = np.array([r["omega_axial"] for r in rows_time], dtype=float)
    w_slope = _bootstrap_slope_ci_SPEC(xw, yw, seed=5)
    t_slope = _bootstrap_slope_ci_SPEC(xt, yt, seed=6)
    yw_rad = np.array([r["radial_delta_mm"] for r in rows_wrist], dtype=float)
    yt_rad = np.array([r["radial_delta_mm"] for r in rows_time], dtype=float)
    w_rad_slope = _bootstrap_slope_ci_SPEC(xw, yw_rad, seed=15)
    t_rad_slope = _bootstrap_slope_ci_SPEC(xt, yt_rad, seed=16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(xw, yw, marker="o")
    ax1.set_xlabel("Wrist knot delta (deg)")
    ax1.set_ylabel("Predicted axial spin (rad/s)")
    ax1.grid(True, lw=0.4)
    ax1.set_title("Wrist perturbation -> omega_ax")
    ax2.plot(xt * 1e3, yt, marker="o")
    ax2.set_xlabel("Release-time offset (ms)")
    ax2.set_ylabel("Predicted axial spin (rad/s)")
    ax2.grid(True, lw=0.4)
    ax2.set_title("Release-time perturbation -> omega_ax")
    out_png = _ARTIFACTS / "SPIN_contact_manipulation_SPEC.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

    return {
        "wrist_rows": rows_wrist,
        "time_rows": rows_time,
        "wrist_slope_ci": w_slope,
        "time_slope_ci": t_slope,
        "wrist_radial_slope_ci": w_rad_slope,
        "time_radial_slope_ci": t_rad_slope,
    }


def run_spin_relevance_study_SPEC(config: SpinStudyConfig_SPEC | None = None) -> dict:
    cfg = SpinStudyConfig_SPEC() if config is None else config
    s6_nom = _nominal_release_state_SPEC()
    rng = np.random.default_rng(cfg.seed)
    release_states6 = np.repeat(s6_nom[None, :], cfg.n_rollouts, axis=0)
    release_states6[:, 3:6] += rng.normal(0.0, cfg.vel_noise_std_mps, size=(cfg.n_rollouts, 3))

    axial_rows = []
    for w_ax in cfg.spin_magnitudes_axial_rad_s:
        metrics = paired_spin_effect_metrics_SPEC(
            release_states6,
            omega_axial_rad_s=float(w_ax),
            c_lift=cfg.c_lift,
            off_axis_noise_std=0.0,
            seed=cfg.seed + int(w_ax),
        )
        axial_rows.append({
            "omega_axial_rad_s": float(w_ax),
            **metrics,
        })

    appendix_rows = []
    for off_std in cfg.off_axis_noise_rad_s:
        metrics = paired_spin_effect_metrics_SPEC(
            release_states6,
            omega_axial_rad_s=0.0,
            c_lift=cfg.c_lift,
            off_axis_noise_std=float(off_std),
            seed=cfg.seed + 200 + int(off_std * 10),
        )
        appendix_rows.append({
            "off_axis_noise_rad_s": float(off_std),
            **metrics,
        })

    proxy = _contact_spin_proxy_experiments_SPEC(cfg)
    _plot_spin_relevance_SPEC(axial_rows, appendix_rows, cfg)
    _write_summary_SPEC(axial_rows, appendix_rows, proxy, cfg)
    print("--- Spin relevance study complete ---")
    print(f"Nominal release state: {s6_nom}")
    return {
        "nominal_release6": s6_nom,
        "axial_rows": axial_rows,
        "appendix_rows": appendix_rows,
        "proxy": proxy,
    }


def _plot_spin_relevance_SPEC(axial_rows: list[dict], appendix_rows: list[dict], cfg: SpinStudyConfig_SPEC) -> None:
    x_ax = np.array([r["omega_axial_rad_s"] for r in axial_rows], dtype=float)
    y_shift = np.array([r["landing_shift_rms_mm"] for r in axial_rows], dtype=float)
    y_score = np.array([r["delta_score_vs_no_spin"] for r in axial_rows], dtype=float)
    x_off = np.array([r["off_axis_noise_rad_s"] for r in appendix_rows], dtype=float)
    y_off = np.array([r["landing_shift_rms_mm"] for r in appendix_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax1, ax2, ax3 = axes
    ax1.plot(x_ax, y_shift, marker="o", color="#1f77b4")
    ax1.axhline(cfg.shift_threshold_mm, color="gray", ls="--", lw=1.0)
    ax1.set_xlabel("Axial spin omega_ax (rad/s)")
    ax1.set_ylabel("RMS landing shift (mm)")
    ax1.set_title("Paired effect: no-spin vs axial-spin")
    ax1.grid(True, lw=0.4)

    ax2.plot(x_ax, y_score, marker="o", color="#d62728")
    ax2.axhline(cfg.score_threshold_pts, color="gray", ls="--", lw=1.0)
    ax2.axhline(-cfg.score_threshold_pts, color="gray", ls="--", lw=1.0)
    ax2.set_xlabel("Axial spin omega_ax (rad/s)")
    ax2.set_ylabel("Delta mean score")
    ax2.set_title("Score sensitivity to axial spin")
    ax2.grid(True, lw=0.4)

    ax3.plot(x_off, y_off, marker="o", color="#2ca02c")
    ax3.set_xlabel("Off-axis noise std (rad/s)")
    ax3.set_ylabel("RMS landing shift (mm)")
    ax3.set_title("Appendix: off-axis disturbance only")
    ax3.grid(True, lw=0.4)

    fig.suptitle(f"Spin Relevance (Axial-First, c_lift={cfg.c_lift})")
    out_png = _ARTIFACTS / "SPIN_relevance_plot_SPEC.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def _write_summary_SPEC(axial_rows: list[dict], appendix_rows: list[dict], proxy: dict, cfg: SpinStudyConfig_SPEC) -> None:
    max_shift = max(float(r["landing_shift_rms_mm"]) for r in axial_rows if np.isfinite(r["landing_shift_rms_mm"]))
    max_abs_score_delta = max(abs(float(r["delta_score_vs_no_spin"])) for r in axial_rows)
    max_abs_radial_delta = max(abs(float(r["delta_radial_miss_mm"])) for r in axial_rows if np.isfinite(r["delta_radial_miss_mm"]))
    relevant = (
        (max_shift > cfg.shift_threshold_mm)
        or (max_abs_score_delta > cfg.score_threshold_pts)
        or (max_abs_radial_delta > cfg.radial_miss_threshold_mm)
    )

    w_s, w_lo, w_hi = proxy["wrist_slope_ci"]
    t_s, t_lo, t_hi = proxy["time_slope_ci"]
    wr_s, wr_lo, wr_hi = proxy["wrist_radial_slope_ci"]
    tr_s, tr_lo, tr_hi = proxy["time_radial_slope_ci"]
    lines = [
        "# Spin relevance summary (SPEC, axial-first)",
        "",
        f"- c_lift used: `{cfg.c_lift}`",
        f"- rollouts per axial setting: `{cfg.n_rollouts}`",
        f"- rollouts per proxy perturbation point: `{cfg.proxy_eval_rollouts}`",
        f"- max RMS landing shift (axial sweep): `{max_shift:.2f} mm`",
        f"- max absolute score delta (axial sweep): `{max_abs_score_delta:.3f}`",
        f"- max absolute radial miss delta (axial sweep): `{max_abs_radial_delta:.2f} mm`",
        f"- shift relevance threshold: `{cfg.shift_threshold_mm:.1f} mm`",
        f"- score relevance threshold: `{cfg.score_threshold_pts:.1f}`",
        f"- radial miss threshold: `{cfg.radial_miss_threshold_mm:.1f} mm`",
        "",
        f"## Recommendation: {'SPIN RELEVANT' if relevant else 'SPIN OPTIONAL AT CURRENT ACCURACY TARGET'}",
        "",
        "## Axial sweep (primary)",
        "",
        "| omega_axial_rad_s | delta_score_vs_no_spin | delta_hit_rate | landing_shift_rms_mm | delta_radial_miss_mm |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in axial_rows:
        lines.append(
            f"| {r['omega_axial_rad_s']:.1f} | {r['delta_score_vs_no_spin']:.3f} | "
            f"{r['delta_hit_rate']:.3f} | {r['landing_shift_rms_mm']:.2f} | {r['delta_radial_miss_mm']:.2f} |"
        )
    lines.extend([
        "",
        "## Off-axis disturbance appendix",
        "",
        "| off_axis_noise_rad_s | delta_score_vs_no_spin | landing_shift_rms_mm |",
        "|---:|---:|---:|",
    ])
    for r in appendix_rows:
        lines.append(
            f"| {r['off_axis_noise_rad_s']:.1f} | {r['delta_score_vs_no_spin']:.3f} | "
            f"{r['landing_shift_rms_mm']:.2f} |"
        )
    lines.extend([
        "",
        "## Contact-to-spin proxy diagnostics",
        "",
        f"- wrist perturbation slope d(omega_ax)/d(wrist_delta_deg): `{w_s:.3f}` (95% CI: `{w_lo:.3f}` to `{w_hi:.3f}`)",
        f"- release-time perturbation slope d(omega_ax)/d(time_offset_s): `{t_s:.3f}` (95% CI: `{t_lo:.3f}` to `{t_hi:.3f}`)",
        f"- wrist->radial coupling slope d(delta_radial_mm)/d(wrist_delta_deg): `{wr_s:.3f}` (95% CI: `{wr_lo:.3f}` to `{wr_hi:.3f}`)",
        f"- release-time->radial coupling slope d(delta_radial_mm)/d(time_offset_s): `{tr_s:.3f}` (95% CI: `{tr_lo:.3f}` to `{tr_hi:.3f}`)",
        "- sign consistency criterion: CI not crossing zero indicates stable directional manipulation.",
    ])
    out_md = _ARTIFACTS / "SPIN_relevance_summary_SPEC.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_md}")

    # Dedicated proxy report for easier inclusion in writeups.
    out_proxy_md = _ARTIFACTS / "SPIN_contact_manipulation_summary_SPEC.md"
    proxy_lines = [
        "# Spin contact manipulation summary",
        "",
        f"- wrist slope: `{w_s:.3f}` (95% CI: `{w_lo:.3f}` to `{w_hi:.3f}`)",
        f"- release-time slope: `{t_s:.3f}` (95% CI: `{t_lo:.3f}` to `{t_hi:.3f}`)",
        f"- wrist->radial slope: `{wr_s:.3f}` (95% CI: `{wr_lo:.3f}` to `{wr_hi:.3f}`)",
        f"- release-time->radial slope: `{tr_s:.3f}` (95% CI: `{tr_lo:.3f}` to `{tr_hi:.3f}`)",
        "",
        "## Wrist perturbation rows",
        "",
        "| wrist_delta_deg | omega_axial | score_delta | radial_delta_mm |",
        "|---:|---:|---:|---:|",
    ]
    for row in proxy["wrist_rows"]:
        proxy_lines.append(
            f"| {row['wrist_delta_deg']:.1f} | {row['omega_axial']:.3f} | "
            f"{row['score_delta']:.3f} | {row['radial_delta_mm']:.3f} |"
        )
    proxy_lines.extend([
        "",
        "## Release-time perturbation rows",
        "",
        "| release_time_offset_s | omega_axial | score_delta | radial_delta_mm |",
        "|---:|---:|---:|---:|",
    ])
    for row in proxy["time_rows"]:
        proxy_lines.append(
            f"| {row['release_time_offset_s']:.4f} | {row['omega_axial']:.3f} | "
            f"{row['score_delta']:.3f} | {row['radial_delta_mm']:.3f} |"
        )
    out_proxy_md.write_text("\n".join(proxy_lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_proxy_md}")


if __name__ == "__main__":
    run_spin_relevance_study_SPEC()

