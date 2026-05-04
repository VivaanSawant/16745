"""
VALIDATION_C2_jacobian_ellipse_vs_mc_SPEC.py
=============================================
**PDF section:** Track C — C2 (Jacobian covariance + optimizer validation)

Steps:
1. Run a nominal MuJoCo throw → get release state s6_nominal.
2. Run an MC sweep (torque_noise=True) → empirical Σ_release and Σ_land.
3. Compute 2×6 Jacobian J at s6_nominal via finite differences (C2).
4. Compute predicted Σ_land = J Σ_release Jᵀ (first-order propagation).
5. Plot: MC scatter of landings + predicted 95% ellipse vs empirical 95% ellipse.
6. Run Nelder-Mead optimizer on mean score with torque noise → compare before/after.

Run: `python track_C_integration_SPEC/VALIDATION_C2_jacobian_ellipse_vs_mc_SPEC.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG,
)
from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    simulate_throw_mujoco_SPEC,
)
from track_C_integration_SPEC.C1_pipeline_arm_release_to_projectile_score_SPEC import (
    score_from_release_state_SPEC,
)
from track_C_integration_SPEC.C2_jacobian_covariance_optimizer_SPEC import (
    jacobian_landing_wrt_release_SPEC,
    landing_info_for_release_SPEC,
    minimize_negative_mc_score_stub_SPEC,
    predicted_landing_covariance_SPEC,
    summarize_release_robustness_SPEC,
)

_ARTIFACTS = _ROOT / "artifacts_SPEC"
_ARTIFACTS.mkdir(exist_ok=True)

_Q_START = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
_XML_PATH = _ROOT / "track_A_arm_SPEC" / "A2_mujoco_mjcf_3link_arm_TALL_SPEC.xml"
_TUNED_KNOTS9 = np.radians(np.array([
    -37.46583288056429, -33.196898815597365,  8.519418496181338,
   -111.8152288749504,  -76.08164184800785, -17.940545379302467,
     -2.162963251122143, 17.39559530401733, 20.99761664028883,
], dtype=float))
_TUNED_KP = 318.1900570627695
_TUNED_KD = 0.6954130109241007
_TUNED_RELEASE_TIME_S = 0.0761809061781466
_CHI2_DOF2_95 = 5.991464547107979

def _covariance_ellipse(cov2x2: np.ndarray, n_std: float = 2.0, n_pts: int = 200) -> np.ndarray:
    """Return (n_pts, 2) points tracing the n_std-sigma covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov2x2)
    vals = np.maximum(vals, 0.0)  # numerical safety
    angle = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    scale = n_std * np.sqrt(vals)
    return circle @ (vecs * scale).T


def _mahalanobis_coverage_SPEC(points2: np.ndarray, mean2: np.ndarray, cov2x2: np.ndarray, threshold: float = _CHI2_DOF2_95) -> float:
    """
    Fraction of points inside covariance ellipse defined by Mahalanobis threshold.
    """
    points2 = np.asarray(points2, dtype=float).reshape(-1, 2)
    mean2 = np.asarray(mean2, dtype=float).reshape(2)
    cov2x2 = np.asarray(cov2x2, dtype=float).reshape(2, 2)
    if points2.size == 0:
        return 0.0
    cov_reg = cov2x2 + 1e-12 * np.eye(2)
    inv_cov = np.linalg.inv(cov_reg)
    centered = points2 - mean2[None, :]
    d2 = np.einsum("bi,ij,bj->b", centered, inv_cov, centered)
    return float(np.mean(d2 <= threshold))


def _relative_covariance_error_SPEC(pred2x2: np.ndarray, emp2x2: np.ndarray) -> float:
    """Relative Frobenius error ||pred-emp||_F / ||emp||_F."""
    pred2x2 = np.asarray(pred2x2, dtype=float).reshape(2, 2)
    emp2x2 = np.asarray(emp2x2, dtype=float).reshape(2, 2)
    den = float(np.linalg.norm(emp2x2, ord="fro"))
    if den < 1e-12:
        return 0.0
    return float(np.linalg.norm(pred2x2 - emp2x2, ord="fro") / den)


def _eps_sensitivity_SPEC(
    nominal_release6: np.ndarray,
    sigma_release6: np.ndarray,
    eps_values: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4),
) -> list[dict]:
    """Finite-difference epsilon sweep for predicted landing covariance."""
    rows = []
    for eps in eps_values:
        J = jacobian_landing_wrt_release_SPEC(nominal_release6, eps=eps)
        sigma_pred = predicted_landing_covariance_SPEC(J, sigma_release6)
        rows.append({
            "eps": float(eps),
            "jacobian_fro_norm": float(np.linalg.norm(J, ord="fro")),
            "sigma_pred_trace_mm2": float(np.trace(sigma_pred) * 1e6),
        })
    return rows


def run_mc_and_collect_states(
    knots9: np.ndarray,
    n: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run MC, returning:
      release_states (n, 6), landings (n, 2) in metres, scores (n,), hit_mask (n,).
    """
    rng = np.random.default_rng(seed)
    release_states = []
    landings = []
    scores = []
    hit_mask = []
    for _ in range(n):
        out = simulate_throw_mujoco_SPEC(
            knots9,
            q_start3=_Q_START.copy(),
            xml_path=_XML_PATH,
            rng=rng,
            torque_noise=True,
            release_time_s=_TUNED_RELEASE_TIME_S,
            kp=_TUNED_KP,
            kd=_TUNED_KD,
            enforce_joint_limits=True,
        )
        s6 = out["release_state6"]
        if s6 is None:
            continue
        result = score_from_release_state_SPEC(s6)
        landing_info = landing_info_for_release_SPEC(s6)
        release_states.append(s6)
        landings.append(landing_info["landing_m"])
        scores.append(result["score"])
        # hit_mask = dart landed within the scoring boundary (r ≤ 170 mm from bull).
        # landing_info["hit"] only means the dart crossed the board plane, which is
        # always True for forward throws — that filter inflates the covariance with
        # darts that miss the board by metres. Using score > 0 restricts to the local
        # regime where the Jacobian linearisation is actually valid.
        hit_mask.append(result["score"] > 0)
    return (np.array(release_states),
            np.array(landings),
            np.array(scores, dtype=float),
            np.array(hit_mask, dtype=bool))


def jacobian_comparison_SPEC(n_mc: int = 80, seed: int = 42):
    """
    Compare predicted covariance ellipse (from Jacobian) against MC scatter.
    """
    print("--- C2 Jacobian covariance study ---")

    # Nominal release state (no noise)
    nom = simulate_throw_mujoco_SPEC(
        _TUNED_KNOTS9,
        q_start3=_Q_START.copy(),
        xml_path=_XML_PATH,
        torque_noise=False,
        rng=np.random.default_rng(seed),
        release_time_s=_TUNED_RELEASE_TIME_S,
        kp=_TUNED_KP,
        kd=_TUNED_KD,
        enforce_joint_limits=True,
    )
    s6_nom = nom["release_state6"]
    nom_score = score_from_release_state_SPEC(s6_nom)
    print(f"  Nominal score: {nom_score['score']},  "
          f"landing (dy,dz) = ({nom_score['delta_y_m']*1e3:.1f}, "
          f"{nom_score['delta_z_m']*1e3:.1f}) mm")

    # MC with torque noise
    print(f"  Running {n_mc} noisy throws …")
    release_states, landings, scores, hit_mask = run_mc_and_collect_states(
        _TUNED_KNOTS9, n=n_mc, seed=seed)

    # Pass only the scoring throws to summarize so that both Σ_release and Σ_land
    # are estimated from the same local population.  Out-of-board throws have
    # extreme release velocities that inflate Σ_release and push the Jacobian
    # prediction far outside its valid linear regime.
    summary = summarize_release_robustness_SPEC(
        s6_nom,
        release_states[hit_mask],
        landings[hit_mask],
        scores[hit_mask],
        hit_mask=None,   # already filtered
    )
    n_total = len(scores)
    n_board = int(np.sum(scores > 0))
    n_plane = int(np.sum(np.all(np.isfinite(landings), axis=1)))
    print(f"  MC throws: {n_total} | reached board plane: {n_plane} "
          f"| landed in scoring area (r≤170mm): {n_board} ({n_board/n_total:.0%})")
    print(f"  Scoring hits used for covariance: {summary['n_hits']} "
          f"(hit rate={summary['hit_rate']:.2%}), mean score: {summary['mean_score']:.2f}")
    print(f"  Note: covariance computed over scoring throws only; out-of-board darts "
          f"({n_total - n_board}) are excluded — they lie in the nonlinear regime "
          f"where the Jacobian linearisation does not hold.")
    pred = summary['Sigma_land_predicted_2x2']
    emp  = summary['Sigma_land_empirical_2x2']
    hit_landings = landings[hit_mask]
    print(f"  Predicted Σ_land (mm²):\n{pred*1e6}")
    print(f"  Empirical  Σ_land (mm²):\n{emp*1e6}")
    rel_err = _relative_covariance_error_SPEC(pred, emp)
    print(f"  Relative covariance error ||Σ_pred-Σ_emp||_F / ||Σ_emp||_F = {rel_err:.3f}")

    mu_nom = np.array([nom_score["delta_y_m"], nom_score["delta_z_m"]], dtype=float)
    mu_emp = np.mean(hit_landings, axis=0) if hit_landings.size else mu_nom
    coverage_pred95 = _mahalanobis_coverage_SPEC(hit_landings, mu_nom, pred, threshold=_CHI2_DOF2_95)
    coverage_emp95 = _mahalanobis_coverage_SPEC(hit_landings, mu_emp, emp, threshold=_CHI2_DOF2_95)
    print(f"  95% ellipse coverage (predicted, centered at nominal): {coverage_pred95:.2%}")
    print(f"  95% ellipse coverage (empirical, centered at empirical mean): {coverage_emp95:.2%}")

    eps_rows = _eps_sensitivity_SPEC(s6_nom, summary["Sigma_release_6x6"])
    print("  FD epsilon sensitivity (J and predicted trace):")
    for row in eps_rows:
        print(
            f"    eps={row['eps']:.1e} | ||J||_F={row['jacobian_fro_norm']:.4f} "
            f"| trace(Σ_pred)={row['sigma_pred_trace_mm2']:.2f} mm²"
        )
    ratio = np.trace(emp) / max(np.trace(pred), 1e-12)
    print(f"  Trace ratio empirical/predicted: {ratio:.2f}x  "
          f"({'good agreement' if ratio < 3 else 'linearisation underestimates — noise is large'})")
    print("  Sensitivity ranking (largest first):")
    for row in summary["sensitivity_ranking"][:3]:
        print(f"    {row['label']}: trace contribution={row['landing_trace_contribution']*1e6:.1f} mm², "
              f"release variance={row['release_variance']:.4f}")
    print("  Uncertainty summary:")
    print(
        f"    release std (x,y,z,vx,vy,vz): "
        f"{np.sqrt(np.clip(np.diag(summary['Sigma_release_6x6']), 0.0, None))}"
    )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 6))

    # MC scatter
    ax.scatter(hit_landings[:, 0]*1e3, hit_landings[:, 1]*1e3,
               s=12, alpha=0.5, color="steelblue", label="MC landings")

    # Nominal landing
    dy0, dz0 = nom_score["delta_y_m"]*1e3, nom_score["delta_z_m"]*1e3
    ax.plot(dy0, dz0, "k+", ms=12, mew=2, label="nominal")

    # 2-sigma ellipses centred on nominal
    for cov, color, label in [
        (summary["Sigma_land_empirical_2x2"], "orange", "empirical 2σ"),
        (summary["Sigma_land_predicted_2x2"], "red",    "predicted 2σ (Jacobian)"),
    ]:
        ell = _covariance_ellipse(cov, n_std=2.0) * 1e3
        ax.plot(ell[:, 0] + dy0, ell[:, 1] + dz0, color=color, lw=2, label=label)

    ax.set_xlabel("Δy  (mm)")
    ax.set_ylabel("Δz  (mm)")
    ax.set_title("C2 — predicted vs empirical landing spread (2σ ellipses)")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4)

    out_path = _ARTIFACTS / "C2_jacobian_ellipse_vs_mc_SPEC.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return s6_nom, summary["Sigma_release_6x6"], scores


def optimizer_SPEC(n_mc_per_eval: int = 30, maxiter: int = 40, seed: int = 7):
    """
    Nelder-Mead on the 9 spline knots, maximising expected score under torque noise.
    Prints before/after mean scores.
    """
    print("--- C2 Nelder-Mead optimizer ---")

    def mc_objective(knots9):
        rng_inner = np.random.default_rng(seed)
        _, _, sc, _ = run_mc_and_collect_states(knots9, n=n_mc_per_eval, seed=rng_inner.integers(1 << 31))
        return float(np.mean(sc)) if len(sc) > 0 else 0.0

    x0 = _TUNED_KNOTS9.copy()
    baseline = mc_objective(x0)
    print(f"  Baseline mean score (default knots, N={n_mc_per_eval}): {baseline:.2f}")

    result = minimize_negative_mc_score_stub_SPEC(mc_objective, x0, maxiter=maxiter)
    after = mc_objective(result.x)
    print(f"  After optimisation (N={n_mc_per_eval}):                  {after:.2f}")
    print(f"  Nelder-Mead converged={result.success},  "
          f"func evals={result.nfev},  best -score={result.fun:.3f}")
    return result


if __name__ == "__main__":
    jacobian_comparison_SPEC(n_mc=60, seed=42)
    optimizer_SPEC(n_mc_per_eval=25, maxiter=30, seed=7)
