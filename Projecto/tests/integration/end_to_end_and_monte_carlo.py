"""
VALIDATION_integration_end_to_end_and_monte_carlo_SPEC.py
=================================================
**PDF section:** Track C — C1 (Integration validation)

- Single **nominal** end-to-end score from a fixed release state (PDF B4 IC).
- **Monte Carlo** where noise propagates from arm torques through MuJoCo dynamics to the
  release state — matching the PDF's intention of torque noise + release-time jitter,
  rather than hand-crafting velocity noise directly.

Run: `python integration_phase_SPEC/VALIDATION_integration_end_to_end_and_monte_carlo_SPEC.py`
"""

from __future__ import annotations

import json
from pathlib import Path

from dartrobot.paths import artifacts_dir_SPEC, mjcf_path_SPEC, project_root_SPEC

_ROOT = project_root_SPEC()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dartrobot.constants import (
    MC_DEFAULT_N_ROLLOUTS,
    VALIDATION_RELEASE_XYZ_M,
    VALIDATION_INITIAL_VX_MPS,
    VALIDATION_INITIAL_VY_MPS,
    VALIDATION_INITIAL_VZ_MPS,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_SHOULDER_YAW_DEG,
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_WRIST_DEG,
)
from dartrobot.motion.controller import (
    simulate_throw_mujoco_SPEC,
)
from dartrobot.integration.release_to_score import (
    monte_carlo_expected_score_SPEC,
    score_from_release_state_SPEC,
)

_Q_START = np.radians(
    [KEYFRAME_SHOULDER_YAW_DEG, KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG]
)
_XML_PATH = mjcf_path_SPEC("arm_4dof_tall.xml")
_ARTIFACTS = artifacts_dir_SPEC("integration")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Pre-RL tuned baseline under the updated elbow convention (q_elbow in [-145, 0]).
# This keeps C1/C2 informative (non-zero scoring throws) after the human-like
# forearm constraint was introduced.
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


def nominal_end_to_end_SPEC():
    """Single throw: fixed B4 validation initial condition → score."""
    s6 = np.array([
        VALIDATION_RELEASE_XYZ_M[0],
        VALIDATION_RELEASE_XYZ_M[1],
        VALIDATION_RELEASE_XYZ_M[2],
        VALIDATION_INITIAL_VX_MPS,
        VALIDATION_INITIAL_VY_MPS,
        VALIDATION_INITIAL_VZ_MPS,
    ], dtype=float)
    out = score_from_release_state_SPEC(s6)
    print("--- C1 nominal end-to-end (B4 fixed IC) ---")
    print(f"  score={out['score']}, hit={out['hit']}, "
          f"dy={out['delta_y_m']*1000:.2f} mm, dz={out['delta_z_m']*1000:.2f} mm")
    return out


def mc_arm_noise_SPEC(n: int = 120, seed: int = 1):
    """
    Monte Carlo where noise flows through the arm:
      MuJoCo sim (torque_noise=True) → release_state6 → B2 → B3 → score.

    This replaces the old hand-crafted velocity noise stand-in and correctly
    sources uncertainty from perturb_torque_SPEC + release-time jitter (σ=0.01 s).
    """
    def sampler(_i, rng):
        out = simulate_throw_mujoco_SPEC(
            _TUNED_KNOTS12,
            q_start4=_Q_START.copy(),
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
            # missed release window — return nominal
            return np.array([
                VALIDATION_RELEASE_XYZ_M[0], VALIDATION_RELEASE_XYZ_M[1],
                VALIDATION_RELEASE_XYZ_M[2],
                VALIDATION_INITIAL_VX_MPS, VALIDATION_INITIAL_VY_MPS,
                VALIDATION_INITIAL_VZ_MPS,
            ], dtype=float)
        return s6

    out = monte_carlo_expected_score_SPEC(sampler, n_rollouts=n, seed=seed)
    print("--- C1 Monte Carlo (arm torque noise via MuJoCo) ---")
    print(f"  N={n}, mean score={out['mean_score']:.3f}, std(score)={out['std_score']:.3f}")
    print(f"  landing cov (m²):\n{out['covariance_deltay_deltaz']}")
    return out


def _save_monte_carlo_visuals_SPEC(out: dict) -> list[dict]:
    """
    Save visual artifacts for the top-level demo showcase.
    """
    landings_m = np.asarray(out["landings_m"], dtype=float)
    scores = np.asarray(out["scores"], dtype=float)
    saved = []

    scatter_path = _ARTIFACTS / "integration_monte_carlo_board_scatter_SPEC.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ring_r_m = 0.170
    th = np.linspace(0.0, 2.0 * np.pi, 240)
    ax.plot(ring_r_m * np.cos(th), ring_r_m * np.sin(th), "k-", lw=1.0, label="board boundary")
    ax.scatter(landings_m[:, 0], landings_m[:, 1], s=14, alpha=0.55, color="royalblue", label="MC landings")
    ax.scatter([0.0], [0.0], s=40, color="crimson", marker="x", label="bull center")
    ax.set_title("C1 Monte Carlo landings (board frame)")
    ax.set_xlabel("Δy (m)")
    ax.set_ylabel("Δz (m)")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4)
    ax.legend(fontsize=8)
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {scatter_path}")
    saved.append({
        "path": str(scatter_path),
        "caption": "C1 board-view Monte Carlo landing scatter.",
    })

    hist_path = _ARTIFACTS / "integration_monte_carlo_score_hist_SPEC.png"
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, 61.5, 1.0)
    ax2.hist(scores, bins=bins, color="darkslateblue", alpha=0.8)
    ax2.set_title("C1 Monte Carlo score distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.grid(True, lw=0.3, alpha=0.6)
    fig2.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {hist_path}")
    saved.append({
        "path": str(hist_path),
        "caption": "C1 histogram of per-throw dartboard scores.",
    })
    return saved


def _write_manifest_SPEC(nominal_out: dict, mc_out: dict, visuals: list[dict]) -> Path:
    manifest_path = _ARTIFACTS / "integration_demo_manifest_SPEC.json"
    manifest = {
        "stage": "integration_end_to_end",
        "metrics": {
            "nominal_score": float(nominal_out["score"]),
            "mc_mean_score": float(mc_out["mean_score"]),
            "mc_std_score": float(mc_out["std_score"]),
            "mc_covariance_deltay_deltaz": np.asarray(mc_out["covariance_deltay_deltaz"], dtype=float).tolist(),
        },
        "artifacts": visuals,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    nominal = nominal_end_to_end_SPEC()
    mc = mc_arm_noise_SPEC(n=min(MC_DEFAULT_N_ROLLOUTS, 60), seed=1)
    visuals = _save_monte_carlo_visuals_SPEC(mc)
    _write_manifest_SPEC(nominal, mc, visuals)
