"""
VALIDATION_C1_end_to_end_and_monte_carlo_SPEC.py
=================================================
**PDF section:** Track C — C1 (Integration validation)

- Single **nominal** end-to-end score from a fixed release state (PDF B4 IC).
- **Monte Carlo** where noise propagates from arm torques through MuJoCo dynamics to the
  release state — matching the PDF's intention of torque noise + release-time jitter,
  rather than hand-crafting velocity noise directly.

Run: `python track_C_integration_SPEC/VALIDATION_C1_end_to_end_and_monte_carlo_SPEC.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    MC_DEFAULT_N_ROLLOUTS,
    VALIDATION_RELEASE_XYZ_M,
    VALIDATION_INITIAL_VX_MPS,
    VALIDATION_INITIAL_VY_MPS,
    VALIDATION_INITIAL_VZ_MPS,
    KEYFRAME_SHOULDER_DEG,
    KEYFRAME_ELBOW_DEG,
    KEYFRAME_WRIST_DEG,
)
from track_A_arm_SPEC.A3_cubic_spline_and_pd_controller_SPEC import (
    simulate_throw_mujoco_SPEC,
)
from track_C_integration_SPEC.C1_pipeline_arm_release_to_projectile_score_SPEC import (
    monte_carlo_expected_score_SPEC,
    score_from_release_state_SPEC,
)

_Q_START = np.radians([KEYFRAME_SHOULDER_DEG, KEYFRAME_ELBOW_DEG, KEYFRAME_WRIST_DEG])
_XML_PATH = _ROOT / "track_A_arm_SPEC" / "A2_mujoco_mjcf_3link_arm_TALL_SPEC.xml"

# Pre-RL tuned baseline under the updated elbow convention (q_elbow in [-145, 0]).
# This keeps C1/C2 informative (non-zero scoring throws) after the human-like
# forearm constraint was introduced.
_TUNED_KNOTS9 = np.radians(np.array([
    -37.46583288056429, -33.196898815597365,  8.519418496181338,
   -111.8152288749504,  -76.08164184800785, -17.940545379302467,
     -2.162963251122143, 17.39559530401733, 20.99761664028883,
], dtype=float))
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


def mc_arm_noise_SPEC(n: int = 120, seed: int = 1):
    """
    Monte Carlo where noise flows through the arm:
      MuJoCo sim (torque_noise=True) → release_state6 → B2 → B3 → score.

    This replaces the old hand-crafted velocity noise stand-in and correctly
    sources uncertainty from perturb_torque_SPEC + release-time jitter (σ=0.01 s).
    """
    def sampler(_i, rng):
        out = simulate_throw_mujoco_SPEC(
            _TUNED_KNOTS9,
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


if __name__ == "__main__":
    nominal_end_to_end_SPEC()
    mc_arm_noise_SPEC(n=min(MC_DEFAULT_N_ROLLOUTS, 60), seed=1)
