"""
VALIDATION_C1_end_to_end_and_monte_carlo_SPEC.py
=================================================
**PDF section:** Track C — C1 (Integration validation)

- Single **nominal** end-to-end score from a fixed release state (PDF B4 IC).
- **Monte Carlo** with Gaussian noise on release velocity (simple stand-in for torque + t_r noise).

Run: `python track_C_integration_SPEC/VALIDATION_C1_end_to_end_and_monte_carlo_SPEC.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import MC_DEFAULT_N_ROLLOUTS, VALIDATION_RELEASE_XYZ_M
from dart_robot_spec.SPEC_QUICK_REFERENCE_constants import (
    VALIDATION_INITIAL_VX_MPS,
    VALIDATION_INITIAL_VY_MPS,
    VALIDATION_INITIAL_VZ_MPS,
)
from track_C_integration_SPEC.C1_pipeline_arm_release_to_projectile_score_SPEC import (
    monte_carlo_expected_score_SPEC,
    score_from_release_state_SPEC,
)


def nominal_end_to_end_SPEC():
    s6 = np.array(
        [
            VALIDATION_RELEASE_XYZ_M[0],
            VALIDATION_RELEASE_XYZ_M[1],
            VALIDATION_RELEASE_XYZ_M[2],
            VALIDATION_INITIAL_VX_MPS,
            VALIDATION_INITIAL_VY_MPS,
            VALIDATION_INITIAL_VZ_MPS,
        ],
        dtype=float,
    )
    out = score_from_release_state_SPEC(s6)
    print("--- C1 nominal end-to-end ---")
    print(f"  score={out['score']}, hit={out['hit']}, dy={out['delta_y_m']*1000:.2f} mm, dz={out['delta_z_m']*1000:.2f} mm")


def mc_noise_SPEC(n=200, seed=0):
    base = np.array(
        [
            VALIDATION_RELEASE_XYZ_M[0],
            VALIDATION_RELEASE_XYZ_M[1],
            VALIDATION_RELEASE_XYZ_M[2],
            VALIDATION_INITIAL_VX_MPS,
            VALIDATION_INITIAL_VY_MPS,
            VALIDATION_INITIAL_VZ_MPS,
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)

    def sampler(_i, r):
        noise = np.array([0, 0, 0, r.normal(0, 0.05), r.normal(0, 0.02), r.normal(0, 0.02)])
        return base + noise

    out = monte_carlo_expected_score_SPEC(sampler, n_rollouts=n, seed=seed)
    print("--- C1 Monte Carlo (release velocity noise stand-in) ---")
    print(f"  N={n}, mean score={out['mean_score']:.3f}, std(score)={out['std_score']:.3f}")
    print(f"  landing cov (m^2):\n{out['covariance_deltay_deltaz']}")


if __name__ == "__main__":
    nominal_end_to_end_SPEC()
    mc_noise_SPEC(n=min(MC_DEFAULT_N_ROLLOUTS, 120), seed=1)
