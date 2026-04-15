# Dart-Throwing Robot — Spec file map (CMU 16-745 task list PDF)

This folder and sibling `track_*` folders implement **`dart_robot_tasklist.pdf`** (Lawrence et al. 2003 reduced arm + point-mass dart). Every module name includes **`SPEC`** so you can grep for “what is the spec code?”

## How to read this repo

| PDF section | What it is | Primary files |
|-------------|------------|----------------|
| **Design rationale** (intro) | Why 3-DOF planar arm | Comments in `SPEC_QUICK_REFERENCE_constants.py`, `A2_mujoco_mjcf_3link_arm_SPEC.xml` |
| **Track A — A1** | Link geometry, masses, joint limits, shoulder mount | `track_A_arm_SPEC/A1_link_geometry_and_inertia_SPEC.py` |
| **Track A — A2** | MuJoCo MJCF, motors, keyframe | `track_A_arm_SPEC/A2_mujoco_mjcf_3link_arm_SPEC.xml` |
| **Track A — A3** | Cubic spline (9 params), PD torque, release, noise | `track_A_arm_SPEC/A3_cubic_spline_and_pd_controller_SPEC.py` |
| **Track A — A4** | Workspace, release speed 5–7 m/s, stick figure | `track_A_arm_SPEC/VALIDATION_A4_workspace_release_speed_SPEC.py` |
| **Track B — B1** | Point mass, gravity, quadratic drag, wind | `track_B_projectile_SPEC/B1_point_mass_force_model_gravity_drag_wind_SPEC.py` |
| **Track B — B2** | 6D ODE, `solve_ivp`, event `x = 2.37` | `track_B_projectile_SPEC/B2_ode_integrator_event_board_plane_SPEC.py` |
| **Track B — B3** | Rings + wedges, score 0–60 | `track_B_projectile_SPEC/B3_dartboard_scoring_radial_angular_SPEC.py` |
| **Track B — B4** | Drag on/off, plots, spot-check coordinates | `track_B_projectile_SPEC/VALIDATION_B4_drag_comparison_and_plots_SPEC.py` |
| **Track C — C1** | Arm release → projectile → score, Monte Carlo | `track_C_integration_SPEC/C1_pipeline_arm_release_to_projectile_score_SPEC.py` |
| **Track C — C2** | Jacobian, Σ_land, optimizer | `track_C_integration_SPEC/C2_jacobian_covariance_optimizer_SPEC.py` |
| **End-to-end demo** | Runnable script | `run_dart_robot_demo_SPEC.py` |

## Coordinate convention (PDF)

- **World:** `x` = toward board, `y` = horizontal left, `z` = up.
- **Shoulder mount:** `(0, 0, 1.50)` m.
- **Board plane:** `x = 2.37` m; **bullseye center** `(2.37, 0, 1.73)` m.
- **Landing (for scoring):** `Δy = y_impact`, `Δz = z_impact - 1.73` (meters); scoring code converts to **mm** for ring radii.

## Legacy folder `physics/`

Older prototype used board in the `z=0` plane. **Spec-compliant code lives under `track_*_SPEC/`**; `physics/` is kept for backward compatibility only (see `physics/README_LEGACY_DEPRECATED.md`).
