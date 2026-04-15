# Dart-throwing robot project (CMU 16-745 style)

This repo has two layers: **SPEC code** (matches `dart_robot_tasklist.pdf`) and **legacy prototypes** under `physics/`. Prefer the SPEC tree for anything graded or demoed against the PDF.

---

## What each file does

### Root

| File | Purpose |
|------|---------|
| `README.md` | This file: file map + completion status. |
| `README_PROJECT_SPEC.md` | Short pointer to folders and how to run the SPEC demo. |
| `README_physics.md` | Describes the **older** `physics/` package (different coordinates than the PDF). |
| `requirements.txt` | Python dependencies (`numpy`, `scipy`, `matplotlib`; `mujoco` optional for XML). |
| `run_dart_robot_demo_SPEC.py` | Runs the three validation scripts below in order (B4, A4, C1). |
| `run_sim.py` | **Legacy** demo for old `physics/` modules (board at `z=0` frame). |

### `dart_robot_spec/` (constants + PDF index)

| File | Purpose |
|------|---------|
| `README_TRACK_MAPPING_CMU16745.md` | Maps **PDF section labels** (A1, B2, …) to filenames. |
| `SPEC_QUICK_REFERENCE_constants.py` | Single copy of distances, masses, limits, drag params, segment order, etc. from the PDF. |
| `__init__.py` | Package marker. |

### `track_A_arm_SPEC/` (Track A — 3-link arm)

| File | Purpose |
|------|---------|
| `A1_link_geometry_and_inertia_SPEC.py` | Documents link lengths/masses/limits; helpers for inertia formula and joint clamping. |
| `A2_mujoco_mjcf_3link_arm_SPEC.xml` | MuJoCo model: fixed shoulder mount, 3 hinges, capsules, 3 motors, `release_site`, cocked keyframe. |
| `A3_cubic_spline_and_pd_controller_SPEC.py` | 9 knot parameters → desired joint trajectory; PD torques; optional torque noise; FK for fingertip; **simplified** joint integration (stand-in until MuJoCo steps the arm). |
| `VALIDATION_A4_workspace_release_speed_SPEC.py` | Coarse joint grid → max fingertip `x,z`; one sample throw → reports release speed. |
| `__init__.py` | Package marker. |

### `track_B_projectile_SPEC/` (Track B — point-mass dart)

| File | Purpose |
|------|---------|
| `B1_point_mass_force_model_gravity_drag_wind_SPEC.py` | Gravity + quadratic drag using `v_rel = v - v_wind`; toggle drag for experiments. |
| `B2_ode_integrator_event_board_plane_SPEC.py` | `solve_ivp` until dart crosses **x = 2.37 m** forward; outputs **Δy, Δz** vs bull at `z = 1.73 m`. |
| `B3_dartboard_scoring_radial_angular_SPEC.py` | Ring + wedge scoring from **(Δy, Δz)** in meters (converts to mm internally). |
| `VALIDATION_B4_drag_comparison_and_plots_SPEC.py` | PDF spot-check coordinates; drag on vs off; saves PNGs under `artifacts_SPEC/`. |
| `__init__.py` | Package marker. |

### `track_C_integration_SPEC/` (Track C — wiring + analysis)

| File | Purpose |
|------|---------|
| `C1_pipeline_arm_release_to_projectile_score_SPEC.py` | Takes 6D release state → B2 → B3 → score; Monte Carlo helper with a user-supplied release sampler. |
| `C2_jacobian_covariance_optimizer_SPEC.py` | Finite-difference **2×6** Jacobian of landing vs release; `Σ_land ≈ J Σ Jᵀ`; Nelder–Mead **stub** for “minimize negative MC score”. |
| `VALIDATION_C1_end_to_end_and_monte_carlo_SPEC.py` | One nominal score + small MC with simple velocity noise (not full arm torque noise yet). |
| `__init__.py` | Package marker. |

### `artifacts_SPEC/` (generated, may be empty until you run validations)

| Content | Purpose |
|---------|---------|
| `B4_side_view_xz_SPEC.png`, `B4_board_view_deltas_SPEC.png` | Written by `VALIDATION_B4_drag_comparison_and_plots_SPEC.py`. |

### `physics/` (**legacy** — not PDF world frame)

| File | Purpose |
|------|---------|
| `README_LEGACY_DEPRECATED.md` | Explains why this folder is not the SPEC coordinate system. |
| `dartboard.py` | Old scoring on a board in the **x–y** plane at **z = 0**. |
| `projectile.py` | Old integration until **z = 0** (different from PDF **x = 2.37** crossing). |
| `variance.py` | Monte Carlo landings in the legacy frame. |
| `target_solver.py` | Aim strings like `T20` in legacy frame. |
| `state_estimator.py` | Simple arm → release estimate for prototypes. |
| `__init__.py` | Re-exports legacy modules. |

### `slides/` (presentation, not required by PDF)

| File | Purpose |
|------|---------|
| `rl_dart_arm_slides.html` | Browser slideshow (Reveal.js) explaining RL + arm idea. |
| `rl_dart_arm_outline.md` | Same content as bullet outline for Slides/PowerPoint. |

---

## What has been completed

Legend: **Done** = implemented and exercised in code/scripts. **Partial** = started or simplified. **Not done** = not in repo or only stubbed.

### PDF Track A (3-link arm)

| Item | Status | Notes |
|------|--------|--------|
| A1 Link geometry, limits, masses, shoulder position | **Done** | In constants + `A1_…py` + MJCF. |
| A2 MJCF + motors + keyframe | **Done** | `A2_mujoco_mjcf_3link_arm_SPEC.xml`. Loading in MuJoCo locally is **your** check (`mujoco` optional). |
| A3 Cubic spline (9 params) + PD + noise hooks + release state | **Partial** | Spline + PD + FK + **simplified** diagonal “fake” inertia integration, not `mj_step` on the XML. |
| A4 Workspace, 5–7 m/s release, stick-figure / MuJoCo render | **Partial** | Grid workspace + one release speed printed; **no** MuJoCo stick-figure render script. |

### PDF Track B (projectile + board)

| Item | Status | Notes |
|------|--------|--------|
| B1 Gravity + drag + wind in `v_rel` | **Done** | `B1_…py`. |
| B2 `solve_ivp` + event at board plane | **Done** | `B2_…py`; handles “never hits” as miss. |
| B3 Rings + wedges + score 0–60 | **Done** | `B3_…py`; PDF spot checks pass in B4 script. |
| B4 Drag compare, plots, spot checks | **Done** | `VALIDATION_B4_…py` + `artifacts_SPEC/`. |

### PDF Track C (integration + analysis)

| Item | Status | Notes |
|------|--------|--------|
| C1 Wire release → projectile → score; MC | **Partial** | Wiring **Done**. MC in validation uses **velocity noise only**; PDF also wants torque noise + release-time jitter from the **arm** sim. |
| C2 Jacobian + predicted covariance + optimizer | **Partial** | Jacobian + `Σ_land` formula **Done**. Nelder–Mead wrapper **stub** only (you pass `mc_objective_fn`). **Not done:** full study comparing ellipse to MC scatter, warm-started target optimization. |

### Other project goals (from your earlier proposal)

| Item | Status | Notes |
|------|--------|--------|
| RL environment (Gymnasium / SB3) | **Not done** | Slides describe it; no training code in repo. |
| MuJoCo arm + projectile in one loop | **Not done** | XML exists; Python still uses analytic B2 for flight after release. |

---

## Quick commands

```bash
pip install -r requirements.txt
python run_dart_robot_demo_SPEC.py
```

For the PDF-to-file index only, open `dart_robot_spec/README_TRACK_MAPPING_CMU16745.md`.
