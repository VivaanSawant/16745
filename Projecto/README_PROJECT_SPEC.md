# Dart-throwing robot (CMU 16-745) — SPEC implementation

This repository implements **`dart_robot_tasklist.pdf`**: 3-DOF planar arm + point-mass dart, MuJoCo-oriented layout, scipy projectile, and integration/analysis hooks.

## Where everything lives

| Path | Role |
|------|------|
| `dart_robot_spec/README_TRACK_MAPPING_CMU16745.md` | **Start here:** PDF section → file mapping |
| `dart_robot_spec/SPEC_QUICK_REFERENCE_constants.py` | All numeric constants from the PDF table + A1 |
| `track_A_arm_SPEC/` | Track A: geometry (A1), MJCF (A2), spline+PD (A3), validation (A4) |
| `track_B_projectile_SPEC/` | Track B: forces (B1), ODE+event (B2), scoring (B3), validation (B4) |
| `track_C_integration_SPEC/` | Track C: pipeline (C1), Jacobian+optimizer stub (C2), validation |
| `run_dart_robot_demo_SPEC.py` | Runs B4 + A4 + C1 validation scripts in sequence |
| `artifacts_SPEC/` | Generated plots (B4), created when you run validations |
| `physics/` | **Legacy** pre-spec code; see `physics/README_LEGACY_DEPRECATED.md` |

## Run

```bash
python run_dart_robot_demo_SPEC.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`. Optional: `mujoco` to load `A2_mujoco_mjcf_3link_arm_SPEC.xml`.

## Naming convention

Files that implement the PDF end with **`_SPEC`** so you can search the codebase for `SPEC` and find spec-driven modules only.
