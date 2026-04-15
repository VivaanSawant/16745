# Legacy `physics/` package (deprecated)

This folder predates **`dart_robot_tasklist.pdf`** (CMU 16-745 spec).

**Spec-compliant code** lives under:

- `dart_robot_spec/` — constants + PDF-to-file mapping README
- `track_A_arm_SPEC/` — 3-link arm (A1–A4)
- `track_B_projectile_SPEC/` — point-mass dart + scoring (B1–B4)
- `track_C_integration_SPEC/` — pipeline + Jacobian / optimizer stubs (C1–C2)

**Legacy differences:** old modules used a board in the `z=0` plane with `(x,y)` scoring.
The PDF uses **world (x toward board, y left, z up)** with the board at **`x = 2.37` m**
and scoring in **`(Δy, Δz)`** relative to the bull at **`z = 1.73` m**.

Use **`python run_dart_robot_demo_SPEC.py`** as the current entry point.
