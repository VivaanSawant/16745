# Dart Throwing Physics Simulator

Implements the physics and dartboard logic for the RL dart-throwing project. Covers the **(Vivaan)** items: 3D physics, variance/trajectory randomness, hitting any target number, and a simple state estimator.

## Layout

- **`physics/dartboard.py`** – Dartboard geometry (DRA dimensions in m). `score_at(x,y)`, `get_target_center(segment, ring)`. Segment 1–20, rings S/D/T, bull.
- **`physics/projectile.py`** – 3D point-mass dart: gravity, drag, wind. `throw_3d(pos0, vel0, wind=...)`, `landing_position(...)`.
- **`physics/target_solver.py`** – **Hit any number**: `aim_point("T20")`, `parse_target("S19")`, `find_velocity_to_hit(pos0, "D5", ...)` to get release velocity and landing.
- **`physics/variance.py`** – **Variance / RL of variance**: `sample_release_noise(vel0, vel_std=..., time_std=...)`, `monte_carlo_landings(...)` for landing distribution and expected score.
- **`physics/state_estimator.py`** – **State estimator** at release: `estimate_release_from_arm(joint_angles, joint_velocities, link_lengths)` or `estimate_release_direct(measured_pos, measured_vel)`.

## Run

```bash
pip install -r requirements.txt
python run_sim.py
```

This runs: one 3D throw, aim at T20/S19/D5/BULL, Monte Carlo variance demo, and state estimator; then saves `physics_demo.png`.

## Coordinate convention

- Board: origin at bull, **x** right, **y** up (board plane).
- World: **z** up; board at `z = 0`; throw from e.g. `(0, 0, 1.7)` m.

## Next steps (from proposal)

- Plug this into an RL env (state/action/reward using `landing_position`, `score_at`, `monte_carlo_landings`).
- Quaternions / spin can be added later for dart orientation if needed.
- MuJoCo arm + this projectile for full sim.
