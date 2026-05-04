# C9 pre-RL handoff

## Phase status

- C5 deterministic lockdown: `True`
- C6 uncertainty calibration 99CI: `True`
- C7 spin contact confidence: `True`
- C8 stress campaign gate: `True`
- RL start gate (all phases pass): `True`

## Locked assumptions

- axial-spin-first model is the production spin model
- feedforward+PD and direct-PD are both kept as valid controller baselines
- tall-arm MJCF is primary for pre-RL studies

## Recommended RL randomization bounds

- torque_noise_sigma_add: `0.4` to `0.6`
- torque_noise_sigma_mult: `0.015` to `0.025`
- wind_x_mps: `-0.25` to `0.25`
- wind_y_mps: `-0.15` to `0.15`
- release_time_jitter_s: `-0.008` to `0.008`
- spin_c_lift: `0.004` to `0.008`
- axial_spin_rad_s: `0.0` to `40.0`

## Residual risks

- release-time proxy channel may be weak in some seeds; keep lower control authority than wrist channel
- covariance linearization remains local; monitor off-nominal drift during RL domain randomization
- stress gate margins should be re-evaluated after any controller retune
