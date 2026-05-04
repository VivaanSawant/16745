# Spin relevance summary (SPEC, axial-first)

- c_lift used: `0.006`
- rollouts per axial setting: `120`
- rollouts per proxy perturbation point: `160`
- max RMS landing shift (axial sweep): `124.34 mm`
- max absolute score delta (axial sweep): `5.708`
- max absolute radial miss delta (axial sweep): `61.61 mm`
- shift relevance threshold: `15.0 mm`
- score relevance threshold: `2.0`
- radial miss threshold: `12.0 mm`

## Recommendation: SPIN RELEVANT

## Axial sweep (primary)

| omega_axial_rad_s | delta_score_vs_no_spin | delta_hit_rate | landing_shift_rms_mm | delta_radial_miss_mm |
|---:|---:|---:|---:|---:|
| 0.0 | 0.000 | 0.000 | 0.00 | 0.00 |
| 20.0 | -1.700 | 0.000 | 31.02 | 5.63 |
| 40.0 | -2.858 | 0.000 | 62.06 | 18.77 |
| 60.0 | -3.517 | 0.000 | 93.16 | 38.19 |
| 80.0 | -5.708 | 0.000 | 124.34 | 61.61 |

## Off-axis disturbance appendix

| off_axis_noise_rad_s | delta_score_vs_no_spin | landing_shift_rms_mm |
|---:|---:|---:|
| 0.0 | 0.000 | 0.00 |
| 2.0 | -0.650 | 9.41 |
| 5.0 | -0.125 | 28.32 |
| 10.0 | -0.675 | 45.40 |

## Contact-to-spin proxy diagnostics

- wrist perturbation slope d(omega_ax)/d(wrist_delta_deg): `0.221` (95% CI: `0.221` to `0.221`)
- release-time perturbation slope d(omega_ax)/d(time_offset_s): `425.011` (95% CI: `-291.466` to `983.529`)
- wrist->radial coupling slope d(delta_radial_mm)/d(wrist_delta_deg): `-0.027` (95% CI: `-0.131` to `0.052`)
- release-time->radial coupling slope d(delta_radial_mm)/d(time_offset_s): `3.857` (95% CI: `-238.628` to `254.086`)
- sign consistency criterion: CI not crossing zero indicates stable directional manipulation.
