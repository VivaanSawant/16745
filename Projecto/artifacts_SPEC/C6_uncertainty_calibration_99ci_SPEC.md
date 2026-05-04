# C6 uncertainty calibration (99% CI)

- target trace(Sigma_release): `0.002447`
- target mean score: `21.667`
- best candidate: sigma_add=`0.500`, sigma_mult=`0.020`
- relative trace error (best): `0.079`
- phase pass (trace error <= 10%): `True`

## Torque/release calibration candidates

| sigma_add | sigma_mult | trace_mean | trace_ci99_lo | trace_ci99_hi | score_mean | score_ci99_lo | score_ci99_hi | fit_error_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.300 | 0.010 | 0.000937 | 0.000828 | 0.000994 | 26.600 | 24.520 | 29.470 | 0.8449 |
| 0.400 | 0.015 | 0.001680 | 0.001480 | 0.001780 | 25.775 | 24.250 | 27.988 | 0.5032 |
| 0.500 | 0.020 | 0.002641 | 0.002322 | 0.002794 | 24.425 | 22.280 | 25.997 | 0.2281 |
| 0.600 | 0.025 | 0.003821 | 0.003351 | 0.004036 | 24.277 | 21.899 | 26.135 | 0.6816 |
| 0.700 | 0.030 | 0.005219 | 0.004566 | 0.005504 | 23.428 | 20.990 | 24.677 | 1.2271 |

## Spin-lift sensitivity range

| c_lift | delta_score_vs_no_spin | delta_radial_miss_mm | landing_shift_rms_mm |
|---:|---:|---:|---:|
| 0.002 | -14.392 | 2.362 | 20.318 |
| 0.004 | -16.875 | 8.909 | 40.643 |
| 0.006 | -7.958 | 18.733 | 60.982 |
| 0.008 | -6.833 | 31.037 | 81.342 |
| 0.010 | -10.750 | 45.199 | 101.730 |
