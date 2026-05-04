# C3 control robustness sweep

| case | mean_score | std_score | hit_rate | tr(Sigma_release) | tr(Sigma_land_emp) | tr(Sigma_land_pred) | top_sensitivity | spin_delta_score_ax40 | spin_delta_hit_rate_ax40 | spin_shift_rms_mm_ax40 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| direct_knots_pd | 21.800 | 15.182 | 1.000 | 0.001984 | 0.000938 | 0.000932 | vz | -12.283 | 0.000 | 61.32 |
| residual_minjerk_pd_zero_residual | 0.000 | 0.000 | 1.000 | 0.001739 | 0.222266 | 0.220073 | vx | 0.000 | 0.000 | 2611.86 |
| residual_minjerk_plus_release_time | 0.000 | 0.000 | 1.000 | 0.002012 | 7.406293 | 7.063762 | vx | 0.000 | -0.517 | 10745.76 |
| direct_knots_feedforward_pd | 22.667 | 9.978 | 1.000 | 0.002090 | 0.000121 | 0.000121 | vz | -11.400 | 0.000 | 13.99 |
