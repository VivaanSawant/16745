# C4 accuracy brownie pack

## Projectile parameter sensitivity

| parameter | multiplier | hit | score | delta_y_mm | delta_z_mm |
|---|---:|---:|---:|---:|---:|
| cd | 0.85 | 1 | 60.0 | 0.0 | 101.3 |
| cd | 1.00 | 1 | 60.0 | 0.0 | 99.7 |
| cd | 1.15 | 1 | 20.0 | 0.0 | 98.2 |
| rho | 0.90 | 1 | 60.0 | 0.0 | 100.8 |
| rho | 1.00 | 1 | 60.0 | 0.0 | 99.7 |
| rho | 1.10 | 1 | 20.0 | 0.0 | 98.7 |
| mass | 0.90 | 1 | 20.0 | 0.0 | 98.6 |
| mass | 1.00 | 1 | 60.0 | 0.0 | 99.7 |
| mass | 1.10 | 1 | 60.0 | 0.0 | 100.7 |

## Torque noise calibration candidate

- target tr(Sigma_release): `0.0022`
- best sigma_add: `0.500`
- best sigma_mult: `0.020`
- achieved tr(Sigma_release): `0.0023`

## Risk-objective ablation

| risk_lambda | mean_score_mc | std_score_mc | hit_rate_mc |
|---:|---:|---:|---:|
| 0.00 | 7.400 | 8.405 | 1.000 |
| 0.25 | 11.250 | 11.200 | 1.000 |
| 0.75 | 11.000 | 11.788 | 1.000 |
