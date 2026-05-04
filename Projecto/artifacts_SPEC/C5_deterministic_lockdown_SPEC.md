# C5 deterministic lockdown

- matrix pass rate: `1.000`
- phase pass: `True`

## Core deterministic checks

- FK position inf error (m): `0.000000`
- FK velocity inf error (m/s): `0.000000`
- Golden release relative drift: `0.000000`
- Golden final-q relative drift: `0.000001`
- Golden max saturation ratio: `0.729`

## Environment matrix

| xml | use_feedforward | release_time_s | release_speed_mps | release_vx_mps | pass | reason |
|---|---:|---:|---:|---:|---:|---|
| standard | 0 | 0.076181 | 6.732 | 5.965 | 1 | ok |
| standard | 0 | 0.100000 | 5.175 | 5.175 | 1 | ok |
| standard | 1 | 0.076181 | 5.598 | 5.539 | 1 | ok |
| standard | 1 | 0.100000 | 5.260 | 4.958 | 1 | ok |
| tall | 0 | 0.076181 | 6.732 | 5.965 | 1 | ok |
| tall | 0 | 0.100000 | 5.175 | 5.175 | 1 | ok |
| tall | 1 | 0.076181 | 5.598 | 5.540 | 1 | ok |
| tall | 1 | 0.100000 | 5.256 | 4.954 | 1 | ok |
