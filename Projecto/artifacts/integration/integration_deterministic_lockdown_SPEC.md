# Integration deterministic lockdown

- matrix pass rate: `1.000`
- phase pass: `True`

## Core deterministic checks

- FK position inf error (m): `0.000000`
- FK velocity inf error (m/s): `0.000000`
- Golden release relative drift: `0.000000`
- Golden final-q relative drift: `0.000002`
- Golden max saturation ratio: `0.714`

## Environment matrix

| xml | use_feedforward | release_time_s | release_speed_mps | release_vx_mps | pass | reason |
|---|---:|---:|---:|---:|---:|---|
| standard | 0 | 0.076181 | 6.752 | 5.964 | 1 | ok |
| standard | 0 | 0.100000 | 5.174 | 5.160 | 1 | ok |
| standard | 1 | 0.076181 | 5.520 | 5.448 | 1 | ok |
| standard | 1 | 0.100000 | 5.194 | 4.885 | 1 | ok |
| tall | 0 | 0.076181 | 6.752 | 5.964 | 1 | ok |
| tall | 0 | 0.100000 | 5.174 | 5.160 | 1 | ok |
| tall | 1 | 0.076181 | 5.520 | 5.448 | 1 | ok |
| tall | 1 | 0.100000 | 5.194 | 4.885 | 1 | ok |
