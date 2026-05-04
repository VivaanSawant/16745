# Spin Relevance Study (SPEC)

This folder contains an **axial-spin-first** ablation to determine whether
adding spin (`omega_ax`) is necessary for the dart project at current
operating speeds, and whether release-side contact features can manipulate it.

## What this study does

1. Builds a nominal release state from the current tuned arm throw.
2. Re-simulates flight with drag + optional Magnus-like term:
   - `a_drag = -k_drag * |v_rel| * v_rel`
   - `a_spin = c_lift * (omega x v_rel)`
3. Sweeps **axial spin** magnitude (`omega_ax`) for a chosen `c_lift`.
4. Adds an off-axis disturbance appendix (sensitivity bound only).
5. Compares each spin setting to no-spin baseline using the same rollout
   perturbations.
6. Runs contact-to-spin proxy experiments:
   - wrist/torque perturbation
   - release-time perturbation
7. Generates:
   - axial relevance plot
   - contact manipulation plot
   - summaries with recommendation + confidence intervals

## Decision criterion

Spin is flagged as relevant if any condition is met:

- RMS landing shift exceeds the configured threshold (default `15 mm`), or
- absolute score delta exceeds the configured threshold (default `2 points`).
- radial miss-distance delta exceeds the configured threshold (default `12 mm`).

Primary modeling assumption:
- axial spin is the dominant controllable component; off-axis components are
  treated as disturbance in this project phase.

## Run

```bash
python spin_relevance_SPEC/spin_ablation_SPEC.py
```

## Outputs

Files are written to `artifacts_SPEC/`:

- `SPIN_relevance_plot_SPEC.png`
- `SPIN_relevance_summary_SPEC.md`
- `SPIN_contact_manipulation_SPEC.png`
- `SPIN_contact_manipulation_summary_SPEC.md`
- `C7_spin_contact_confidence_SPEC.md` (multi-seed CI99 confidence gate summary)
- `C7_spin_contact_confidence_SPEC.json` (machine-readable gate metrics)

## Confidence gate relationship

The spin package is a sub-component of the pre-RL confidence program:

- C7 validates spin relevance and contact manipulability confidence.
- C8 checks cross-scenario operational stability before RL starts.
- C9 aggregates final go/no-go status for RL start.

## Current state

- Latest integrated run reports `C7=True`, `C8=True`, and `RL start gate=True` in C9.
- Spin is currently retained as an axial-first production approximation for RL-stage experiments.
- Wrist-mediated contact shaping remains the preferred control channel; release-time shaping is available but should be lower-authority.

## Paper notes

- Include the axial-spin-first modeling rationale (tractability vs fidelity).
- Include paired no-spin vs spin deltas and contact-proxy slope findings as evidence of controlled relevance.
- Explicitly state that off-axis spin is treated as disturbance in this phase, not as a primary policy output.
