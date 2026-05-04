# High-Level Optimal Control and RL Strategy

## Executive Perspective

This dart-throwing system is best treated as a **structured stochastic control problem** rather than a monolithic black-box policy search.  
The repository already contains two distinct regimes:

1. **Arm dynamics and release generation** (`track_A_arm_SPEC/`): nonlinear, noisy, actuator-constrained.
2. **Flight and scoring dynamics** (`track_B_projectile_SPEC/` + `track_C_integration_SPEC/`): explicit physics with known equations and interpretable sensitivity.

The most effective architecture is therefore a **hybrid optimal-control plus reinforcement-learning stack**:
- Optimal control provides nominal trajectories, feasibility, and robustness structure.
- RL provides residual adaptation, disturbance compensation, and policy improvement under uncertainty.

This division yields a model that is both technically rigorous and practically trainable.

---

## Core Design Principle: Use the Right Abstraction Boundary

The critical interface is the **release state**:
`s6 = [x, y, z, vx, vy, vz]`.

Why this is the correct boundary:
- It is physically meaningful and low-dimensional.
- It is the exact input required by the white-box projectile model.
- It allows sensitivity analysis (`J = d(landing)/d(s6)`), covariance propagation, and robust optimization.
- It decouples high-frequency arm-control complexity from downstream ballistic reasoning.

In other words, the project should optimize **how the arm realizes robust release states**, not only maximize score via trial-and-error over raw spline parameters.

---

## Layered Control Architecture

### 1) Nominal Motion Synthesis (Optimal Control Layer)

Use smooth trajectory generation (minimum-jerk / minimum-effort) to construct a nominal throw profile that respects anatomical and actuator constraints.

Implemented support:
- minimum-jerk nominal knot construction
- optional feedforward+PD tracking
- torque-noise-aware rollout path in MuJoCo

Why this is valuable:
- reduces search space entropy for RL
- improves repeatability and kinematic plausibility
- establishes a high-quality prior policy manifold

### 2) Release-Space Robustness Optimization (Analytical Layer)

Use the known flight model to optimize robustness directly in release space:
- inverse aiming: target landing delta -> solve for release velocity
- Jacobian-based covariance projection: `Sigma_land ~= J Sigma_release J^T`
- Monte Carlo refinement for discrete scoring and nonlinearity

Why this is valuable:
- converts ballistic physics from a black box into actionable structure
- enables risk-aware objectives (mean score, variance, hit probability)
- identifies which release dimensions dominate uncertainty (e.g., `vz`, `vx`)

### 3) Residual Reinforcement Learning (Adaptation Layer)

Train RL as a **corrective policy**, not a full-throw synthesizer:
- residual actions around a nominal warm start
- dense shaping terms (torque, speed, landing distance)
- score remains the terminal objective

Why this is valuable:
- significantly improves sample efficiency
- stabilizes training under stochastic torque perturbations
- preserves interpretability while retaining adaptive capacity

---

## Why This Strategy Fits a Dart-Throwing Robot

This system has three defining characteristics:

1. **Nonlinear arm mechanics** with contact-like release timing sensitivity.
2. **Known downstream physics** after release (gravity + drag + wind).
3. **Sparse/discrete reward geometry** from dartboard scoring.

A pure RL strategy underutilizes known structure and often learns brittle, opaque behaviors.  
A pure optimal-control strategy struggles with model mismatch and stochastic disturbances.

The hybrid approach is superior because it is:
- **Model-aware** where equations are trustworthy.
- **Data-adaptive** where uncertainty and unmodeled effects dominate.
- **Hierarchical** in a way that mirrors the physical process (arm -> release -> flight -> score).

## Spin Modeling Scope

For this project phase, spin should be modeled using an **axial-spin-first** approximation:

- keep one dominant spin scalar (`omega_ax`) aligned with the dart/flight axis surrogate,
- treat off-axis spin as disturbance rather than a primary control variable,
- validate relevance using paired no-spin vs spin effect metrics (landing shift, score delta, hit-rate delta),
- promote spin into the main trajectory pipeline only if acceptance thresholds are exceeded consistently.

This scope balances physical realism with tractable control design and avoids over-parameterizing a release-state model that does not explicitly estimate full dart attitude.

---

## Practical Research Trajectory

Short-term:
- calibrate nominal motion and release distribution quality
- validate C2 sensitivity predictions against Monte Carlo scatter
- tune reward shaping coefficients for stable residual learning

Mid-term:
- add curriculum over wind and torque-noise intensity
- train SAC/TD3/PPO policies on residual or hierarchical action spaces
- compare black-box RL vs OC+RL under matched compute budgets

Long-term:
- extend to multi-target aiming policies
- include risk-sensitive objectives (CVaR, chance constraints)
- formalize robustness benchmarks for reproducible evaluation

## Official Evaluation Pipeline

For project reporting and grading consistency, treat the SPEC chain as the only authoritative pipeline:

- `track_A_arm_SPEC` (arm dynamics and release extraction)
- `track_B_projectile_SPEC` (white-box flight and scoring)
- `track_C_integration_SPEC` (robustness, optimization, and RL-facing analysis)
- `artifacts_SPEC/C5_...` through `artifacts_SPEC/C9_...` (pre-RL confidence reports and gate status)

Legacy `physics/` modules remain useful for prototypes, but they should not be mixed into quantitative claims for the SPEC-based deliverable.

---

## Pre-RL Confidence Gate Status

The formal C5-C9 gate now exists and is executable end-to-end:

- C5 deterministic lockdown: pass.
- C6 uncertainty calibration with 99% CIs: pass.
- C7 spin/contact confidence: pass.
- C8 stress campaign: pass under operational-readiness criteria.
- C9 handoff: generated with explicit `RL start gate` boolean and recommended randomization ranges.

Implication: the simulator/control architecture is now considered ready for RL experimentation; the next phase is training-policy implementation and comparative evaluation.

---

## Current Project Snapshot

- **Model status:** arm + release + projectile + scoring + axial-spin-first stack is integrated and validated.
- **Control status:** nominal + robust release analysis is in place; residual RL pathway is scaffolded.
- **Readiness status:** pre-RL handoff currently reports `RL start gate = True`.
- **Execution gap:** algorithmic RL training loop (PPO/SAC/TD3 pipelines) remains to be implemented.

## Paper framing notes

- Emphasize that C5-C9 constitutes a pre-training validation harness, not the final performance benchmark.
- Explain that C8 now encodes operational robustness (for agent readiness) rather than strict model-perfect statistical confidence.
- Report both strengths (deterministic correctness, uncertainty calibration, spin-contact evidence) and open risks (local linearization, retraining after retunes).

---

## Bottom Line

For this repository and problem formulation, the strongest control philosophy is:

**Nominal optimal control for disciplined motion generation, release-space analytical optimization for ballistic robustness, and residual RL for adaptive performance under uncertainty.**

That combination is not only performant; it is scientifically legible, extensible, and well aligned with the pedagogical objectives of an optimal control + RL project.
