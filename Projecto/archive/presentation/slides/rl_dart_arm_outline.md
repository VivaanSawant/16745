# RL for Dart-Throwing Arm — Slide outline

Use this in Google Slides, PowerPoint, or Marp. One section = one slide.

---

## Slide 1: Title
**Reinforcement Learning for the Dart-Throwing Arm**
Using RL to learn how the arm should move to maximize score under uncertainty.

---

## Slide 2: The problem
- Robot arm throws a dart at a board.
- Small changes (release timing, torque, wind) → different landing.
- Goal: a **policy** that gets **high expected score** even with noise, not one perfect throw.

---

## Slide 3: Why RL to “figure out the arm”?
- Arm + release + flight are **complex and stochastic**.
- Closed-form control is hard.
- RL learns **state → action** by trial and error in simulation.
- The learned policy **is** our encoding of “how the arm should work.”

---

## Slide 4: High-level RL loop
**State → Policy (arm) → Action → Physics sim → Landing → Reward**
- State: joint angles, velocities, (optional) wind.
- Action: joint torques, release timing.
- Sim: our 3D projectile + dartboard.
- Reward: dartboard score (+ optional penalties).

---

## Slide 5: State space
- Arm: joint angles θ, angular velocities θ̇.
- Optional: wind estimate, dart orientation.
- From state we get release position & velocity → then physics → landing.

---

## Slide 6: Action space
- Continuous: e.g. wrist torque, other joint torques.
- Release timing: when to release.
- Policy (e.g. neural network) outputs these given state.

---

## Slide 7: Reward
- Primary: **dartboard score** at landing.
- Optional: penalize unstable release, large torques, variance.
- Can optimize **expected score** (Monte Carlo) for robustness.

---

## Slide 8: How we “figure out how the arm works”
- We don’t hand-design torques/timings.
- RL (SAC, TD3, PPO) learns a policy: state → action.
- The arm’s behavior is **discovered** by the algorithm using our physics + dartboard.

---

## Slide 9: Training pipeline
1. Build RL environment: step(state, action) → arm to release → projectile → landing → score.
2. Train policy with standard RL (e.g. Stable-Baselines3).
3. Optional curriculum: increase wind/release noise over time.
4. Evaluate with Monte Carlo: many rollouts → landing distribution, expected score.

---

## Slide 10: What we already have
- 3D physics, dartboard, “hit any number,” variance/Monte Carlo, state estimator.
- **Next:** wrap in RL env and train policy.

---

## Slide 11: Summary
- RL learns a **policy**: state → torques + release timing.
- Our **simulator** gives: action → release → trajectory → landing → score.
- Maximizing expected score under noise = robust “recipe” for how the arm works.
