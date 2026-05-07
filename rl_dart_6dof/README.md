# RL Dart 6-DOF (actor–critic stub)

Self-contained PyTorch project: a **6-joint** arm plus a **continuous release** dimension is trained with a **DDPG-style** actor–critic (deterministic policy + Q with target network and soft updates).

## Setup

From this folder:

```bash
cd rl_dart_6dof
pip install torch numpy scikit-learn matplotlib
```

(Use the [official PyTorch install](https://pytorch.org/get-started/locally/) if you need CUDA.)

## Train

```bash
python train.py
```

This runs rollout collection, replay-buffer updates (critic MSE toward Bellman target, policy gradient maximizing Q), and periodic evaluation.

**Checkpoints** (relative to `rl_dart_6dof/`):

- `checkpoints/policy_best.pt` — best mean eval reward seen so far
- `checkpoints/policy_final.pt` — last weights after training

**Overrides** (optional):

```bash
python train.py --total-env-steps 10000 --seed 1
```

Hyperparameters live in `config.py` (`state_dim=12`, `action_dim=7`, learning rates, `gamma`, buffer size, etc.).

## Evaluate (effective DOF / PCA)

After training produces a checkpoint:

```bash
python evaluate.py
```

This loads `checkpoints/policy_best.pt` if present, otherwise `checkpoints/policy_final.pt`, rolls out episodes, then:

1. **PCA** on stacked trajectory states (6 angles + 6 velocities per timestep).
2. Prints **explained variance ratio** per principal component and **effective DOF** (smallest number of PCs reaching **≥ 95%** cumulative variance).
3. Prints **torque variance** per joint (from the first six action dimensions): how much each joint’s commanded torque varies along trajectories.

**Plots** (saved under `eval_outputs/`):

- `pca_variance.png` — cumulative explained variance vs. number of components (95% reference line).
- `torque_variance.png` — bar chart of per-joint torque variance.

Optional:

```bash
python evaluate.py --checkpoint checkpoints/policy_final.pt --episodes 80 --output-dir eval_outputs
```

If a GUI backend is available, `evaluate.py` will also open the figures (`plt.show()`).

## What the outputs mean

- **Higher eval reward** in `train.py` logs means higher dartboard-style score after release or timeout termination.
- **Lower effective DOF @ 95%** on joint trajectories suggests the motion lies near a lower-dimensional manifold (e.g. coordinated throwing motion), compared to unstructured exploration—not a proof of causality but a useful **post-hoc summary** of dimensionality along the trained policy rollouts.

## Project layout

| File             | Role                                      |
|------------------|-------------------------------------------|
| `env.py`         | `DartEnv`: 12-D state, 7-D action, throw  |
| `models.py`      | Policy (MLP+tanh), critic Q(s,a), target |
| `replay_buffer.py` | Ring buffer `add` / `sample`           |
| `train.py`       | Main RL loop + checkpointing              |
| `evaluate.py`    | PCA, effective DOF, torque variance plots |
| `utils.py`       | Seeds, plots, trajectory logging helpers  |
| `config.py`      | Hyperparameters                           |
