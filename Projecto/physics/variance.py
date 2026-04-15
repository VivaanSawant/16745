"""
Variance / trajectory randomness: given a nominal release state, apply noise
and run Monte Carlo to get landing distribution and variance.
Used for RL learning of variance and distribution-aware optimization.
"""
import numpy as np
from .projectile import throw_3d, landing_position
from . import dartboard


def sample_release_noise(
    vel0,
    release_time=0.0,
    vel_std=0.1,
    time_std=0.005,
    rng=None,
):
    """
    Add Gaussian noise to release velocity and optional release timing.
    vel0: (3,) m/s. noise is scaled by vel_std (fraction or absolute).
    Returns (vel_noisy, release_time_noisy).
    """
    rng = rng or np.random.default_rng()
    vel0 = np.asarray(vel0, dtype=float)
    # Scale: if vel_std < 1 treat as fraction of |vel|
    vmag = np.linalg.norm(vel0)
    if vmag > 1e-10 and vel_std < 1:
        scale = vel_std * vmag
    else:
        scale = vel_std if vel_std >= 1 else 0.5
    vel_noisy = vel0 + scale * rng.standard_normal(3)
    t_noisy = release_time + time_std * rng.standard_normal()
    return vel_noisy, max(0.0, t_noisy)


def monte_carlo_landings(
    pos0,
    vel0,
    n_samples=100,
    vel_std=0.08,
    time_std=0.005,
    wind_mean=(0.0, 0.0, 0.0),
    wind_std=0.0,
    seed=None,
    **throw_kw,
):
    """
    Run n_samples throws with release and optional wind noise.
    Returns (landings_xy, scores): landings_xy (n, 2), scores (n,) from dartboard.
    """
    rng = np.random.default_rng(seed)
    landings = []
    for _ in range(n_samples):
        v, _ = sample_release_noise(vel0, vel_std=vel_std, time_std=time_std, rng=rng)
        w = np.array(wind_mean) + wind_std * rng.standard_normal(3)
        w[2] = wind_mean[2]  # optional: no vertical wind
        x, y = landing_position(pos0, v, wind=tuple(w), **throw_kw)
        landings.append((x, y))
    landings_xy = np.array(landings)
    scores = np.array([dartboard.score_at(x, y) for x, y in landings_xy])
    return landings_xy, scores


def landing_stats(landings_xy, scores):
    """Given (n,2) landings and (n,) scores, return mean landing, std, expected score."""
    mean_xy = np.mean(landings_xy, axis=0)
    std_xy = np.std(landings_xy, axis=0)
    expected_score = np.mean(scores)
    return mean_xy, std_xy, expected_score
