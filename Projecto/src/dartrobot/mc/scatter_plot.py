"""Board-view scatter plots for Monte Carlo landing diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dartrobot.constants import (
    R_BOARD_MISS_MM,
    R_DOUBLE_INNER_MM,
    R_DOUBLE_OUTER_MM,
    R_INNER_BULL_MM,
    R_OUTER_BULL_MM,
    R_TRIPLE_INNER_MM,
    R_TRIPLE_OUTER_MM,
)


def draw_board_rings_mm_SPEC(ax) -> None:
    th = np.linspace(0.0, 2.0 * math.pi, 360)
    for r_mm, style in (
        (R_INNER_BULL_MM, "k-"),
        (R_OUTER_BULL_MM, "k-"),
        (R_TRIPLE_INNER_MM, "k:"),
        (R_TRIPLE_OUTER_MM, "k:"),
        (R_DOUBLE_INNER_MM, "k--"),
        (R_DOUBLE_OUTER_MM, "k--"),
        (R_BOARD_MISS_MM, "k-"),
    ):
        ax.plot(r_mm * np.sin(th), r_mm * np.cos(th), style, lw=0.6, alpha=0.65)


def plot_mc_board_SPEC(
    dy_m: np.ndarray,
    dz_m: np.ndarray,
    scores: np.ndarray,
    aim_dy_m: float,
    aim_dz_m: float,
    out_path: Path,
    *,
    episode_totals: np.ndarray | None = None,
    dartboard_face_hit_rate: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dy_mm = np.asarray(dy_m, dtype=float) * 1000.0
    dz_mm = np.asarray(dz_m, dtype=float) * 1000.0
    valid = np.isfinite(dy_mm) & np.isfinite(dz_mm)

    if episode_totals is not None and episode_totals.size > 0:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 5.2))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6.5, 6.5))
        ax1 = None

    draw_board_rings_mm_SPEC(ax0)
    sc = ax0.scatter(
        dy_mm[valid],
        dz_mm[valid],
        c=np.asarray(scores, dtype=float)[valid],
        cmap="viridis",
        s=18,
        alpha=0.75,
        vmin=0,
        vmax=60,
    )
    ax0.scatter(
        [aim_dy_m * 1000.0],
        [aim_dz_m * 1000.0],
        marker="*",
        s=220,
        c="red",
        edgecolors="k",
        linewidths=0.4,
        zorder=5,
        label="aim",
    )
    ax0.set_aspect("equal")
    ax0.set_xlabel("Δy (mm)")
    ax0.set_ylabel("Δz (mm)")
    title = "Monte Carlo landings (bull at origin)"
    if dartboard_face_hit_rate is not None:
        title += (
            f"\nplane x=2.37 m crossing ≠ on board face; "
            f"face (r≤{R_BOARD_MISS_MM:.0f} mm): {100.0 * float(dartboard_face_hit_rate):.1f}%"
        )
    ax0.set_title(title, fontsize=10)
    ax0.grid(True, alpha=0.35)
    ax0.legend(loc="upper right")
    fig.colorbar(sc, ax=ax0, label="score", shrink=0.72)

    if ax1 is not None:
        ax1.hist(episode_totals, bins=min(30, max(8, int(np.sqrt(len(episode_totals))))), color="steelblue", edgecolor="k", alpha=0.85)
        ax1.set_xlabel("episode total score")
        ax1.set_ylabel("count")
        ax1.set_title("Per-episode sum of scores")
        ax1.grid(True, alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
