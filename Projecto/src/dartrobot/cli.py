"""Unified CLI: ``dartrobot <subcommand> ...``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(
        prog="dartrobot",
        description="Dart robot SPEC tools (demo, Monte Carlo, baseline cache, debug, RL).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Subcommands registered after their modules exist (lazy import avoids import errors mid-migration)
    sub.add_parser("demo", help="Run visual demo pipeline + showcase report")
    sub.add_parser("mc", help="Target-score Monte Carlo toward a board aim")
    sub.add_parser("baseline", help="Build strong classical baseline cache (knots + release time)")
    sub.add_parser("debug", help="Stage-by-stage throw pipeline diagnostics")
    sub.add_parser("rl-train", help="Train PPO residual policy (SB3)")
    sub.add_parser("rl-eval", help="Evaluate saved PPO policy at one target")

    args, rest = p.parse_known_args(argv)
    if args.cmd == "demo":
        from dartrobot.demo.main import main as demo_main

        return int(demo_main(rest))
    if args.cmd == "mc":
        from dartrobot.mc import monte_carlo as mc_mod

        return int(mc_mod.main(rest))
    if args.cmd == "baseline":
        from dartrobot.integration import strengthen_baseline as bl_mod

        return int(bl_mod.main(rest))
    if args.cmd == "debug":
        from dartrobot.integration import debug_throw_pipeline as dbg_mod

        return int(dbg_mod.main(rest))
    if args.cmd == "rl-train":
        from dartrobot.rl import train_sb3_residual as tr_mod

        return int(tr_mod.main(rest))
    if args.cmd == "rl-eval":
        from dartrobot.rl import eval_sb3_residual as ev_mod

        return int(ev_mod.main(rest))
    return 1
