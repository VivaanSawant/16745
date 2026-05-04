"""
run_dart_robot_demo_SPEC.py
===========================
**Top-level demo** for the **CMU 16-745 dart robot task list** implementation.

Runs, in order:
1. **B3/B4** spot checks + drag comparison + saves plots to `artifacts/`
2. **A4** workspace + sample release speed
3. **C1** nominal end-to-end + small Monte Carlo

**PDF:** This is the integration smoke test described under **Track C — C1** + validations.

Usage:
    dartrobot demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

from dartrobot.paths import project_root_SPEC


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual dart robot demo pipeline.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip C1 Monte Carlo stage and build showcase from available artifacts.",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open generated HTML report in default browser (best effort).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    py = sys.executable
    ROOT = project_root_SPEC()
    scripts = [
        ROOT / "tests" / "flight" / "drag_comparison_and_plots.py",
        ROOT / "tests" / "motion" / "workspace_release_speed.py",
    ]
    if not args.quick:
        scripts.append(ROOT / "tests" / "integration" / "end_to_end_and_monte_carlo.py")

    showcase_script = Path(__file__).resolve().parent / "showcase.py"
    for s in scripts:
        print(f"\n===== Running {s.name} =====\n", flush=True)
        r = subprocess.run([py, str(s)], cwd=str(ROOT))
        if r.returncode != 0:
            return int(r.returncode)

    print(f"\n===== Running {showcase_script.name} =====\n", flush=True)
    r = subprocess.run([py, str(showcase_script)], cwd=str(ROOT))
    if r.returncode != 0:
        return int(r.returncode)

    report = ROOT / "artifacts" / "demo" / "dart_robot_visual_demo_SPEC.html"
    video = ROOT / "artifacts" / "motion" / "motion_throw_video_SPEC.mp4"
    print("\n===== Dart Robot Demo Complete =====")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Showcase report: {report}")
    print(f"Throw video: {video}")
    print("Artifacts folder: {0}".format(ROOT / "artifacts"))
    print("===================================\n")

    if args.open_report:
        try:
            webbrowser.open(report.resolve().as_uri())
            print(f"Opened in browser: {report}")
        except Exception as exc:
            print(f"Could not open browser automatically: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
