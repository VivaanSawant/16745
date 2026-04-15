"""
run_dart_robot_demo_SPEC.py
===========================
**Top-level demo** for the **CMU 16-745 dart robot task list** implementation.

Runs, in order:
1. **B3/B4** spot checks + drag comparison + saves plots to `artifacts_SPEC/`
2. **A4** workspace + sample release speed
3. **C1** nominal end-to-end + small Monte Carlo

**PDF:** This is the integration smoke test described under **Track C — C1** + validations.

Usage:
    python run_dart_robot_demo_SPEC.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    py = sys.executable
    scripts = [
        ROOT / "track_B_projectile_SPEC" / "VALIDATION_B4_drag_comparison_and_plots_SPEC.py",
        ROOT / "track_A_arm_SPEC" / "VALIDATION_A4_workspace_release_speed_SPEC.py",
        ROOT / "track_C_integration_SPEC" / "VALIDATION_C1_end_to_end_and_monte_carlo_SPEC.py",
    ]
    for s in scripts:
        print(f"\n===== Running {s.name} =====\n", flush=True)
        r = subprocess.run([py, str(s)], cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    print("\n===== run_dart_robot_demo_SPEC: all steps completed =====\n")


if __name__ == "__main__":
    main()
