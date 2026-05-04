"""Single source of truth for repo-root-relative filesystem paths."""

from __future__ import annotations

from pathlib import Path

# src/dartrobot/paths.py -> parents[2] == repository root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root_SPEC() -> Path:
    return _PROJECT_ROOT


def artifacts_dir_SPEC(*sub: str) -> Path:
    return _PROJECT_ROOT.joinpath("artifacts", *sub)


def policies_dir_SPEC() -> Path:
    return _PROJECT_ROOT / "policies"


def mjcf_path_SPEC(name: str) -> Path:
    return Path(__file__).resolve().parent / "motion" / "mjcf" / name
