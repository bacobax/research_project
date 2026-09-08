#!/usr/bin/env python3
"""Create consistent raw and portable output locations for a research phase."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


RESULT_SUBDIRECTORIES = ("figures", "tables", "configs", "manifests")


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def scaffold_phase(repo_root: Path, phase_number: int) -> tuple[Path, Path]:
    if phase_number < 0:
        raise ValueError("phase number must be non-negative")

    repo_root = repo_root.resolve()
    phase_name = f"phase_{phase_number}"
    phase_dir = repo_root / phase_name
    results_dir = phase_dir / "results"
    raw_dir = repo_root / "outputs" / "runs" / phase_name

    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory in RESULT_SUBDIRECTORIES:
        directory = results_dir / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        _write_if_missing(directory / ".gitkeep", "")

    _write_if_missing(
        phase_dir / "README.md",
        f"""# Phase {phase_number}

- Raw training artifacts: `outputs/runs/{phase_name}/` (local and Git-ignored)
- Portable paper results: `{phase_name}/results/` (Git-tracked)

Put experiment code and analysis for this phase in this directory. Export only
compact numerical results, figures, tables, configuration snapshots, and
provenance to `results/`.
""",
    )
    _write_if_missing(
        results_dir / "README.md",
        f"""# Phase {phase_number} results

This directory is the portable, Git-trackable results bundle for Phase {phase_number}.

Expected contents:

- `figures/`: publication-ready PDF figures and PNG previews
- `tables/`: LaTeX tables
- `configs/`: exact experiment configuration snapshots
- `manifests/`: dataset and provenance manifests
- CSV/JSON summaries at this directory's top level

Do not copy checkpoints, TensorBoard event files, full WAV files, or other large
raw artifacts here. Keep them under `outputs/runs/{phase_name}/` and record their
paths and hashes in a provenance manifest.
""",
    )
    return raw_dir, results_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_number", type=int, help="Non-negative phase number")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw_dir, results_dir = scaffold_phase(args.repo_root, args.phase_number)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Raw outputs: {raw_dir}")
    print(f"Portable results: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
