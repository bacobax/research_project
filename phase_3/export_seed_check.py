#!/usr/bin/env python3
"""Export the Phase 3 seed-robustness check: does the "more data (phase1_songs17) beats
regularization alone (phase2_regularization)" contrast hold across 3 seeds each?

Reuses the low-level TensorBoard-reading pieces from phase_2/export_results.py verbatim (this
project's own convention: each phase's exporter is an adapted copy, not a shared library). The
genuinely new logic here is the comparison methodology: rather than each run's own all-time best
(which isn't comparable -- the two new seeds per family only ran 300,000 steps, while the
original seed-42 runs ran 600,000/1,000,000), every run's best `val/combined_loss` is computed
using only points at step <= CUTOFF_STEP, applied identically to all 6 runs including the two
original seed-42 runs. This is NOT the runs' headline results (songs17: 4.87 @ step 950k;
phase2_regularization: 4.9936 @ step 25k, already <=300k so unaffected) -- it exists specifically
to make a fair, budget-matched 3-seed comparison.

Run from the repository root:
    uv run python phase_3/export_seed_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Copied verbatim from phase_2/export_results.py (zero seed-check-specific logic)
# ---------------------------------------------------------------------------
from phase_2.export_results import (  # noqa: E402
    ScalarPoint,
    _parse_run_log,
    _point_value,
    _run_command,
    _sha256,
    _save_figure,
    _style_axes,
    _tex_escape,
    _utc_timestamp,
    _validate_run,
    _write_csv,
    load_scalar_points,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT_DEFAULT = REPO_ROOT / "phase_3" / "results" / "seed_check"
CUTOFF_STEP_DEFAULT = 300_000

RUN_SPECS: list[dict[str, Any]] = [
    {
        "name": "phase1_songs17_seed42",
        "family": "songs17_data_scaling",
        "seed": 42,
        "run_dir": REPO_ROOT / "outputs/runs/phase1_data_scaling/phase1_songs17",
        "is_original": True,
    },
    {
        "name": "phase1_songs17_seed43",
        "family": "songs17_data_scaling",
        "seed": 43,
        "run_dir": REPO_ROOT / "outputs/runs/phase3_seed_check/phase3_songs17_seed43",
        "is_original": False,
    },
    {
        "name": "phase1_songs17_seed44",
        "family": "songs17_data_scaling",
        "seed": 44,
        "run_dir": REPO_ROOT / "outputs/runs/phase3_seed_check/phase3_songs17_seed44",
        "is_original": False,
    },
    {
        "name": "phase2_regularization_seed42",
        "family": "regularization_ablation",
        "seed": 42,
        "run_dir": REPO_ROOT / "outputs/runs/phase2_regularization/phase2_regularization",
        "is_original": True,
    },
    {
        "name": "phase2_regularization_seed43",
        "family": "regularization_ablation",
        "seed": 43,
        "run_dir": REPO_ROOT / "outputs/runs/phase3_seed_check/phase3_regularization_seed43",
        "is_original": False,
    },
    {
        "name": "phase2_regularization_seed44",
        "family": "regularization_ablation",
        "seed": 44,
        "run_dir": REPO_ROOT / "outputs/runs/phase3_seed_check/phase3_regularization_seed44",
        "is_original": False,
    },
]

FAMILY_METRIC_TAGS = {
    "best_combined_loss": "val/combined_loss",
    "best_high_acc_top1": "val/high_acc_top1",
    "best_low_acc_top1": "val/low_acc_top1",
    "best_si_sdr_db_gap_only": "val/si_sdr_db_gap_only",
}

PER_RUN_FIELDS = (
    "run", "family", "seed", "is_original", "cutoff_step", "best_step",
    "best_combined_loss", "best_high_acc_top1", "best_low_acc_top1",
    "best_si_sdr_db_gap_only", "stop_step", "stop_reason",
)
FAMILY_FIELDS = (
    "family", "n_seeds", "seeds",
    "mean_combined_loss", "std_combined_loss",
    "mean_high_acc_top1", "std_high_acc_top1",
    "mean_low_acc_top1", "std_low_acc_top1",
    "mean_si_sdr_db_gap_only", "std_si_sdr_db_gap_only",
)


@dataclass
class RunAtCutoff:
    name: str
    family: str
    seed: int
    is_original: bool
    run_dir: Path
    points: list[ScalarPoint]
    cutoff_step: int
    best_step: int
    best_combined_loss: float
    best_high_acc_top1: float | None
    best_low_acc_top1: float | None
    best_si_sdr_db_gap_only: float | None
    stop_step: int
    stop_reason: str


def analyze_run_at_cutoff(spec: Mapping[str, Any], cutoff_step: int) -> RunAtCutoff:
    run_dir: Path = spec["run_dir"]
    _validate_run(run_dir)
    points = load_scalar_points(run_dir / "tb")
    by_tag_step = {(p.tag, p.step): p for p in points}

    # The one-line methodological difference from phase_1/phase_2's analyze_run: filter to
    # step <= cutoff_step *before* taking the min, applied identically to original and new runs.
    combined = [p for p in points if p.tag == "val/combined_loss" and p.step <= cutoff_step]
    if not combined:
        raise ValueError(f"Run {spec['name']} has no val/combined_loss points at step <= {cutoff_step}")
    best_point = min(combined, key=lambda p: (p.value, p.step))
    best_step = best_point.step

    log_info = _parse_run_log(run_dir / "launch.log")

    return RunAtCutoff(
        name=spec["name"],
        family=spec["family"],
        seed=spec["seed"],
        is_original=spec["is_original"],
        run_dir=run_dir,
        points=[p for p in points if p.step <= cutoff_step],
        cutoff_step=cutoff_step,
        best_step=best_step,
        best_combined_loss=best_point.value,
        best_high_acc_top1=_point_value(by_tag_step, "val/high_acc_top1", best_step, required=False),
        best_low_acc_top1=_point_value(by_tag_step, "val/low_acc_top1", best_step, required=False),
        best_si_sdr_db_gap_only=_point_value(by_tag_step, "val/si_sdr_db_gap_only", best_step, required=False),
        stop_step=log_info["stop_step"],
        stop_reason=log_info["stop_reason"],
    )


def family_summary(runs: Sequence[RunAtCutoff]) -> list[dict[str, Any]]:
    families: dict[str, list[RunAtCutoff]] = {}
    for run in runs:
        families.setdefault(run.family, []).append(run)

    rows = []
    for family, members in families.items():
        members = sorted(members, key=lambda r: r.seed)

        def _mean_std(attr: str) -> tuple[float, float]:
            values = [getattr(m, attr) for m in members if getattr(m, attr) is not None]
            if len(values) < 2:
                return (values[0] if values else float("nan")), float("nan")
            return statistics.mean(values), statistics.stdev(values)

        mean_loss, std_loss = _mean_std("best_combined_loss")
        mean_high, std_high = _mean_std("best_high_acc_top1")
        mean_low, std_low = _mean_std("best_low_acc_top1")
        mean_sdr, std_sdr = _mean_std("best_si_sdr_db_gap_only")
        rows.append({
            "family": family,
            "n_seeds": len(members),
            "seeds": ",".join(str(m.seed) for m in members),
            "mean_combined_loss": mean_loss, "std_combined_loss": std_loss,
            "mean_high_acc_top1": mean_high, "std_high_acc_top1": std_high,
            "mean_low_acc_top1": mean_low, "std_low_acc_top1": std_low,
            "mean_si_sdr_db_gap_only": mean_sdr, "std_si_sdr_db_gap_only": std_sdr,
        })
    return rows


def _write_family_tex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def fmt(mean: Any, std: Any, digits: int = 4) -> str:
        if mean is None or (isinstance(mean, float) and mean != mean):
            return "--"
        if std is None or (isinstance(std, float) and std != std):
            return f"{float(mean):.{digits}f}"
        return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"

    lines = [
        "% Generated by phase_3/export_seed_check.py; requires \\usepackage{booktabs}.",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Family & Combined NLL (mean $\pm$ std) & High acc. & SI-SDR gap-only (dB) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} ({} seeds) & {} & {} & {} \\\\".format(
                _tex_escape(row["family"]), row["n_seeds"],
                fmt(row["mean_combined_loss"], row["std_combined_loss"]),
                fmt(row["mean_high_acc_top1"], row["std_high_acc_top1"]),
                fmt(row["mean_si_sdr_db_gap_only"], row["std_si_sdr_db_gap_only"], 2),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_seed_check(runs: Sequence[RunAtCutoff], cutoff_step: int, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {"songs17_data_scaling": "tab:blue", "regularization_ablation": "tab:orange"}
    linestyles = {42: "-", 43: "--", 44: ":"}
    for run in runs:
        steps = [p.step for p in run.points if p.tag == "val/combined_loss"]
        values = [p.value for p in run.points if p.tag == "val/combined_loss"]
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        steps = [steps[i] for i in order]
        values = [values[i] for i in order]
        color = colors[run.family]
        style = linestyles.get(run.seed, "-.")
        label = f"{run.family} (seed {run.seed}{', original' if run.is_original else ''})"
        ax.plot(steps, values, color=color, linestyle=style, marker="o", markersize=2.5, label=label)
        ax.scatter([run.best_step], [run.best_combined_loss], color=color, marker="*", s=130, zorder=5)
    ax.axvline(cutoff_step, color="grey", linestyle=":", linewidth=1)
    ax.text(cutoff_step, ax.get_ylim()[1], f" {cutoff_step:,}-step cutoff", va="top", fontsize=8, color="grey")
    ax.set(xlabel="Training step", ylabel="val/combined_loss", title="Phase 3 seed-robustness check (steps ≤ cutoff)")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "seed_check_combined_loss")


def _provenance(runs: Sequence[RunAtCutoff], cutoff_step: int, command: str) -> dict[str, Any]:
    raw_runs: dict[str, Any] = {}
    latest_mtime = 0.0
    config_relpaths = {
        "phase1_songs17_seed42": "configs/train/phase1_songs17.yaml",
        "phase1_songs17_seed43": "configs/train/phase3_songs17_seed43.yaml",
        "phase1_songs17_seed44": "configs/train/phase3_songs17_seed44.yaml",
        "phase2_regularization_seed42": "configs/train/phase2_regularization.yaml",
        "phase2_regularization_seed43": "configs/train/phase3_regularization_seed43.yaml",
        "phase2_regularization_seed44": "configs/train/phase3_regularization_seed44.yaml",
    }
    for run in runs:
        artifacts = []
        candidates = sorted((run.run_dir / "tb").glob("events.out.tfevents.*")) + [
            run.run_dir / "checkpoints" / "best_val.pt",
            run.run_dir / "launch.log",
        ]
        for path in candidates:
            if not path.exists():
                continue
            stat = path.stat()
            latest_mtime = max(latest_mtime, stat.st_mtime)
            artifacts.append({
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": _sha256(path),
            })
        raw_runs[run.name] = {
            "cutoff_step": cutoff_step,
            "best_step_at_cutoff": run.best_step,
            "config": config_relpaths[run.name],
            "raw_artifacts": artifacts,
        }
    status = _run_command(REPO_ROOT, ["git", "status", "--short", "--", "configs/train/phase3_*.yaml"])
    return {
        "schema_version": 1,
        "generated_from_artifacts_at_utc": _utc_timestamp(latest_mtime) if latest_mtime else None,
        "export_command": command,
        "git_commit": _run_command(REPO_ROOT, ["git", "rev-parse", "HEAD"]),
        "git_status_relevant": status.splitlines() if status else [],
        "cutoff_step": cutoff_step,
        "runs": raw_runs,
    }


def _write_readme(output_root: Path, family_rows: Sequence[Mapping[str, Any]], cutoff_step: int) -> None:
    def fmt(mean: Any, std: Any, digits: int = 4) -> str:
        if mean is None or (isinstance(mean, float) and mean != mean):
            return "--"
        if std is None or (isinstance(std, float) and std != std):
            return f"{float(mean):.{digits}f} (n=1)"
        return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"

    lines = []
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['n_seeds']} | {row['seeds']} "
            f"| {fmt(row['mean_combined_loss'], row['std_combined_loss'])} "
            f"| {fmt(row['mean_high_acc_top1'], row['std_high_acc_top1'])} "
            f"| {fmt(row['mean_si_sdr_db_gap_only'], row['std_si_sdr_db_gap_only'], 2)} |"
        )
    text = f"""# Phase 3 seed-robustness check

**All values in this table are the best `val/combined_loss` reached at or before step
{cutoff_step:,}** -- for the two original seed-42 runs this is *not* their eventual best
(reported elsewhere as 4.87 at step 950,000 for `phase1_songs17`, 4.9936 at step 25,000 for
`phase2_regularization`, the latter already ≤{cutoff_step:,} so unaffected). This table exists
specifically to make a fair, budget-matched 3-seed comparison of the "more data beats
regularization alone" contrast, not to restate the headline single-seed results.

## Family summary (mean ± sample std across 3 seeds)

| family | n seeds | seeds | combined NLL | high-activity acc | SI-SDR gap-only (dB) |
|---|---|---|---|---|---|
{chr(10).join(lines)}

## Contents

- `per_run_at_cutoff.csv`: one row per run (6 total) with its own best-at-cutoff step and metrics.
- `family_summary.csv` / `tables/family_summary.tex`: the table above.
- `metrics.csv`: all loaded scalar points for all 6 runs, filtered to step ≤ {cutoff_step:,}.
- `figures/seed_check_combined_loss.{{png,pdf}}`: val/combined_loss trajectories to the cutoff,
  colored by family, one linestyle per seed, star at each run's best-at-cutoff point.
- `provenance.json`: source hashes and config paths for every run.

## Regeneration

```bash
uv run python phase_3/export_seed_check.py
```
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def export_seed_check(*, output_root: Path, cutoff_step: int, command: str) -> list[RunAtCutoff]:
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = [analyze_run_at_cutoff(spec, cutoff_step) for spec in RUN_SPECS]

    per_run_rows = [{
        "run": r.name, "family": r.family, "seed": r.seed, "is_original": r.is_original,
        "cutoff_step": r.cutoff_step, "best_step": r.best_step,
        "best_combined_loss": r.best_combined_loss, "best_high_acc_top1": r.best_high_acc_top1,
        "best_low_acc_top1": r.best_low_acc_top1, "best_si_sdr_db_gap_only": r.best_si_sdr_db_gap_only,
        "stop_step": r.stop_step, "stop_reason": r.stop_reason,
    } for r in runs]
    family_rows = family_summary(runs)

    metrics_rows = [{
        "run": r.name, "tag": p.tag, "step": p.step,
        "wall_time_utc": _utc_timestamp(p.wall_time), "value": f"{p.value:.10g}",
    } for r in runs for p in r.points]

    _write_csv(output_root / "per_run_at_cutoff.csv", PER_RUN_FIELDS, per_run_rows)
    _write_csv(output_root / "family_summary.csv", FAMILY_FIELDS, family_rows)
    _write_csv(output_root / "metrics.csv", ("run", "tag", "step", "wall_time_utc", "value"), metrics_rows)
    _write_family_tex(tables_dir / "family_summary.tex", family_rows)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
    _plot_seed_check(runs, cutoff_step, figures_dir)

    provenance = _provenance(runs, cutoff_step, command)
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(output_root, family_rows, cutoff_step)
    return runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-step", type=int, default=CUTOFF_STEP_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = f"uv run python phase_3/export_seed_check.py --cutoff-step {args.cutoff_step} --output-root {args.output_root.as_posix()}"
    runs = export_seed_check(output_root=args.output_root, cutoff_step=args.cutoff_step, command=command)
    for run in runs:
        print(f"{run.name}: best-at-cutoff step {run.best_step}, combined_loss {run.best_combined_loss:.4f}")
    print(f"Wrote paper-ready results to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
