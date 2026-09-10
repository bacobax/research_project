#!/usr/bin/env python3
"""
Phase 3 / Experiment 1 -- Gap-length x activity-band degradation sweep.

Evaluates the two Phase 3 headline checkpoints (phase1_songs17, phase2_regularization; both
best_val.pt, the checkpoints already reported as "the" result for each config in their
respective phase_1/phase_2 run_summary.csv) at mask_lengths=(1,2,4,8,16) frames x
activity_band=(high,low) activity. Model-only: no non-learned baselines (deliberately out of
scope -- see phase_3/README.md).

IMPORTANT -- both checkpoints were trained with mask_len_max=4 and curriculum=false (never saw
a gap longer than 4 frames / ~53ms during training). mask_len in {8, 16} is genuinely
OUT-OF-DISTRIBUTION for both models, not a same-distribution grid extension. Every output of
this script (JSON, summary.csv rows, console output, and the generated FINDINGS.md) marks these
rows explicitly (ood_beyond_trained_range: true) -- do not describe degradation (or lack of it)
at mask_len 8/16 as "generalization," only as an out-of-distribution probe.

Usage:
    uv run python phase_3/exp01_gap_length_activity_sweep.py
    uv run python phase_3/exp01_gap_length_activity_sweep.py --runs phase1_songs17 --mask-lengths 1
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MASK_LENGTHS = (1, 2, 4, 8, 16)
TRAINED_MASK_LEN_MAX = 4  # both headline runs' mask_len_max -- anything above this is OOD
CHECKPOINT_TAG = "best_val"
RUN_NAMES = ("phase1_songs17", "phase2_regularization")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_0"))
import common as c  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSON_PATH = RESULTS_DIR / "exp01_gap_length_activity_sweep.json"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"  # phase_3's OWN summary.csv, not phase_0's
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_COLUMNS = c.SUMMARY_COLUMNS

METRIC_KEYS = (
    "loss", "nll", "ppl", "acc_top1", "acc_top5",
    "si_sdr_db", "si_sdr_db_gap_only",
    "spectral_convergence", "spectral_convergence_gap_only",
)


def append_summary_rows(rows: List[Dict[str, Any]]) -> None:
    """Same csv.DictWriter logic as c.append_summary_rows, but targets phase_3's own
    summary.csv -- calling c.append_summary_rows itself would write into phase_0/results/,
    polluting Phase 0's own committed results."""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({**{col: "" for col in SUMMARY_COLUMNS}, **row})


def run_one(run_name: str, mask_lengths: Tuple[int, ...]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trainer = c.build_trainer(run_name, CHECKPOINT_TAG)
    step = trainer.global_step
    frame_rate = trainer.encoder.frame_rate

    grouped_examples, _ = c.build_real_validation_examples(trainer, mask_lengths=mask_lengths)
    dataloaders, group_specs = c.build_validation_dataloaders(trainer, grouped_examples)
    trainer.validation_audio_targets = c.build_validation_audio_targets(trainer, grouped_examples)
    c.run_validation_safe(trainer, dataloaders, group_specs, step=step)

    scalars = trainer.writer.as_dict()
    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for band_key, band_name in (("high", "high_activity"), ("low", "low_activity")):
        for mask_len in mask_lengths:
            prefix = f"val/{band_key}_len_{mask_len}"
            metrics = {key: scalars[f"{prefix}_{key}"] for key in METRIC_KEYS}
            record = {
                "run": run_name,
                "checkpoint": CHECKPOINT_TAG,
                "step": step,
                "band": band_name,
                "mask_len": mask_len,
                "gap_ms": 1000.0 * mask_len / frame_rate,
                "ood_beyond_trained_range": mask_len > TRAINED_MASK_LEN_MAX,
                **metrics,
            }
            records.append(record)
            for metric_name, value in metrics.items():
                rows.append({
                    "experiment": "exp01_gap_length_activity_sweep",
                    "run": run_name,
                    "checkpoint": CHECKPOINT_TAG,
                    "step": step,
                    "band": band_name,
                    "mask_len": mask_len,
                    "method": "model",
                    "metric": metric_name,
                    "value": value,
                })
    return records, rows


def plot_degradation_curves(all_records: List[Dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    bands = ("high_activity", "low_activity")
    plot_metrics = ("acc_top1", "si_sdr_db_gap_only")
    runs = sorted({r["run"] for r in all_records})
    ms_per_frame = next(r["gap_ms"] / r["mask_len"] for r in all_records)
    ood_boundary_ms = TRAINED_MASK_LEN_MAX * ms_per_frame

    fig, axes = plt.subplots(len(bands), len(plot_metrics), figsize=(11, 8), squeeze=False)
    for row_idx, band in enumerate(bands):
        for col_idx, metric in enumerate(plot_metrics):
            ax = axes[row_idx][col_idx]
            for run_name in runs:
                pts = sorted(
                    (r["gap_ms"], r[metric]) for r in all_records if r["run"] == run_name and r["band"] == band
                )
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", label=run_name)
            ax.axvline(ood_boundary_ms, color="gray", linestyle="--", linewidth=1)
            if row_idx == 0 and col_idx == len(plot_metrics) - 1:
                ax.text(
                    ood_boundary_ms, ax.get_ylim()[1], " OOD beyond here", va="top", ha="left",
                    fontsize=8, color="gray",
                )
            ax.set_title(f"{band} -- {metric}")
            ax.set_xlabel("gap length (ms)")
            ax.set_ylabel(metric)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        "Phase 3: gap-length x activity-band degradation (dashed line = trained range ends; "
        "beyond it is out-of-distribution)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"gap_length_degradation.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=list(RUN_NAMES), choices=list(RUN_NAMES))
    parser.add_argument("--mask-lengths", nargs="+", type=int, default=list(MASK_LENGTHS))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    mask_lengths = tuple(args.mask_lengths)

    all_records: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for run_name in args.runs:
        records, rows = run_one(run_name, mask_lengths)
        all_records.extend(records)
        all_rows.extend(rows)
        print(f"=== {run_name} ({CHECKPOINT_TAG}) ===")
        for r in records:
            ood_flag = " [OOD]" if r["ood_beyond_trained_range"] else ""
            print(
                f"  {r['band']:14s} len={r['mask_len']:2d}{ood_flag:5s} "
                f"acc_top1={r['acc_top1']:.3f} si_sdr_gap={r['si_sdr_db_gap_only']:7.2f}dB"
            )

    c.save_json(JSON_PATH, all_records)
    append_summary_rows(all_rows)
    if len(all_records) > 1:
        plot_degradation_curves(all_records, FIGURES_DIR)
    print(f"\nWrote {JSON_PATH}")


if __name__ == "__main__":
    main()
