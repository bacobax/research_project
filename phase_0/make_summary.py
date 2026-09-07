#!/usr/bin/env python3
"""
Phase 0 -- aggregate exp01-exp04 JSON output into results/FINDINGS.md.

Every number in FINDINGS.md is read back from the experiment JSON files (no numbers are
hand-authored here) so the report can't drift from what the scripts actually produced. Run this
after exp01-exp04 (run_all.sh does this automatically).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def load(name: str):
    path = c.RESULTS_DIR / name
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def section_exp01(records) -> str:
    lines = ["## Experiment 1 — Train-region control\n"]
    if not records:
        lines.append("_exp01 did not run._\n")
        return "\n".join(lines)
    lines.append("| run | checkpoint | step | training loop's own train_acc | holdout acc (high/low) | control acc (high/low) |")
    lines.append("|---|---|---|---|---|---|")
    verdicts = []
    for r in records:
        h = r["holdout"]
        ctl = r["control"]
        train_acc = r["train_acc_top1_reported"]
        h_high, h_low = h.get("val/high_acc_top1"), h.get("val/low_acc_top1")
        c_high, c_low = ctl.get("val/high_acc_top1"), ctl.get("val/low_acc_top1")
        lines.append(
            f"| {r['run']} | {r['checkpoint']} | {r['step']} | {fmt(train_acc)} "
            f"| {fmt(h_high)} / {fmt(h_low)} | {fmt(c_high)} / {fmt(c_low)} |"
        )
        # Is control closer to the training loop's own number, or to holdout?
        ctl_mean = np.mean([v for v in (c_high, c_low) if v is not None])
        holdout_mean = np.mean([v for v in (h_high, h_low) if v is not None])
        dist_to_train = abs(ctl_mean - train_acc) if train_acc == train_acc else float("nan")
        dist_to_holdout = abs(ctl_mean - holdout_mean)
        verdicts.append((r["run"], r["checkpoint"], dist_to_train, dist_to_holdout))
    lines.append("")
    lines.append("Interpretation: if control accuracy sits close to the training loop's own reported ")
    lines.append("train accuracy (and far from holdout), the validation pipeline is measuring a real ")
    lines.append("generalization gap. If control sits close to holdout instead, validation construction ")
    lines.append("has a bug and prior results are void.\n")
    for run, ckpt, d_train, d_holdout in verdicts:
        verdict = "control tracks TRAIN (pipeline looks correct)" if d_train < d_holdout else "control tracks HOLDOUT (possible pipeline issue -- investigate)"
        lines.append(f"- {run}/{ckpt}: |control-train|={fmt(d_train,3)}, |control-holdout|={fmt(d_holdout,3)} -> **{verdict}**")
    return "\n".join(lines) + "\n"


def section_exp02(records) -> str:
    lines = ["\n## Experiment 2 — Token-space baselines vs. model\n"]
    if not records:
        lines.append("_exp02 did not run._\n")
        return "\n".join(lines)
    lines.append("Mean acc_top1 across all bands/mask-lengths, per (run, checkpoint, method):\n")
    lines.append("| run | checkpoint | " + " | ".join(c.BASELINE_NAMES + ["model"]) + " |")
    lines.append("|---|---|" + "---|" * (len(c.BASELINE_NAMES) + 1))
    beats = {"beats_repeat_left": 0, "loses_to_repeat_left": 0, "total": 0}
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            subset = [r for r in records if r["run"] == run_name and r["checkpoint"] == checkpoint_tag]
            if not subset:
                continue
            by_method = {m: float(np.mean([r["acc_top1"] for r in subset if r["method"] == m])) for m in c.BASELINE_NAMES + ["model"]}
            lines.append(
                f"| {run_name} | {checkpoint_tag} | " + " | ".join(fmt(by_method[m]) for m in c.BASELINE_NAMES + ["model"]) + " |"
            )
            beats["total"] += 1
            if by_method["model"] > by_method["repeat_left"]:
                beats["beats_repeat_left"] += 1
            else:
                beats["loses_to_repeat_left"] += 1
    lines.append("")
    lines.append(f"Model beats `repeat_left` in {beats['beats_repeat_left']}/{beats['total']} (run, checkpoint) combinations.")
    return "\n".join(lines) + "\n"


def section_exp03(records) -> str:
    lines = ["\n## Experiment 3 — Decoded-audio metrics (synthetic validation masks)\n"]
    if not records:
        lines.append("_exp03 did not run._\n")
        return "\n".join(lines)
    lines.append("Mean SI-SDR (dB, higher is better) and spectral convergence (lower is better) across all examples:\n")
    lines.append("| run | checkpoint | metric | " + " | ".join(c.BASELINE_NAMES + ["model"]) + " |")
    lines.append("|---|---|---|" + "---|" * (len(c.BASELINE_NAMES) + 1))
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            subset = [r for r in records if r["run"] == run_name and r["checkpoint"] == checkpoint_tag]
            if not subset:
                continue
            for metric_key, label in [("si_sdr_db", "SI-SDR(dB)"), ("stft_spectral_convergence", "spectral_convergence")]:
                by_method = {m: float(np.mean([r[metric_key] for r in subset if r["method"] == m])) for m in c.BASELINE_NAMES + ["model"]}
                lines.append(
                    f"| {run_name} | {checkpoint_tag} | {label} | " + " | ".join(fmt(by_method[m], 3) for m in c.BASELINE_NAMES + ["model"]) + " |"
                )
    return "\n".join(lines) + "\n"


def section_exp04(records) -> str:
    lines = ["\n## Experiment 4 — Real gaps, end-to-end\n"]
    if not records:
        lines.append("_exp04 did not run._\n")
        return "\n".join(lines)
    lines.append("SI-SDR (dB) per real gap, per (run, checkpoint, method):\n")
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            subset = [r for r in records if r["run"] == run_name and r["checkpoint"] == checkpoint_tag]
            if not subset:
                continue
            lines.append(f"\n**{run_name} / {checkpoint_tag}**\n")
            gap_indices = sorted(set(r["gap_index"] for r in subset))
            lines.append("| gap (ms) | " + " | ".join(c.BASELINE_NAMES + ["model"]) + " |")
            lines.append("|---|" + "---|" * (len(c.BASELINE_NAMES) + 1))
            for gi in gap_indices:
                gap_records = [r for r in subset if r["gap_index"] == gi]
                gap_ms = gap_records[0]["gap_len_ms"]
                by_method = {r["method"]: r["si_sdr_db"] for r in gap_records}
                lines.append(f"| {gap_ms:.0f} | " + " | ".join(fmt(by_method.get(m), 2) for m in c.BASELINE_NAMES + ["model"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    exp01 = load("exp01_train_region_control.json")
    exp02 = load("exp02_baselines.json")
    exp03 = load("exp03_decoded_metrics.json")
    exp04 = load("exp04_real_gaps.json")

    report = ["# Phase 0 Findings\n"]
    report.append(
        "Generated by `phase_0/make_summary.py` from `phase_0/results/exp0[1-4]_*.json`. "
        "Every number below is read back from those files, not hand-authored.\n"
    )
    report.append(section_exp01(exp01))
    report.append(section_exp02(exp02))
    report.append(section_exp03(exp03))
    report.append(section_exp04(exp04))

    out_path = c.RESULTS_DIR / "FINDINGS.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(report))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
