#!/usr/bin/env python3
"""Phase 4, exp03 -- STFT-loss ablation: paired comparison of two matched twins.

Question (supervisor Q2): is the decoded-domain STFT/waveform loss helping, hurting, or doing
nothing? Both runs are the same configuration (regularized single song, seed 43, 300k steps); the
"on" run is phase3_regularization_seed43 (decoded loss weight 0.1), the "off" run is
phase4_stft_off_seed43 (`decoded_loss_enabled: false`). Pure post-processing of TensorBoard
scalars -- no GPU, no training.

Outputs under phase_4/results/exp03_stft_ablation/:
  comparison.csv  one row per (metric, validation step): on, off, off - on
  summary.json    final-step and best-at-cutoff comparison per metric, with the seed-noise
                  band of the regularization family (phase_3/results/seed_check) and a verdict
  stft_ablation.{png,pdf}   paired validation curves

Verdict rule (fixed in advance, per metric that has a seed-noise reference):
  |off - on| < 1 std -> "no detectable effect"; 1-2 std -> "inconclusive"; > 2 std -> "effect".
One seed cannot resolve differences below the seed spread, so "no detectable effect" is not "no effect".

    uv run python phase_4/exp03_stft_ablation.py             # full comparison
    uv run python phase_4/exp03_stft_ablation.py --self-check # on-run alone vs exported seed-check numbers
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from phase_2.export_results import load_scalar_points  # noqa: E402

ON_TB = REPO_ROOT / "outputs/runs/phase3_seed_check/phase3_regularization_seed43/tb"
OFF_TB = REPO_ROOT / "outputs/runs/phase4_stft_ablation/phase4_stft_off_seed43/tb"
FAMILY_CSV = REPO_ROOT / "phase_3/results/seed_check/family_summary.csv"
PER_RUN_CSV = REPO_ROOT / "phase_3/results/seed_check/per_run_at_cutoff.csv"
OUT_DIR = REPO_ROOT / "phase_4/results/exp03_stft_ablation"
CUTOFF_STEP = 300_000
FAMILY = "regularization_ablation"

# metric tag -> (direction where larger is better?, family_summary std column or None)
METRICS: Dict[str, Tuple[bool, Optional[str]]] = {
    "val/combined_loss": (False, "std_combined_loss"),
    "val/high_acc_top1": (True, "std_high_acc_top1"),
    "val/low_acc_top1": (True, "std_low_acc_top1"),
    "val/si_sdr_db_gap_only": (True, "std_si_sdr_db_gap_only"),
    "val/si_sdr_db": (True, None),
    "val/spectral_convergence_gap_only": (False, None),
    "val/spectral_convergence": (False, None),
    "train/avg_nll": (False, None),
}


def series(points, tag: str) -> Dict[int, float]:
    return {p.step: p.value for p in points if p.tag == tag and p.step <= CUTOFF_STEP}


def best_at_cutoff(points) -> Tuple[int, Dict[str, float]]:
    comb = series(points, "val/combined_loss")
    step = min(comb, key=lambda s: (comb[s], s))
    vals = {tag: series(points, tag).get(step) for tag in METRICS}
    return step, vals


def final_values(points) -> Tuple[int, Dict[str, float]]:
    comb = series(points, "val/combined_loss")
    step = max(comb)
    return step, {tag: series(points, tag).get(step) for tag in METRICS}


def family_std() -> Dict[str, float]:
    with open(FAMILY_CSV, newline="") as fh:
        row = next(r for r in csv.DictReader(fh) if r["family"] == FAMILY)
    return {tag: float(row[col]) for tag, (_, col) in METRICS.items() if col}


def verdict(delta: float, std: Optional[float], higher_better: bool) -> str:
    if std is None:
        return "no seed-noise reference"
    z = abs(delta) / std
    if z < 1:
        return "no detectable effect"
    if z < 2:
        return "inconclusive"
    better = (delta > 0) == higher_better
    return "effect: STFT-off better" if better else "effect: STFT-off worse"


def self_check() -> None:
    """The on-run alone must reproduce the numbers already exported by phase_3's seed check."""
    pts = load_scalar_points(ON_TB)
    step, vals = best_at_cutoff(pts)
    with open(PER_RUN_CSV, newline="") as fh:
        ref = next(r for r in csv.DictReader(fh) if r["run"] == "phase2_regularization_seed43")
    checks = [("best_step", step, int(ref["best_step"])),
              ("val/combined_loss", vals["val/combined_loss"], float(ref["best_combined_loss"])),
              ("val/high_acc_top1", vals["val/high_acc_top1"], float(ref["best_high_acc_top1"])),
              ("val/low_acc_top1", vals["val/low_acc_top1"], float(ref["best_low_acc_top1"])),
              ("val/si_sdr_db_gap_only", vals["val/si_sdr_db_gap_only"], float(ref["best_si_sdr_db_gap_only"]))]
    ok = True
    for name, got, want in checks:
        good = abs(got - want) < 1e-6
        ok &= good
        print(f"  {name:28s} got {got!r:>22}  exported {want!r:>22}  {'OK' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit("self-check failed: this script does not reproduce the exported seed-check numbers")
    print("self-check passed")


def make_figure(on: Dict[str, Dict[int, float]], off: Dict[str, Dict[int, float]], std: Dict[str, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = ["val/combined_loss", "val/high_acc_top1", "val/low_acc_top1", "val/si_sdr_db_gap_only",
            "val/spectral_convergence_gap_only", "train/avg_nll"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, tag in zip(axes.ravel(), tags):
        for label, data, color in (("decoded loss ON (0.1)", on[tag], "tab:blue"), ("decoded loss OFF", off[tag], "tab:red")):
            xs = sorted(data)
            ax.plot(xs, [data[s] for s in xs], "o-" if tag.startswith("val/") else "-", color=color, label=label,
                    markersize=4, linewidth=1.3)
        if tag in std and on[tag]:
            xs = sorted(on[tag])
            lo = [on[tag][s] - std[tag] for s in xs]
            hi = [on[tag][s] + std[tag] for s in xs]
            ax.fill_between(xs, lo, hi, color="tab:blue", alpha=0.12, label="ON +/- 1 seed std (family)")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("STFT-loss ablation: regularized single song, seed 43 (matched twins)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"stft_ablation.{ext}", dpi=150 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true", help="validate the on-run numbers against phase_3's export and exit")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return

    on_pts, off_pts = load_scalar_points(ON_TB), load_scalar_points(OFF_TB)
    std = family_std()
    on = {tag: series(on_pts, tag) for tag in METRICS}
    off = {tag: series(off_pts, tag) for tag in METRICS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for tag in METRICS:
        for step in sorted(set(on[tag]) & set(off[tag])):
            rows.append({"metric": tag, "step": step, "on": on[tag][step], "off": off[tag][step],
                         "off_minus_on": off[tag][step] - on[tag][step]})
    with (OUT_DIR / "comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "step", "on", "off", "off_minus_on"])
        w.writeheader()
        w.writerows(rows)

    summary: Dict = {"cutoff_step": CUTOFF_STEP, "noise_reference": f"{FAMILY} std over seeds 42/43/44 "
                     "(phase_3/results/seed_check/family_summary.csv; std of best-at-cutoff values)"}
    for name, fn in (("final", final_values), ("best_at_cutoff", best_at_cutoff)):
        s_on, v_on = fn(on_pts)
        s_off, v_off = fn(off_pts)
        entry = {"step_on": s_on, "step_off": s_off, "metrics": {}}
        for tag, (higher_better, _) in METRICS.items():
            a, b = v_on[tag], v_off[tag]
            if a is None or b is None:
                continue
            entry["metrics"][tag] = {"on": a, "off": b, "off_minus_on": b - a, "seed_std": std.get(tag),
                                     "z": (abs(b - a) / std[tag]) if tag in std else None,
                                     "verdict": verdict(b - a, std.get(tag), higher_better)}
        summary[name] = entry
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    make_figure(on, off, std)

    print(f"Wrote {OUT_DIR}")
    for name in ("final", "best_at_cutoff"):
        e = summary[name]
        print(f"\n{name}: on step {e['step_on']}, off step {e['step_off']}")
        for tag, m in e["metrics"].items():
            z = "" if m["z"] is None else f" (|d|/std = {m['z']:.2f})"
            print(f"  {tag:36s} on {m['on']:+.4f} off {m['off']:+.4f} d {m['off_minus_on']:+.4f}{z}  -> {m['verdict']}")


if __name__ == "__main__":
    main()
