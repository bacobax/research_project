#!/usr/bin/env python3
"""Phase 4 -- build results/FINDINGS.md from the exported H1-H4 and exp02 result files.

Pure post-processing (no GPU, no training): reads phase_4/results/{h1_mode_averaging.json,
h2_cells.csv, h2_consistency_check.json, h3_attention_by_distance.csv, h4_convergence.json,
run_metadata.json, exp02/summary.csv} and writes one markdown file with a cross-run table per
diagnostic, so every number in the write-up can be regenerated:

    uv run python phase_4/make_summary.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "FINDINGS.md"
BANDS = ("high_activity", "low_activity")
FAR_BIN = "[257,1000000)"


def _load_json(name):
    return json.loads((RESULTS / name).read_text())


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def section_runs(meta):
    rows = [(m["run"], m["checkpoint"], f"{m['step']:,}", m["d_model"], m["n_layers"], m["seq_len"],
             f"{m['mask_len_min']}-{m['mask_len_max']}") for m in meta]
    return _table(["run", "checkpoint", "step", "d_model", "layers", "seq_len", "trained gap (frames)"], rows)


def section_h1(h1):
    rows = []
    for run in dict.fromkeys(r["run"] for r in h1):
        for band in BANDS:
            g = [r for r in h1 if r["run"] == run and r["band"] == band]
            rows.append((
                run, band.split("_")[0],
                f"{np.mean([r['pred_entropy_mean'] for r in g]):.2f}",
                f"{np.mean([r['entropy_ratio_to_marginal'] for r in g]):.2f}",
                f"{np.mean([r['nll_mean'] for r in g]):.2f}",
                f"{np.mean([r['marginal_predictor_nll'] for r in g]):.2f}",
                f"{sum(r['beats_marginal_predictor'] for r in g)}/{len(g)}",
                f"{np.mean([r['p_argmax_eq_mode'] for r in g]):.3f}",
                f"{np.mean([r['acc_top1'] for r in g]):.3f}",
            ))
    return _table(["run", "band", "pred. entropy (nats)", "pred. / train-marginal entropy", "model NLL",
                   "marginal-predictor NLL", "codebooks where model wins", "P(argmax = most frequent token)",
                   "top-1 acc"], rows)


def section_h2(cells):
    merged = [r for r in cells if r["aggregation"] == "merged_over_mask_len"]
    rows, slopes = [], []
    for run in dict.fromkeys(r["run"] for r in merged):
        for band in BANDS:
            nll, acc = {}, {}
            for k in range(8):
                g = [r for r in merged if r["run"] == run and r["band"] == band and int(r["codebook"]) == k]
                n = np.array([int(r["n"]) for r in g], dtype=float)
                nll[k] = float((np.array([float(r["nll_mean"]) for r in g]) * n).sum() / n.sum())
                acc[k] = float((np.array([float(r["acc_top1"]) for r in g]) * n).sum() / n.sum())
            slope = float(np.polyfit(range(8), [nll[k] for k in range(8)], 1)[0])
            slopes.append((band, slope))
            rows.append((run, band.split("_")[0], f"{nll[0]:.2f}", f"{nll[7]:.2f}", f"{slope:+.2f}",
                         f"{acc[0]:.3f}", f"{acc[7]:.3f}"))
    t = _table(["run", "band", "NLL codebook 1", "NLL codebook 8", "NLL slope / codebook", "top-1 codebook 1",
                "top-1 codebook 8"], rows)
    pos = {b: sum(1 for bb, s in slopes if bb == b and s > 0) for b in BANDS}
    tot = {b: sum(1 for bb, _ in slopes if bb == b) for b in BANDS}
    return t, pos, tot


def section_h3(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r["run"], r["layer"], r["distance_bin"])].append(float(r["value"]))
    out = []
    for run in dict.fromkeys(r["run"] for r in rows):
        cells = []
        for layer in ("0", "1", "2", "3"):
            near = np.mean(d[(run, layer, "[1,2)")])
            far = np.mean(d[(run, layer, FAR_BIN)])
            cells.append(f"{near / far:.0f}x")
        out.append((run, *cells))
    return _table(["run", "layer 0", "layer 1", "layer 2", "layer 3"], out)


def section_h4(h4):
    rows = []
    for r in h4:
        gap = r["train_val_gap_at_checkpoint"]
        rows.append((r["run"], f"{r['min_value']:.3f} @ {r['min_step']:,}", f"{r['last_value']:.3f} @ {r['last_step']:,}",
                     f"{r['last_value'] - r['min_value']:+.3f}", f"{r['tail_slope_nats_per_100k_steps']:+.3f}",
                     "-" if gap is None else f"{gap:.3f}", "yes" if r["plateaued"] else "no"))
    return _table(["run", "best val NLL @ step", "last val NLL @ step", "last - best", "tail slope (nats/100k)",
                   "train/val gap at checkpoint", "plateaued"], rows)


def section_fp(cons):
    def _f(x):
        return "-" if x is None else f"{x:.3f}"  # runs without an exported metrics.csv have no logged value

    rows = [(c["run"], _f(c["max_abs_delta_vs_reference"]), _f(c["max_abs_delta_vs_historical"])) for c in cons]
    return _table(["run", "max abs. difference, fp32 batch-1 vs fp16 batch-8 reference (nats)", "max abs. difference, fp32 vs logged training-time value (nats)"], rows)


def section_exp02(exp02):
    seen, rows = set(), []
    for r in exp02:
        key = (r["run"], r["step"])
        if key in seen:  # best_val.pt can be the same step as final.pt
            continue
        seen.add(key)
        rows.append((r["run"], f"{int(r['step']):,}", f"{float(r['token_median']):.1f}", f"{float(r['dec_stft_median']):.2f}",
                     f"{float(r['dec_l1_median']):.3f}", f"{100 * float(r['stft_over_token_median']):.1f}%",
                     f"{float(r['param_l2']):.0f}", r["weight_decay"]))
    ratios = [float(r["stft_over_token_median"]) for r in exp02]
    return _table(["run", "checkpoint step", "token-loss grad norm", "STFT-term grad norm", "waveform-L1 grad norm",
                   "STFT / token", "weight L2 norm", "weight decay"], rows), min(ratios), max(ratios)


def section_exp03():
    d = RESULTS / "exp03_stft_ablation"
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())
    rows = _load_csv(d / "comparison.csv")
    fin = summ["final"]
    t_rows = []
    for tag, m in fin["metrics"].items():
        z = "-" if m["z"] is None else f"{m['z']:.2f}"
        sd = "-" if m["seed_std"] is None else f"{m['seed_std']:.4g}"
        t_rows.append((tag, f"{m['on']:+.4f}", f"{m['off']:+.4f}", f"{m['off_minus_on']:+.4f}", sd, z, m["verdict"]))
    final_table = _table(["metric", "decoded loss ON", "decoded loss OFF", "OFF - ON", "seed std (family)",
                          "|diff| / std", "verdict"], t_rows)
    c_rows = []
    for tag in ("val/combined_loss", "val/high_acc_top1", "val/low_acc_top1", "val/si_sdr_db_gap_only"):
        if tag not in summ["final"]["metrics"] or summ["final"]["metrics"][tag]["seed_std"] is None:
            continue
        sd = summ["final"]["metrics"][tag]["seed_std"]
        diff = np.array([float(r["off_minus_on"]) for r in rows if r["metric"] == tag])
        c_rows.append((tag, len(diff), f"{diff.mean() / sd:+.2f}", f"{np.abs(diff).max() / sd:.2f}",
                       f"{np.mean(np.abs(diff) > sd):.2f}", f"{np.mean(np.abs(diff) > 2 * sd):.2f}"))
    curve_table = _table(["metric", "validation points", "mean diff (in std)", "max |diff| (in std)",
                          "fraction > 1 std", "fraction > 2 std"], c_rows)
    return final_table, curve_table


def main():
    meta = _load_json("run_metadata.json")
    h1 = _load_json("h1_mode_averaging.json")
    cells = _load_csv(RESULTS / "h2_cells.csv")
    cons = _load_json("h2_consistency_check.json")
    h3 = _load_csv(RESULTS / "h3_attention_by_distance.csv")
    h4 = [r for r in _load_json("h4_convergence.json") if "error" not in r]
    exp02 = _load_csv(RESULTS / "exp02" / "summary.csv")

    h2_table, pos, tot = section_h2(cells)
    e_table, rmin, rmax = section_exp02(exp02)
    lines = [
        "# Phase 4 findings (generated by `phase_4/make_summary.py`)",
        "",
        "All numbers below are regenerated from the files in this directory; do not edit by hand.",
        "Every diagnostic is evaluated on held-out validation examples of the key single-song material "
        "(Nuvole Bianche with carved gaps), gap lengths 1-4 frames, 8 examples per (activity band, gap length).",
        "",
        "## Checkpoints analysed (H1-H3)", "", section_runs(meta), "",
        "## H1 - do the per-codebook heads collapse to the marginal token distribution?", "",
        "Means over the 8 codebooks. 'Marginal-predictor NLL' is the NLL of a context-free predictor that always "
        "outputs the training-token marginal.", "", section_h1(h1), "",
        "## H2 - does quality degrade with codebook index?", "",
        f"NLL slope per codebook is positive (later codebooks worse) in {pos['high_activity']}/{tot['high_activity']} "
        f"high-activity and {pos['low_activity']}/{tot['low_activity']} low-activity run/band pairs. "
        "Count-weighted over gap lengths and positions; each cell has n=8.", "", h2_table, "",
        "## H3 - how concentrated is cross-attention near the gap?", "",
        "Ratio of the mean attention weight per context column at distance 1 frame from the gap vs. >=257 frames away "
        "(uniform attention would give 1x). 2 examples per band, gap length 4, averaged over heads.", "", section_h3(h3), "",
        "## H4 - convergence of held-out loss", "",
        "From the exported `metrics.csv` files (validation every 10k steps for the long runs). "
        "'Plateaued' = |tail slope| < 0.01 nats/100k steps and last value within 1% of the minimum.", "", section_h4(h4), "",
        "## Evaluation precision (fp16 batch-8 vs fp32 batch-1)", "",
        "Largest per-group difference between the fp32 batch-of-1 evaluation used by H1/H2 and the fp16-autocast "
        "evaluation used during training/model selection.", "", section_fp(cons), "",
        "## Exp02 - which loss term produces the gradient? (STFT evidence)", "",
        "Pre-clip gradient L2 norm of each loss term, back-propagated separately on 100 training batches per "
        "checkpoint (median over batches; model in train mode, fp16 autocast). The STFT term is weighted exactly as in "
        "training (`decoded_loss_weight` x `decoded_loss_stft_weight`).", "", e_table, "",
        f"Across all measured checkpoints the STFT term's gradient norm is {100 * rmin:.1f}%-{100 * rmax:.1f}% of the "
        "token-loss gradient norm (median over batches).", "",
    ]
    exp03 = section_exp03()
    if exp03 is not None:
        lines += [
            "## Exp03 - STFT-loss training ablation (decoded loss ON vs OFF)", "",
            "Matched twins: regularized single song, seed 43, 300k steps; the OFF run is identical except "
            "`decoded_loss_enabled: false`. ON = `phase3_regularization_seed43` (weight 0.1). Seed std = std of "
            "the regularization family's best-at-cutoff values over seeds 42/43/44. Verdict rule fixed in advance: "
            "|diff| < 1 std no detectable effect, 1-2 std inconclusive, > 2 std effect.", "",
            "Final step (300,000; also the best-validation step for both runs):", "", exp03[0], "",
            "Across all 12 validation points (every 25k steps):", "", exp03[1], "",
            "One seed per arm: effects below the seed spread cannot be resolved, so 'no detectable effect' is not "
            "'no effect'. Figure: `results/exp03_stft_ablation/stft_ablation.png`.", "",
        ]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
