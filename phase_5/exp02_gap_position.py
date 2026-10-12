#!/usr/bin/env python3
"""Phase 5 / Experiment 2 -- does the gap's position in the window differ between training and
inference, and does it matter? (supervisor Q3)

Training: each example is a 512-frame window and the gap is placed at a sampled position inside it
(activity-guided sampling, `ActivityAwareMaskedSpanDataset._choose_mask_span`).
Inference (`Trainer.inpaint_all_gaps`): the window is built around the gap with a context budget of
`max_len - gap_len - 16` frames split evenly, so the gap sits at the centre of a 493-496-frame window.

Two measurements per single-song checkpoint:
  1. the distribution of gap start positions the training sampler actually produces (20,000 draws);
  2. token NLL / top-1 accuracy of the SAME held-out gaps placed at different positions of a
     512-frame window (local start 8 ... 496), plus the exact inference-style window.
The gaps come from the run's held-out validation regions (never trained on); every shifted window
stays inside the region. No training, no decoding (token metrics only).

    uv run python phase_5/exp02_gap_position.py --device cuda:1
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "phase_0"))
sys.path.insert(0, str(REPO_ROOT / "phase_5"))
import common as c  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = REPO_ROOT / "phase_5" / "results" / "exp02_gap_position"
FIG = OUT / "figures"
DEFAULT_RUNS = ("baseline", "phase1_songs01", "phase2_regularization", "phase3_regularization_seed43")
POSITIONS = (8, 32, 64, 128, 192, 248, 320, 384, 448, 496)
GAPS_PER_REGION_LEN = 12
TRAIN_DRAWS = 20_000


@torch.no_grad()
def token_metrics(trainer, codes: torch.Tensor, w0: int, w1: int, g0: int, g1: int) -> Dict[str, float]:
    """Token NLL/top-1 accuracy of the gap [g0, g1) when predicted from window [w0, w1) of the codes (gap masked)."""
    x = codes[:, w0:w1].clone().long()
    T = w1 - w0
    mask = torch.zeros(T, dtype=torch.bool)
    mask[g0 - w0:g1 - w0] = True
    tgt = x[:, mask].clone()
    x[:, mask] = trainer.mask_token
    dev = trainer.device
    m = mask.unsqueeze(0).to(dev)
    seg, left, right = trainer._build_model_boundary_tensors(m)
    logits = trainer.model(x.unsqueeze(0).to(dev), seg, left, right)[0].float()
    logp = F.log_softmax(logits[:, mask.to(dev), :], dim=-1).cpu()
    nll = float(-logp.gather(-1, tgt.unsqueeze(-1)).mean())
    acc = float((logp.argmax(-1) == tgt).float().mean())
    return {"nll": nll, "acc_top1": acc}


def region_bands(grouped: Dict, holdout_ranges) -> List[Dict]:
    """Band (high/low activity) of each held-out region, by majority vote of its validation examples."""
    regions = []
    for rs, re_ in holdout_ranges:
        bands = [("high" if k.startswith("high") else "low") for k, exs in grouped.items() if "_len_" in k
                 for ex in exs if rs <= ex.window_start < re_]
        if bands:
            regions.append({"start": int(rs), "end": int(re_), "band": max(set(bands), key=bands.count)})
    return regions


def run_one(run: str, device: str) -> Dict:
    """For one run: sample the training sampler's gap positions, then score the same held-out gaps at shifted window positions and the inference-style window."""
    trainer = c.build_trainer(run, "best_val", device=device)
    trainer.model.eval()
    cfg = trainer.cfg
    T = int(cfg.seq_len)
    # 1. training gap-position distribution (the sampler's own draws)
    rng_state = (torch.random.get_rng_state(), np.random.get_state())
    c.set_seed(cfg.seed)
    starts, lens = [], []
    ds = trainer.dataset
    for _ in range(TRAIN_DRAWS):
        _, _, lm = ds[0]
        idx = torch.nonzero(lm).flatten()
        starts.append(int(idx[0])); lens.append(int(len(idx)))
    torch.random.set_rng_state(rng_state[0]); np.random.set_state(rng_state[1])

    # 2. same gaps at different positions
    grouped, holdout = c.build_real_validation_examples(trainer)
    regions = region_bands(grouped, holdout)
    rows = []
    for reg in regions:
        lo_g, hi_g = reg["start"] + max(POSITIONS), reg["end"] - T + min(POSITIONS)
        if hi_g <= lo_g:
            continue
        for L in (1, 2, 3, 4):
            for g0 in np.linspace(lo_g, hi_g - L, GAPS_PER_REGION_LEN).astype(int):
                g0 = int(g0); g1 = g0 + L
                for p in POSITIONS:
                    w0 = g0 - p
                    rows.append({"run": run, "band": reg["band"], "gap_start": g0, "mask_len": L, "window": "train_512",
                                 "position": p, "window_len": T, **token_metrics(trainer, trainer.codes, w0, w0 + T, g0, g1)})
                budget = max(0, cfg.max_len - L - 16)
                w0 = max(0, g0 - budget // 2)
                w1 = min(trainer.frames, g1 + (budget - budget // 2))
                rows.append({"run": run, "band": reg["band"], "gap_start": g0, "mask_len": L, "window": "inference",
                             "position": g0 - w0, "window_len": w1 - w0, **token_metrics(trainer, trainer.codes, w0, w1, g0, g1)})
    meta = {"run": run, "step": trainer.global_step, "seq_len": T, "max_len": cfg.max_len,
            "regions": regions, "train_positions": starts, "train_lens": lens}
    del trainer
    torch.cuda.empty_cache()
    return {"rows": rows, "meta": meta}


def mean_ci(v):
    """Mean and 95% CI half-width (NaN with one value)."""
    v = np.asarray(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def fig_train_distribution(metas: List[Dict]) -> None:
    """Histogram of gap start positions drawn by the training sampler."""
    ps.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    pos = np.concatenate([np.array(m["train_positions"]) for m in metas])
    ax.hist(pos, bins=np.arange(0, 513, 8), color=ps.CAT[0], edgecolor=ps.SURFACE, linewidth=1.0,
            weights=np.full(len(pos), 100.0 / len(pos)), label=f"training sampler ({len(pos):,} draws, {len(metas)} checkpoints)")
    ax.axvspan(244, 250, color=ps.CAT[1], alpha=0.9, label="inference: gap centred (local start 244-249 in a 493-496-frame window)")
    ax.set_xlabel("local start frame of the gap inside the 512-frame window")
    ax.set_ylabel("share of training examples (%)")
    ax.set_title("Where the gap sits in the window: training vs inference", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    ps.save(fig, FIG, "train_vs_inference_gap_position")


def fig_position_effect(rows: List[Dict]) -> None:
    """Figure of NLL/accuracy versus gap position in the window, per band."""
    ps.apply_style()
    import matplotlib.pyplot as plt
    runs = sorted({r["run"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)
    greys = ["#52514e", "#898781", "#6d6c68", "#a8a69f"]
    for j, band in enumerate(("high", "low")):
        for i, (metric, ylabel) in enumerate((("nll", "token NLL (nats), lower is better"), ("acc_top1", "top-1 accuracy, higher is better"))):
            ax = axes[i, j]
            per_run = {}
            for ri, run in enumerate(runs):
                ys = [np.mean([r[metric] for r in rows if r["run"] == run and r["band"] == band and r["window"] == "train_512"
                               and r["position"] == p]) for p in POSITIONS]
                per_run[run] = ys
                ax.plot(POSITIONS, ys, color=greys[ri % 4], linewidth=1.1, marker=ps.MARKERS[ri % 4], markersize=4,
                        label=run if (i == 0 and j == 0) else None)
            mean = np.mean(np.array(list(per_run.values())), axis=0)
            ax.plot(POSITIONS, mean, color=ps.CAT[0], linewidth=2.6, label="mean over checkpoints" if (i == 0 and j == 0) else None)
            inf = [r[metric] for r in rows if r["band"] == band and r["window"] == "inference"]
            pos = np.mean([r["position"] for r in rows if r["band"] == band and r["window"] == "inference"])
            m, ci = mean_ci(inf)
            ax.errorbar([pos], [m], yerr=[ci], fmt="D", color=ps.CAT[1], markersize=8, capsize=3, zorder=6,
                        label="inference-style window (centred, 493-496 frames)" if (i == 0 and j == 0) else None)
            ax.set_ylabel(ylabel, fontsize=8.5)
            if i == 0:
                ax.set_title(f"{band}-activity held-out gaps", loc="left")
            if i == 1:
                ax.set_xlabel("local start frame of the gap in a 512-frame window")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ps.save(fig, FIG, "quality_vs_gap_position")


def summarize(rows: List[Dict]) -> List[Dict]:
    """Per-band summary: edge positions versus the centre position, and the inference-style window versus the centre."""
    out = []
    for band in ("high", "low", "all"):
        sel = [r for r in rows if band == "all" or r["band"] == band]
        centre = {(r["run"], r["gap_start"], r["mask_len"]): r for r in sel if r["window"] == "train_512" and r["position"] == 248}
        for label, pred in (("edge_8", lambda r: r["window"] == "train_512" and r["position"] == 8),
                            ("edge_496", lambda r: r["window"] == "train_512" and r["position"] == 496),
                            ("inference_window", lambda r: r["window"] == "inference")):
            for metric in ("nll", "acc_top1"):
                d = [r[metric] - centre[(r["run"], r["gap_start"], r["mask_len"])][metric] for r in sel if pred(r)
                     and (r["run"], r["gap_start"], r["mask_len"]) in centre]
                m, ci = mean_ci(d)
                out.append({"band": band, "comparison": f"{label} minus centre_248 (512 window)", "metric": metric,
                            "n": len(d), "mean_diff": m, "ci95": ci})
    return out


def main() -> None:
    """CLI: run the gap-position experiment for each run and write CSV/JSON results and figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, metas = [], []
    for run in args.runs:
        print(f"=== {run} ===", flush=True)
        res = run_one(run, args.device)
        rows += res["rows"]; metas.append(res["meta"])
    with (OUT / "records.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summ = summarize(rows)
    with (OUT / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    hist = {m["run"]: {"positions_hist_8": np.histogram(m["train_positions"], bins=np.arange(0, 513, 8))[0].tolist(),
                       "share_within_32_of_centre": float(np.mean(np.abs(np.array(m["train_positions"]) - 248) <= 32)),
                       "regions": m["regions"], "step": m["step"]} for m in metas}
    (OUT / "train_position_distribution.json").write_text(json.dumps(hist, indent=1))
    fig_train_distribution(metas)
    fig_position_effect(rows)
    for s in summ:
        print(f"  {s['band']:4s} {s['comparison']:42s} {s['metric']:8s} n={s['n']:4d} d={s['mean_diff']:+.4f} ±{s['ci95']:.4f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
