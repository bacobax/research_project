#!/usr/bin/env python3
"""Phase 5 / Experiment 3 -- the remaining train/inference consistency checks (supervisor Q3).

  (a) inference iterations: `Trainer.inpaint_all_gaps` re-runs the model `inpaint_iters` times, feeding
      its own predictions back into the gap. The encoder-decoder model replaces the gap with mask
      tokens internally, so iterations should be a no-op: check that 1 and 10 iterations give identical
      tokens at the real gaps.
  (b) numeric precision: training and validation run under fp16 autocast, inference in fp32. Check how
      often the argmax token (what is actually output) changes, on held-out gaps and real gaps.
  (c) checkpoint selection: checkpoints are chosen by validation token NLL; the deliverable is audio.
      Correlate validation NLL with validation SI-SDR (gap-only) across the logged validation points.
  (d) gap content: the training sampler favours active regions; where do the 4 real gaps fall in the
      distribution of training-gap activity?

    uv run python phase_5/exp03_consistency_checks.py --device cuda:1
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase_0"))
sys.path.insert(0, str(REPO_ROOT / "phase_5"))
import common as c  # noqa: E402
import plot_style as ps  # noqa: E402
from phase_2.export_results import load_scalar_points  # noqa: E402

OUT = REPO_ROOT / "phase_5" / "results" / "exp03_consistency_checks"
FIG = OUT / "figures"
DEFAULT_RUNS = ("baseline", "phase1_songs01", "phase2_regularization", "phase3_regularization_seed43")
TB_RUNS = {  # single-song runs with a validation history
    "baseline": "outputs/runs/nuvole_bianche_short_gaps/nuvole_bianche_short_gaps_encoder_decoder/tb",
    "phase1_songs01": "outputs/runs/phase1_data_scaling/phase1_songs01/tb",
    "phase2_regularization": "outputs/runs/phase2_regularization/phase2_regularization/tb",
    "phase3_regularization_seed43": "outputs/runs/phase3_seed_check/phase3_regularization_seed43/tb",
    "phase3_regularization_seed44": "outputs/runs/phase3_seed_check/phase3_regularization_seed44/tb",
    "phase4_stft_off_seed43": "outputs/runs/phase4_stft_ablation/phase4_stft_off_seed43/tb",
}
TRAIN_DRAWS = 20_000


@torch.no_grad()
def logits_for(trainer, x: torch.Tensor, mask: torch.Tensor, fp16: bool) -> torch.Tensor:
    """Model logits for window x with the `mask` frames set to the mask token, optionally under fp16 autocast."""
    dev = trainer.device
    xb = x.clone().long()
    xb[:, mask] = trainer.mask_token
    m = mask.unsqueeze(0).to(dev)
    seg, left, right = trainer._build_model_boundary_tensors(m)
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=fp16):
        out = trainer.model(xb.unsqueeze(0).to(dev), seg, left, right)[0]
    return out.float().cpu()


def inference_window(trainer, g0: int, g1: int):
    """Window [w0, w1) Trainer.inpaint_all_gaps would use around gap [g0, g1): context budget max_len - gap_len - 16 split evenly."""
    L = g1 - g0
    budget = max(0, trainer.cfg.max_len - L - 16)
    w0 = max(0, g0 - budget // 2)
    w1 = min(trainer.frames, g1 + (budget - budget // 2))
    return w0, w1


def run_one(run: str, device: str) -> Dict:
    """For one run: (a) 1 vs many inpaint iterations, (b) fp16 vs fp32 argmax agreement, (c) NLL vs SI-SDR checkpoint-selection correlation, (d) real-gap activity vs training distribution."""
    trainer = c.build_trainer(run, "best_val", device=device)
    trainer.model.eval()
    res: Dict = {"run": run, "step": trainer.global_step}

    # (a) iterations: replicate inpaint_all_gaps' loop on the real gaps
    iters = []
    for gi, (g0, g1) in enumerate(trainer.gaps_f):
        w0, w1 = inference_window(trainer, g0, g1)
        x = trainer.codes[:, w0:w1].clone().long()
        mask = torch.zeros(w1 - w0, dtype=torch.bool); mask[g0 - w0:g1 - w0] = True
        x[:, mask] = trainer.mask_token
        preds = []
        for _ in range(10):
            lg = logits_for(trainer, x, mask, fp16=False)
            p = lg.argmax(-1)
            x[:, mask] = p[:, mask]       # exactly what inpaint_all_gaps does between iterations
            preds.append(p[:, mask].clone())
        iters.append({"gap": gi, "identical_1_vs_10": bool(torch.equal(preds[0], preds[-1])),
                      "changed_tokens_any_iteration": int(sum(int((preds[k] != preds[0]).sum()) for k in range(1, 10)))})
    res["iterations"] = iters

    # (b) precision: held-out examples + real gaps
    grouped, _ = c.build_real_validation_examples(trainer)
    agree, dnll, n_tok = 0, [], 0
    for key, exs in grouped.items():
        if "_len_" not in key:
            continue
        for ex in exs:
            mask = ex.loss_mask.bool()
            l32, l16 = logits_for(trainer, ex.y, mask, False), logits_for(trainer, ex.y, mask, True)
            tgt = ex.y[:, mask].long()
            a32, a16 = l32[:, mask].argmax(-1), l16[:, mask].argmax(-1)
            agree += int((a32 == a16).sum()); n_tok += a32.numel()
            n32 = -F.log_softmax(l32[:, mask], -1).gather(-1, tgt.unsqueeze(-1)).mean()
            n16 = -F.log_softmax(l16[:, mask], -1).gather(-1, tgt.unsqueeze(-1)).mean()
            dnll.append(float(n16 - n32))
    res["precision_heldout"] = {"argmax_agreement": agree / n_tok, "tokens": n_tok,
                                "mean_nll_fp16_minus_fp32": float(np.mean(dnll)), "max_abs_example_dnll": float(np.max(np.abs(dnll)))}
    agree_r, n_r = 0, 0
    for g0, g1 in trainer.gaps_f:
        w0, w1 = inference_window(trainer, g0, g1)
        mask = torch.zeros(w1 - w0, dtype=torch.bool); mask[g0 - w0:g1 - w0] = True
        x = trainer.codes[:, w0:w1]
        a32 = logits_for(trainer, x, mask, False)[:, mask].argmax(-1)
        a16 = logits_for(trainer, x, mask, True)[:, mask].argmax(-1)
        agree_r += int((a32 == a16).sum()); n_r += a32.numel()
    res["precision_real_gaps"] = {"argmax_agreement": agree_r / n_r, "tokens": n_r}

    # (d) training-gap activity vs real gaps
    c.set_seed(trainer.cfg.seed)
    ds = trainer.dataset
    if hasattr(ds, "pop_recent_metrics"):
        ds.pop_recent_metrics()
    acts = []
    for _ in range(TRAIN_DRAWS):
        ds[0]
        acts += [m["mask_mean_activity"] for m in ds.pop_recent_metrics()]
    act = trainer.activity_per_frame
    real_act = [float(np.mean(act[g0:g1])) for g0, g1 in trainer.gaps_f]
    res["activity"] = {"train_mask_activity": acts, "real_gap_activity": real_act,
                       "real_gap_percentile_in_training": [float(100 * np.mean(np.array(acts) <= a)) for a in real_act],
                       "low_thr": float(trainer.activity_low_thr), "high_thr": float(trainer.activity_high_thr)}
    del trainer
    torch.cuda.empty_cache()
    return res


def selection_alignment() -> List[Dict]:
    """Per run: validation NLL vs validation gap-only SI-SDR across all logged validation points."""
    from scipy.stats import spearmanr
    rows = []
    for run, tb in TB_RUNS.items():
        p = REPO_ROOT / tb
        if not p.exists():
            continue
        pts = load_scalar_points(p)
        nll = {x.step: x.value for x in pts if x.tag == "val/combined_loss"}
        sdr = {x.step: x.value for x in pts if x.tag == "val/si_sdr_db_gap_only"}
        steps = sorted(set(nll) & set(sdr))
        if len(steps) < 4:
            continue
        rho, pval = spearmanr([nll[s] for s in steps], [sdr[s] for s in steps])
        best_nll = min(steps, key=lambda s: nll[s])
        best_sdr = max(steps, key=lambda s: sdr[s])
        rows.append({"run": run, "n_points": len(steps), "spearman_nll_vs_sisdr": float(rho), "p_value": float(pval),
                     "step_best_nll": best_nll, "step_best_sisdr": best_sdr,
                     "sisdr_at_best_nll": sdr[best_nll], "best_sisdr": sdr[best_sdr],
                     "sisdr_lost_by_selecting_on_nll": sdr[best_sdr] - sdr[best_nll],
                     "steps": steps, "nll": [nll[s] for s in steps], "sisdr": [sdr[s] for s in steps]})
    return rows


def fig_selection(rows: List[Dict]) -> None:
    """Scatter of validation NLL against gap-only SI-SDR across logged validation points."""
    if not rows:
        print("skipping selection_nll_vs_audio: no TensorBoard validation history found (outputs/runs/**/tb)")
        return
    ps.apply_style()
    import matplotlib.pyplot as plt
    n = len(rows)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.1 * nrow), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    for ax, r in zip(axes.ravel(), rows):
        steps = np.array(r["steps"]); order = np.argsort(steps)
        x, y = np.array(r["nll"])[order], np.array(r["sisdr"])[order]
        ax.plot(x, y, color=ps.GRID, linewidth=1.0, zorder=1)
        sc = ax.scatter(x, y, c=steps[order], cmap=_blue_cmap(), s=36, zorder=2, edgecolors=ps.SURFACE, linewidths=0.8)
        bi = int(np.argmin(x)); bj = int(np.argmax(y))
        ax.scatter([x[bi]], [y[bi]], marker="s", s=90, facecolors="none", edgecolors=ps.CAT[1], linewidths=2, zorder=3,
                   label="selected (min val NLL)")
        ax.scatter([x[bj]], [y[bj]], marker="^", s=90, facecolors="none", edgecolors=ps.CAT[2], linewidths=2, zorder=3,
                   label="best val SI-SDR")
        ax.set_title(f"{r['run']}  (Spearman {r['spearman_nll_vs_sisdr']:+.2f})", fontsize=9.5, loc="left")
        ax.set_xlabel("validation token NLL (nats)")
        ax.set_ylabel("validation gap-only SI-SDR (dB)")
    spare = axes.ravel()[n:]
    for ax in spare:
        ax.set_visible(False)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    if len(spare):
        cax = spare[0].inset_axes([0.1, 0.45, 0.8, 0.07])
        spare[0].set_visible(True); spare[0].axis("off")
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        cb.ax.xaxis.set_major_locator(MaxNLocator(4))
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        cb.set_label("training step of the validation point", color=ps.INK_2, fontsize=8.5)
    else:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.suptitle("Checkpoint selection: validation NLL vs audio quality (one point per validation step)", color=ps.INK, fontsize=11, y=0.97)
    ps.save(fig, FIG, "selection_nll_vs_audio")


def _blue_cmap():
    """Ordered blue colormap (light to dark) for step ordering."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_ord", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def fig_activity(results: List[Dict]) -> None:
    """Histogram of training mask activity with the real gaps marked."""
    ps.apply_style()
    import matplotlib.pyplot as plt
    r = results[0]["activity"]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    a = np.array(r["train_mask_activity"])
    ax.hist(a, bins=40, color=ps.CAT[0], edgecolor=ps.SURFACE, linewidth=1.0, weights=np.full(len(a), 100 / len(a)),
            label="training gaps (sampler draws)")
    for k, (v, pct) in enumerate(zip(r["real_gap_activity"], r["real_gap_percentile_in_training"])):
        ax.axvline(v, color=ps.CAT[1], linewidth=2.0, linestyle=ps.LINESTYLES[k % 4],
                   label=f"real gap {k}: {pct:.0f}th percentile of training gaps")
    ax.axvline(r["low_thr"], color=ps.INK_2, linewidth=1, linestyle=(0, (1, 2)), label="low / high activity validation thresholds")
    ax.axvline(r["high_thr"], color=ps.INK_2, linewidth=1, linestyle=(0, (1, 2)))
    ax.set_xlabel("mean activity over the gap frames (normalised)")
    ax.set_ylabel("share of training gaps (%)")
    ax.set_title("Content of the gaps: training sampler vs the 4 real gaps", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ps.save(fig, FIG, "gap_activity_train_vs_real")


def main() -> None:
    """CLI: run the consistency checks for each run (or redraw figures with --figures-only) and write results."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from saved outputs + TensorBoard logs")
    args = ap.parse_args()
    if args.figures_only:
        checks = json.loads((OUT / "checks.json").read_text())
        draws = json.loads((OUT / "train_gap_activity_draws.json").read_text())
        results = [{"activity": {**r["activity_summary"], "train_mask_activity": draws[r["run"]]}} for r in checks]
        fig_selection(selection_alignment())
        fig_activity(results)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs:
        print(f"=== {run} ===", flush=True)
        results.append(run_one(run, args.device))
        r = results[-1]
        print(f"  iterations identical: {[x['identical_1_vs_10'] for x in r['iterations']]}; "
              f"fp16/fp32 argmax agreement held-out {r['precision_heldout']['argmax_agreement']:.4f}, "
              f"real gaps {r['precision_real_gaps']['argmax_agreement']:.4f}", flush=True)
    sel = selection_alignment()
    compact = [{k: v for k, v in r.items() if k != "activity"} | {"activity_summary": {k: v for k, v in r["activity"].items()
               if k != "train_mask_activity"}} for r in results]
    (OUT / "checks.json").write_text(json.dumps(compact, indent=1))
    (OUT / "train_gap_activity_draws.json").write_text(json.dumps({r["run"]: r["activity"]["train_mask_activity"] for r in results}))
    with (OUT / "selection_alignment.csv").open("w", newline="") as fh:
        keys = [k for k in sel[0] if k not in ("steps", "nll", "sisdr")]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(sel)
    fig_selection(sel)
    fig_activity(results)
    for s in sel:
        print(f"  {s['run']:30s} spearman {s['spearman_nll_vs_sisdr']:+.2f} (p={s['p_value']:.3f}) best-NLL step {s['step_best_nll']} "
              f"best-SI-SDR step {s['step_best_sisdr']} SI-SDR lost {s['sisdr_lost_by_selecting_on_nll']:+.2f} dB")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
