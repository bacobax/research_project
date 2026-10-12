"""Phase 4, exp02 -- which loss term produces the large gradient norms?

The Phase B' re-audit (docs/RERUN_PLAN.md Sec 3 method, new aggregated logs) found that the four
eps-fixed reruns still show 10-21% of steps with grad_norm > 100 or non-finite, versus ~0.045%
(all non-finite, no finite spikes) for the clean 17-song / 1-song runs. The aggregated logs cannot
say which loss term is responsible, so this script measures it directly on saved checkpoints:
for N training batches (model in train mode, fp16 autocast, exactly as the training loop), it
backpropagates three terms *separately* and records the pre-clip gradient norm of each

  token   : token cross-entropy only
  dec_l1  : decoded-domain waveform-L1 term only   (weight = decoded_loss_weight * l1_weight)
  dec_stft: decoded-domain multi-resolution STFT term only

plus the norm of their sum (= what the optimizer sees, ignoring the grad scaler), and whether any
gradient is non-finite. It does this at several checkpoints per run (early / mid / final, see
CHECKPOINT_PLAN) so the *trend over training* is visible, and records the L2 norm of the model's
weights at each checkpoint (exploratory: tests whether growing gradient norm co-occurs with
shrinking weights, e.g. under weight decay). No new training, nothing written outside
phase_4/results/exp02/.

    uv run python phase_4/exp02_gradient_spike_sources.py                      # full plan, 100 batches each
    uv run python phase_4/exp02_gradient_spike_sources.py --runs phase2_regularization --batches 200
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_0"))
import common as c  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "phase_4" / "results" / "exp02"
SPIKE = 100.0
TERMS = ("token", "dec_l1", "dec_stft")

# Checkpoint files (inside each run's checkpoints/ dir) to evaluate. Intermediate step_*.pt files
# only exist for the runs that were launched with a checkpoint cadence; the 17-song runs keep just
# best_val/best/final, so for them best_val is the "early" point where it is early (seed 43/44:
# step 20k/280k, see phase_3 per_run_at_cutoff.csv) and final is the end of training.
CHECKPOINT_PLAN = {
    # Phase 0 single-song runs (original, eps=1e-7 default, grad_clip 1.5): killed manually, so no
    # final.pt -- `latest` is the last checkpoint saved before the kill; no intermediate step_*.pt.
    # NOTE baseline/latest.pt (step 162k) has non-finite weights (the run diverged after its best
    # checkpoint at step 78k), so best.pt is used as its late point instead.
    "baseline": ["best_val.pt", "best.pt"],
    "bigmodel": ["best_val.pt", "latest.pt"],
    "phase1_songs01": ["best_val.pt", "final.pt"],
    "phase1_songs04": ["best_val.pt", "step_100000.pt", "step_200000.pt", "final.pt"],
    "phase1_songs17": ["best_val.pt", "final.pt"],
    "phase2_regularization": ["best_val.pt", "step_150000.pt", "step_300000.pt", "step_450000.pt", "final.pt"],
    "phase3_songs17_seed43": ["best_val.pt", "final.pt"],
    "phase3_songs17_seed44": ["best_val.pt", "final.pt"],
    "phase3_regularization_seed43": ["best_val.pt", "step_150000.pt", "final.pt"],
    "phase3_regularization_seed44": ["best_val.pt", "step_150000.pt", "final.pt"],
}


def _flat_grad(loss: torch.Tensor, params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Gradient of `loss` w.r.t. params as one flat float vector (zeros for unused params), keeping the graph for further terms."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(grads, params)])


def _pct(a: np.ndarray, q: float) -> float:
    """Percentile of an array (NaN if empty)."""
    return float(np.percentile(a, q)) if a.size else float("nan")


def measure_checkpoint(trainer, run_name: str, ckpt_file: str, ckpt_path: Path, n_batches: int) -> Dict:
    """Backprop the token, decoded-L1 and decoded-STFT terms separately over n_batches training batches of one checkpoint and summarise their pre-clip gradient norms."""
    trainer.load_checkpoint(str(ckpt_path))
    cfg = trainer.cfg
    assert trainer.decoded_loss_enabled, "decoded loss is disabled for this run; nothing to attribute"
    model = trainer.model
    model.train()  # dropout on, as in training
    params = [p for p in model.parameters() if p.requires_grad]
    param_l2 = float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in params)))
    use_amp = trainer.device.type == "cuda"
    w_l1, w_stft = cfg.decoded_loss_waveform_l1_weight, cfg.decoded_loss_stft_weight

    rows = []
    it = iter(trainer.dataloader)
    for b in range(n_batches):
        try:
            x, y, loss_mask = next(it)
        except StopIteration:
            it = iter(trainer.dataloader)
            x, y, loss_mask = next(it)
        x, y, loss_mask = x.to(trainer.device), y.to(trainer.device), loss_mask.to(trainer.device).bool()
        seg, left, right = trainer._build_model_boundary_tensors(loss_mask)
        with torch.autocast(device_type=trainer.device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, seg, left, right)
            token_loss = trainer._compute_loss(logits, y, loss_mask)
            # Isolate each decoded sub-term by zeroing the other's weight (read at call time).
            cfg.decoded_loss_waveform_l1_weight, cfg.decoded_loss_stft_weight = w_l1, 0.0
            l1_loss, _ = trainer._compute_decoded_domain_loss(x, logits, y, loss_mask)
            cfg.decoded_loss_waveform_l1_weight, cfg.decoded_loss_stft_weight = 0.0, w_stft
            stft_loss, m = trainer._compute_decoded_domain_loss(x, logits, y, loss_mask)
            cfg.decoded_loss_waveform_l1_weight, cfg.decoded_loss_stft_weight = w_l1, w_stft
        g = {"token": _flat_grad(token_loss, params),
             "dec_l1": _flat_grad(l1_loss, params),
             "dec_stft": _flat_grad(stft_loss, params)}
        total = g["token"] + g["dec_l1"] + g["dec_stft"]
        row = {"batch": b, "mask_len": int(loss_mask.sum(dim=1).max().item()),
               "token_loss": float(token_loss), "dec_stft_loss": float(stft_loss),
               "stft_raw": m["decoded_loss_stft"], "sc_raw": m["decoded_loss_spectral_convergence"],
               "logmag_raw": m["decoded_loss_log_magnitude"],
               "total_norm": float(total.norm()), "total_finite": bool(torch.isfinite(total).all())}
        for k, v in g.items():
            row[f"{k}_norm"] = float(v.norm())
            row[f"{k}_finite"] = bool(torch.isfinite(v).all())
        rows.append(row)
        del g, total, logits
        if (b + 1) % 50 == 0:
            print(f"  [{run_name}/{ckpt_file}] {b + 1}/{n_batches}", flush=True)

    summary: Dict = {"run": run_name, "checkpoint": ckpt_file, "step": trainer.global_step,
                     "eps": cfg.decoded_loss_log_eps, "grad_clip": cfg.grad_clip, "n_batches": len(rows),
                     "weight_decay": cfg.weight_decay, "param_l2": param_l2}
    for k in (*TERMS, "total"):
        norms = np.array([r[f"{k}_norm"] for r in rows])
        fin = np.array([r[f"{k}_finite"] if k != "total" else r["total_finite"] for r in rows])
        nf = norms[fin]
        summary[k] = {"nonfinite_frac": float(1 - fin.mean()), "frac_gt_100": float((nf > SPIKE).mean()) if nf.size else 0.0,
                      "median": _pct(nf, 50), "p90": _pct(nf, 90), "p99": _pct(nf, 99), "max": float(nf.max()) if nf.size else float("nan")}
    # Which term drives the big total norms? Among batches with total > SPIKE or non-finite,
    # fraction where each term alone exceeds SPIKE (or is non-finite).
    spike_idx = [i for i, r in enumerate(rows) if (not r["total_finite"]) or r["total_norm"] > SPIKE]
    summary["n_spike_batches"] = len(spike_idx)
    summary["spike_attribution"] = {
        k: (sum(1 for i in spike_idx if (not rows[i][f"{k}_finite"]) or rows[i][f"{k}_norm"] > SPIKE) / len(spike_idx)) if spike_idx else None
        for k in TERMS}
    # Do spikes track near-silent targets (tiny raw STFT/log-mag values) or gap length?
    if spike_idx and len(spike_idx) < len(rows):
        spike_set = set(spike_idx)
        rest = [i for i in range(len(rows)) if i not in spike_set]
        for key in ("mask_len", "stft_raw", "logmag_raw", "sc_raw", "token_loss"):
            summary[f"mean_{key}_spike_vs_rest"] = [float(np.mean([rows[i][key] for i in spike_idx])),
                                                     float(np.mean([rows[i][key] for i in rest]))]
    torch.cuda.empty_cache()
    return {"summary": summary, "rows": rows}


SUMMARY_COLUMNS = ["run", "checkpoint", "step", "n_batches", "weight_decay", "param_l2",
                   "token_median", "token_p90", "dec_stft_median", "dec_stft_p90", "dec_stft_max",
                   "dec_l1_median", "total_median", "stft_over_token_median", "frac_total_gt_100",
                   "nonfinite_frac"]


def summary_row(sm: Dict) -> Dict:
    """Flatten one checkpoint's summary into a rounded CSV row."""
    return {"run": sm["run"], "checkpoint": sm["checkpoint"], "step": sm["step"], "n_batches": sm["n_batches"],
            "weight_decay": sm["weight_decay"], "param_l2": round(sm["param_l2"], 4),
            "token_median": round(sm["token"]["median"], 4), "token_p90": round(sm["token"]["p90"], 4),
            "dec_stft_median": round(sm["dec_stft"]["median"], 4), "dec_stft_p90": round(sm["dec_stft"]["p90"], 4),
            "dec_stft_max": round(sm["dec_stft"]["max"], 4), "dec_l1_median": round(sm["dec_l1"]["median"], 5),
            "total_median": round(sm["total"]["median"], 4),
            "stft_over_token_median": round(sm["dec_stft"]["median"] / sm["token"]["median"], 5),
            "frac_total_gt_100": sm["total"]["frac_gt_100"], "nonfinite_frac": sm["total"]["nonfinite_frac"]}


def make_figure(summaries: List[Dict]) -> None:
    """Plot gradient-norm trends per loss term across checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted({s["run"] for s in summaries})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.get_cmap("tab10")
    for k, run in enumerate(runs):
        pts = sorted((s for s in summaries if s["run"] == run), key=lambda s: s["step"])
        xs = [s["step"] for s in pts]
        col = cmap(k % 10)
        axes[0].plot(xs, [s["token"]["median"] for s in pts], "o-", color=col, label=run)
        axes[1].plot(xs, [s["dec_stft"]["median"] / s["token"]["median"] for s in pts], "o-", color=col, label=run)
        axes[2].plot(xs, [s["param_l2"] for s in pts], "o-", color=col, label=run)
    axes[0].set(title="Token-loss gradient norm (median, pre-clip)", xlabel="checkpoint step", ylabel="L2 norm", yscale="log")
    axes[0].axhline(1.0, color="gray", ls=":", lw=1)
    axes[0].text(0.02, 0.04, "dotted: grad_clip = 1.0", transform=axes[0].transAxes, fontsize=8, color="gray")
    axes[1].set(title="STFT-term norm / token-term norm (median)", xlabel="checkpoint step", yscale="log")
    axes[2].set(title="Model weight L2 norm", xlabel="checkpoint step")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1e4) if False else None
        ax.grid(alpha=0.3)
    axes[2].legend(fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"gradient_attribution.{ext}", dpi=160, metadata={"Creator": "phase_4/exp02"} if ext == "pdf" else None)
    plt.close(fig)


def main():
    """CLI: measure the planned checkpoints of each run and write results under phase_4/results/exp02."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(CHECKPOINT_PLAN), choices=list(CHECKPOINT_PLAN))
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: List[Dict] = []
    for run in args.runs:
        print(f"=== {run} ===", flush=True)
        trainer = c.build_trainer(run, "final" if "final" in c.RUNS[run]["checkpoints"] else "best_val", device=args.device)
        ckpt_dir = c.RUNS[run]["checkpoints"]["best_val"].parent
        results = []
        for f in CHECKPOINT_PLAN[run]:
            res = measure_checkpoint(trainer, run, f, ckpt_dir / f, args.batches)
            results.append(res)
            all_summaries.append(res["summary"])
            sm = res["summary"]
            print(f"  {f} step={sm['step']} token_med={sm['token']['median']:.2f} stft_med={sm['dec_stft']['median']:.2f} "
                  f"l1_med={sm['dec_l1']['median']:.3f} |w|={sm['param_l2']:.1f}", flush=True)
        (OUT_DIR / f"{run}.json").write_text(json.dumps(results, indent=1))
        del trainer
        torch.cuda.empty_cache()
    # Merge with any previously exported runs so a partial re-run doesn't drop the others.
    csv_path = OUT_DIR / "summary.csv"
    keep = []
    if csv_path.exists():
        with csv_path.open() as fh:
            keep = [r for r in csv.DictReader(fh) if r["run"] not in args.runs]
    rows = keep + [summary_row(s) for s in all_summaries]
    rows.sort(key=lambda r: (r["run"], int(r["step"])))
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    # Figure from the full merged table (re-read per-run JSONs).
    merged = []
    for jf in sorted(p for p in OUT_DIR.glob("*.json")):
        merged += [r["summary"] for r in json.loads(jf.read_text())]
    make_figure(merged)
    print(f"Wrote {csv_path} and figure under {OUT_DIR}")


if __name__ == "__main__":
    main()
