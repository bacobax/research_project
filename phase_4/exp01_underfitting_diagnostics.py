#!/usr/bin/env python3
"""
Phase 4 / Experiment 1 -- Causal underfitting diagnostics (H1-H4).

RERUN_PLAN.md Section 10 lists four hypotheses for *why* the model underfits even as training
data grows (train NLL rises 1.32 -> 4.51 nats from 1 -> 17 songs, not what a purely data-starved
model looks like), to be checked on the already eps-bug-clean `phase1_songs17` checkpoint before
deciding whether the fix is more data, more capacity, or a different architecture/objective:

  H1 - mode-averaging: do the K independent per-codebook heads collapse toward the context-free
       training-token marginal instead of committing to one plausible fill?
  H2 - RVQ-hierarchy structure: is failure uniform across codebooks/gap-positions, or
       concentrated in a way that points at the independent-head design specifically?
  H3 - context usage: does the decoder's cross-attention actually localize near the gap, or is
       it diffuse/uniform over the whole 512-frame window?
  H4 - convergence: is held-out loss still decreasing at the end of training, or has it
       plateaued (i.e. is "underfits" even a safe conclusion yet)?

Only run this against eps-bug-clean checkpoints (see RERUN_PLAN.md Section 3) -- a checkpoint
trained with ~1-in-5 steps hijacked by the STFT-loss epsilon bug cannot be trusted for these
mechanism-level diagnostics; the script prints a loud warning (not a hard refusal) if asked to.

Usage:
    uv run python phase_4/exp01_underfitting_diagnostics.py
    uv run python phase_4/exp01_underfitting_diagnostics.py --runs phase1_songs17 --diagnostics h1 h2
    uv run python phase_4/exp01_underfitting_diagnostics.py --diagnostics h4 --runs phase1_songs01 phase1_songs17
"""
import argparse
import contextlib
import csv
import gc
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_0"))
import common as c  # noqa: E402

from audio_infill.train import EncoderDecoderCodebookInfiller  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"  # phase_4's OWN file -- never c.append_summary_rows

RUN_NAMES = tuple(c.RUNS.keys())
DEFAULT_RUNS = ("phase1_songs17",)
DIAGNOSTIC_NAMES = ("h1", "h2", "h3", "h4")

# RERUN_PLAN.md Section 3 -- runs confirmed (empirically, from TensorBoard grad_norm histories)
# NOT to have a significant fraction of steps hijacked by the STFT-loss eps=1e-7 bug.
EPS_BUG_CLEAN = frozenset({
    "phase1_songs01", "phase1_songs17", "phase3_songs17_seed43", "phase3_songs17_seed44",
    # Phase B reruns (decoded_loss_log_eps=1e-3, grad_clip=1.0). Their grad_norm "spike" rate is a
    # late-training norm drift past the 100 threshold, not outliers (see phase_4/results/exp02).
    "phase1_songs04", "phase2_regularization", "phase3_regularization_seed43", "phase3_regularization_seed44",
    # STFT-ablation twin of phase3_regularization_seed43 (same eps/clip, decoded loss off).
    "phase4_stft_off_seed43",
})

BINS = 1024
UNIFORM_ENTROPY_NATS = math.log(BINS)
MARGINAL_FLOOR = 1e-12  # only inside KL/cross-entropy logs, to avoid -inf on unseen tokens
# |signed distance in frames| from the nearest gap edge -> right-open bin edges for H3.
DIST_BIN_EDGES = (1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 1_000_000)

SUMMARY_COLUMNS = list(c.SUMMARY_COLUMNS) + ["codebook", "gap_pos", "layer", "head", "distance_bin", "n"]

# H4: (metrics.csv path, `run` column value) per run name. Two schema variants on disk:
# phase_1/phase_2 = run,category,tag,step,wall_time_utc,value ; phase_3 seed_check = run,tag,step,wall_time_utc,value.
# Always read by column NAME (csv.DictReader) so both work without special-casing.
METRICS_SOURCES: Dict[str, Tuple[Path, str]] = {
    "phase1_songs01": (REPO_ROOT / "phase_1/results/metrics.csv", "phase1_songs01"),
    "phase1_songs04": (REPO_ROOT / "phase_1/results/metrics.csv", "phase1_songs04"),
    "phase1_songs17": (REPO_ROOT / "phase_1/results/metrics.csv", "phase1_songs17"),
    "phase2_regularization": (REPO_ROOT / "phase_2/results/metrics.csv", "phase2_regularization"),
    "phase3_songs17_seed43": (REPO_ROOT / "phase_3/results/seed_check/metrics.csv", "phase1_songs17_seed43"),
    "phase3_songs17_seed44": (REPO_ROOT / "phase_3/results/seed_check/metrics.csv", "phase1_songs17_seed44"),
    "phase3_regularization_seed43": (REPO_ROOT / "phase_3/results/seed_check/metrics.csv", "phase2_regularization_seed43"),
    "phase3_regularization_seed44": (REPO_ROOT / "phase_3/results/seed_check/metrics.csv", "phase2_regularization_seed44"),
}


def _save_figure(fig, base_path: Path) -> None:
    """Local copy of phase_2/export_results.py's helper, with phase_4's own Creator string."""
    fig.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "phase_4/exp01_underfitting_diagnostics.py", "CreationDate": None, "ModDate": None},
    )
    import matplotlib.pyplot as plt

    plt.close(fig)


def _style_axes(ax) -> None:
    """Light grid and no top/right spines for all figures."""
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def append_summary_rows(rows: List[Dict[str, Any]]) -> None:
    """Same csv.DictWriter logic as c.append_summary_rows, but targets phase_4's own summary.csv
    (with extra columns) -- calling c.append_summary_rows would pollute phase_0/results/ and drop
    the codebook/gap_pos/layer/head/distance_bin/n columns this phase needs.

    Re-running a run replaces that (experiment, run)'s previous rows instead of appending
    duplicates (plain appending made repeated invocations double-count in any later average)."""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    replaced = {(r["experiment"], r["run"]) for r in rows}
    kept: List[Dict[str, Any]] = []
    if SUMMARY_CSV.exists():
        with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if (r.get("experiment"), r.get("run")) not in replaced]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
        for row in rows:
            writer.writerow({**{col: "" for col in SUMMARY_COLUMNS}, **row})


def _check_disk_space(min_free_gib: float = 1.0) -> None:
    """Abort early if the root filesystem is nearly full (the script itself writes only a few MB)."""
    free_gib = shutil.disk_usage("/").free / (1024 ** 3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB free on / (need >= {min_free_gib} GiB). "
            "This script should only ever write a few MB, but refusing to start on a "
            "near-full disk rather than risk a partial write. See RERUN_PLAN.md Section 3."
        )
    print(f"[disk] {free_gib:.1f} GiB free on / -- OK")


def write_tidy_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    """Write rows as a tidy CSV with the given column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


# ---------------------------------------------------------------------------
# Shared pipeline: one trainer, one held-out example set, one prediction pass,
# feeding H1, H2 and H3.
# ---------------------------------------------------------------------------


@dataclass
class PredictionCell:
    """Per-(example, codebook, gap position) prediction record: target, top-1, top-5 hit, NLL, entropy and divergence from the training token marginal."""
    band: str
    mask_len: int
    example_id: int
    codebook: int
    gap_pos: int
    target: int
    pred_top1: int
    top5_hit: bool
    nll: float
    entropy: float
    top1_prob: float
    kl_to_train_marginal: float
    xent_to_train_marginal: float


def compute_token_marginals(
    trainer, grouped_examples: Dict[str, List[Any]], holdout_ranges: List[Tuple[int, int]]
) -> Dict[str, Dict[str, Any]]:
    """Per-codebook empirical token marginals, three sources:
      'train'           - pivot trainer.codes with holdout_ranges and trainer.gaps_f excluded,
                           PLUS every trainer.extra_samples[i]['codes'] in full.
      'holdout_all'     - every frame inside holdout_ranges (the held-out territory).
      'holdout_targets' - exactly the gap target tokens in grouped_examples (what H1 is scored
                           against: tiny, ~8*(1+2+3+4)=80 tokens/codebook/band at the exact set).
    Each entry: {"probs": [K,BINS] float64 rows summing to 1, "support_size": [K] int}.
    """
    K = trainer.K
    frames = trainer.frames
    codes_np = trainer.codes.cpu().numpy()  # [K, frames]

    excluded = np.zeros(frames, dtype=bool)
    for s, e in holdout_ranges:
        excluded[s:e] = True
    for g0, g1 in trainer.gaps_f:
        excluded[g0:g1] = True

    def _marginal_from_frames(codes_k: np.ndarray) -> Tuple[np.ndarray, int]:
        """Empirical token distribution and support size for one codebook's tokens."""
        counts = np.bincount(codes_k, minlength=BINS).astype(np.float64)
        support = int((counts > 0).sum())
        total = counts.sum()
        probs = counts / total if total > 0 else np.full(BINS, 1.0 / BINS)
        return probs, support

    train_probs = np.zeros((K, BINS), dtype=np.float64)
    train_support = np.zeros(K, dtype=np.int64)
    pivot_keep = ~excluded
    for k in range(K):
        counts = np.bincount(codes_np[k, pivot_keep], minlength=BINS).astype(np.float64)
        for extra in trainer.extra_samples:
            extra_codes_k = extra["codes"][k].cpu().numpy() if torch.is_tensor(extra["codes"]) else extra["codes"][k]
            counts += np.bincount(extra_codes_k, minlength=BINS)
        total = counts.sum()
        train_support[k] = int((counts > 0).sum())
        train_probs[k] = counts / total if total > 0 else np.full(BINS, 1.0 / BINS)

    holdout_probs = np.zeros((K, BINS), dtype=np.float64)
    holdout_support = np.zeros(K, dtype=np.int64)
    for k in range(K):
        holdout_probs[k], holdout_support[k] = _marginal_from_frames(codes_np[k, excluded])

    target_counts = np.zeros((K, BINS), dtype=np.float64)
    for key, examples in grouped_examples.items():
        if "_len_" not in key:
            continue
        for ex in examples:
            y = ex.y[:, ex.mask_start : ex.mask_start + ex.mask_len].numpy()  # [K, L]
            for k in range(K):
                target_counts[k] += np.bincount(y[k], minlength=BINS)
    target_support = (target_counts > 0).sum(axis=1).astype(np.int64)
    target_totals = target_counts.sum(axis=1, keepdims=True)
    target_probs = np.divide(
        target_counts, target_totals, out=np.full_like(target_counts, 1.0 / BINS), where=target_totals > 0
    )

    excluded_frame_count = int(excluded.sum())
    return {
        "train": {"probs": train_probs, "support_size": train_support, "excluded_pivot_frames": excluded_frame_count},
        "holdout_all": {"probs": holdout_probs, "support_size": holdout_support},
        "holdout_targets": {"probs": target_probs, "support_size": target_support},
    }


@torch.no_grad()
def _forward_gap_logits(trainer, example) -> torch.Tensor:
    """fp32, no autocast, batch of 1 -- mirrors Trainer._predict_validation_window (train.py
    ~2672-2680) but keeps logits instead of argmax-filled codes, and immediately slices to the
    gap frames only. EncoderDecoderCodebookInfiller.forward returns zeros outside the gap --
    softmax(zeros) is exactly uniform, which would masquerade as the mode-averaging signal H1
    looks for if this slice were forgotten."""
    assert not trainer.model.training, "model must be in eval() mode for diagnostic forward passes"
    device = trainer.device
    xb = example.x.unsqueeze(0).to(device)
    loss_mask = example.loss_mask.to(device)
    seg_ids, left_idx, right_idx = trainer._build_model_boundary_tensors(loss_mask)
    logits = trainer.model(xb, seg_ids, left_idx, right_idx)  # [1, K, T, BINS]

    ms, ml = example.mask_start, example.mask_len
    assert int(loss_mask.sum().item()) == ml
    assert bool(loss_mask[ms : ms + ml].all().item())
    outside = logits[0][:, ~loss_mask, :]
    if outside.numel() > 0:
        assert float(outside.abs().max().item()) == 0.0, "logits outside the gap are not exactly zero"
    return logits[0][:, ms : ms + ml, :].float()  # [K, L, BINS]


def collect_gap_predictions(
    trainer, grouped_examples: Dict[str, List[Any]], marginals: Dict[str, Dict[str, Any]]
) -> List[PredictionCell]:
    """Run the model on every validation example and record a PredictionCell for each gap token, using the training marginal for comparison."""
    train_probs = marginals["train"]["probs"]  # [K, BINS]
    train_log_probs = np.log(np.clip(train_probs, MARGINAL_FLOOR, None))
    cells: List[PredictionCell] = []

    for group_key, examples in grouped_examples.items():
        if "_len_" not in group_key:
            continue
        band, _, len_str = group_key.rpartition("_len_")
        mask_len = int(len_str)
        for example_id, example in enumerate(examples):
            gap_logits = _forward_gap_logits(trainer, example)  # [K, L, BINS]
            logp = F.log_softmax(gap_logits, dim=-1)
            p = logp.exp()
            entropy = -(p * logp).sum(-1)  # [K, L]
            target = example.y[:, example.mask_start : example.mask_start + example.mask_len].to(logp.device)
            nll = -logp.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)  # [K, L]
            pred_top1 = logp.argmax(-1)  # [K, L]
            top5_idx = logp.topk(5, dim=-1).indices  # [K, L, 5]
            top5_hit = (top5_idx == target.long().unsqueeze(-1)).any(-1)  # [K, L]
            top1_prob = p.max(-1).values  # [K, L]

            p_np = p.cpu().numpy()
            K, L, _ = p_np.shape
            for k in range(K):
                q = train_probs[k]
                logq = train_log_probs[k]
                kl_kl = (p_np[k] * (np.log(np.clip(p_np[k], MARGINAL_FLOOR, None)) - logq[None, :])).sum(-1)  # [L]
                xent_kl = -(p_np[k] * logq[None, :]).sum(-1)  # [L]
                for pos in range(L):
                    cells.append(
                        PredictionCell(
                            band=band,
                            mask_len=mask_len,
                            example_id=example_id,
                            codebook=k,
                            gap_pos=pos,
                            target=int(target[k, pos].item()),
                            pred_top1=int(pred_top1[k, pos].item()),
                            top5_hit=bool(top5_hit[k, pos].item()),
                            nll=float(nll[k, pos].item()),
                            entropy=float(entropy[k, pos].item()),
                            top1_prob=float(top1_prob[k, pos].item()),
                            kl_to_train_marginal=float(kl_kl[pos]),
                            xent_to_train_marginal=float(xent_kl[pos]),
                        )
                    )
            # Spot-check (safeguard #4): recompute one cell's NLL via plain indexing, independent
            # of any tensor-reshape convention, and assert it matches.
            if example_id == 0:
                k0, pos0 = 0, 0
                manual_nll = -math.log(max(float(p_np[k0, pos0, int(target[k0, pos0].item())]), 1e-30))
                assert abs(manual_nll - float(nll[k0, pos0].item())) < 1e-3, "NLL spot-check failed"
    return cells


# ---------------------------------------------------------------------------
# H1 -- mode-averaging
# ---------------------------------------------------------------------------


def run_h1(
    run_name: str, checkpoint_tag: str, step: int, cells: List[PredictionCell], marginals: Dict[str, Dict[str, Any]],
    eps_bug_clean: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """H1 (mode averaging): compare the model's gap predictions with the context-free training-token marginal. Returns (records, summary rows)."""
    train_probs = marginals["train"]["probs"]
    train_support = marginals["train"]["support_size"]
    holdout_target_probs = marginals["holdout_targets"]["probs"]

    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    by_band_codebook: Dict[Tuple[str, int], List[PredictionCell]] = defaultdict(list)
    for cell in cells:
        by_band_codebook[(cell.band, cell.codebook)].append(cell)

    for (band, k), group in sorted(by_band_codebook.items()):
        n = len(group)
        pred_entropies = np.array([x.entropy for x in group])
        nlls = np.array([x.nll for x in group])
        top1_probs = np.array([x.top1_prob for x in group])
        kls = np.array([x.kl_to_train_marginal for x in group])

        q = train_probs[k]
        train_marginal_entropy = float(-(q[q > 0] * np.log(q[q > 0])).sum())
        hq = holdout_target_probs[k]
        holdout_target_entropy = float(-(hq[hq > 0] * np.log(hq[hq > 0])).sum())

        mode_token = int(np.argmax(q))
        mode_token_train_freq = float(q[mode_token])
        p_argmax_eq_mode = float(np.mean([x.pred_top1 == mode_token for x in group]))
        p_target_eq_mode = float(np.mean([x.target == mode_token for x in group]))
        marginal_predictor_nll = float(-np.mean([math.log(max(q[x.target], MARGINAL_FLOOR)) for x in group]))

        distinct_pred = len({x.pred_top1 for x in group})
        distinct_target = len({x.target for x in group})

        record = {
            "run": run_name,
            "checkpoint": checkpoint_tag,
            "step": step,
            "band": band,
            "codebook": k,
            "n_cells": n,
            "eps_bug_clean": eps_bug_clean,
            "pred_entropy_mean": float(pred_entropies.mean()),
            "pred_entropy_std": float(pred_entropies.std()),
            "pred_entropy_p50": float(np.percentile(pred_entropies, 50)),
            "train_marginal_entropy": train_marginal_entropy,
            "holdout_target_entropy": holdout_target_entropy,
            "uniform_entropy": UNIFORM_ENTROPY_NATS,
            "entropy_ratio_to_marginal": float(pred_entropies.mean() / max(train_marginal_entropy, 1e-9)),
            "kl_to_train_marginal_mean": float(kls.mean()),
            "kl_to_train_marginal_p50": float(np.percentile(kls, 50)),
            "nll_mean": float(nlls.mean()),
            "marginal_predictor_nll": marginal_predictor_nll,
            "beats_marginal_predictor": bool(nlls.mean() < marginal_predictor_nll),
            "mode_token": mode_token,
            "mode_token_train_freq": mode_token_train_freq,
            "p_argmax_eq_mode": p_argmax_eq_mode,
            "p_target_eq_mode": p_target_eq_mode,
            "mode_fallback_lift": float(p_argmax_eq_mode / max(p_target_eq_mode, 1e-6)),
            "distinct_pred_tokens": distinct_pred,
            "distinct_target_tokens": distinct_target,
            "top1_prob_mean": float(top1_probs.mean()),
            "acc_top1": float(np.mean([x.pred_top1 == x.target for x in group])),
            "train_marginal_support_size": int(train_support[k]),
        }
        records.append(record)
        for metric in (
            "entropy_ratio_to_marginal", "kl_to_train_marginal_mean", "mode_fallback_lift", "nll_mean",
            "marginal_predictor_nll", "acc_top1",
        ):
            rows.append({
                "experiment": "exp01_h1_mode_averaging", "run": run_name, "checkpoint": checkpoint_tag,
                "step": step, "band": band, "method": "model", "metric": metric, "value": record[metric],
                "codebook": k, "n": n,
            })
    return records, rows


def plot_h1(records: List[Dict[str, Any]], figures_dir: Path, run_name: str, step: int) -> None:
    """Figure for H1: model vs marginal statistics per band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
    for band in sorted({r["band"] for r in records}):
        band_records = sorted((r for r in records if r["band"] == band), key=lambda r: r["codebook"])
        if not band_records:
            continue
        codebooks = [r["codebook"] for r in band_records]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        ax = axes[0][0]
        ax.bar(codebooks, [r["pred_entropy_mean"] for r in band_records], color="#4C72B0", alpha=0.7, label="predicted entropy (mean)")
        ax.scatter(codebooks, [r["train_marginal_entropy"] for r in band_records], marker="D", color="#DD8452", s=60, zorder=3, label="train marginal entropy")
        ax.scatter(codebooks, [r["holdout_target_entropy"] for r in band_records], marker="o", facecolors="none", edgecolors="#55A868", s=60, zorder=3, label="holdout-target entropy")
        ax.axhline(UNIFORM_ENTROPY_NATS, color="gray", linestyle="--", linewidth=1)
        ax.text(7.4, UNIFORM_ENTROPY_NATS, " uniform(1024)", va="bottom", ha="right", fontsize=7, color="gray")
        ax.set_xlabel("codebook index")
        ax.set_ylabel("entropy (nats)")
        ax.set_title("(a) predicted entropy vs. context-free marginal")
        ax.legend(fontsize=7, frameon=False)
        _style_axes(ax)

        ax = axes[0][1]
        ax.bar(codebooks, [r["kl_to_train_marginal_mean"] for r in band_records], color="#C44E52")
        ax.set_yscale("log")
        ax.axhspan(1e-9, 1e-2, color="gray", alpha=0.15)
        ax.text(0.5, 1.2e-2, "indistinguishable from the marginal", fontsize=7, color="gray")
        ax.set_xlabel("codebook index")
        ax.set_ylabel("KL(model || train marginal), nats")
        ax.set_title("(b) divergence from the marginal (0 = pure mode-averaging)")
        _style_axes(ax)

        ax = axes[1][0]
        width = 0.35
        xs = np.array(codebooks)
        ax.bar(xs - width / 2, [r["p_argmax_eq_mode"] for r in band_records], width, label="P(argmax == mode token)")
        ax.bar(xs + width / 2, [r["p_target_eq_mode"] for r in band_records], width, label="P(target == mode token)")
        ax.set_xlabel("codebook index")
        ax.set_ylabel("probability")
        ax.set_title("(c) collapse onto the globally most-frequent token")
        ax.legend(fontsize=7, frameon=False)
        _style_axes(ax)

        ax = axes[1][1]
        ax.plot(codebooks, [r["nll_mean"] for r in band_records], marker="o", label="model NLL")
        ax.plot(codebooks, [r["marginal_predictor_nll"] for r in band_records], marker="s", linestyle="--", label="context-free marginal predictor NLL")
        worse = [r["codebook"] for r in band_records if not r["beats_marginal_predictor"]]
        for wk in worse:
            ax.axvspan(wk - 0.5, wk + 0.5, color="#C44E52", alpha=0.12)
        ax.set_xlabel("codebook index")
        ax.set_ylabel("NLL (nats)")
        ax.set_title("(d) model NLL vs. context-free marginal predictor")
        ax.legend(fontsize=7, frameon=False)
        _style_axes(ax)

        fig.suptitle(
            f"H1 mode-averaging -- {run_name} best_val (step {step:,}), band={band}\n"
            "K=8 independent linear heads on a shared decoder state",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save_figure(fig, figures_dir / f"h1_mode_averaging_{band}")


# ---------------------------------------------------------------------------
# H2 -- per-codebook x per-gap-position breakdown
# ---------------------------------------------------------------------------


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_h2(
    run_name: str, checkpoint_tag: str, step: int, cells: List[PredictionCell], eps_bug_clean: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """H2 (RVQ hierarchy): NLL/accuracy broken down by codebook and gap position, merged over mask lengths. Returns (per-mask-length rows, merged rows)."""
    by_cell: Dict[Tuple[str, int, int, int], List[PredictionCell]] = defaultdict(list)
    for x in cells:
        by_cell[(x.band, x.mask_len, x.codebook, x.gap_pos)].append(x)

    per_mask_len: List[Dict[str, Any]] = []
    for (band, mask_len, k, pos), group in sorted(by_cell.items()):
        n = len(group)
        hits = sum(1 for x in group if x.pred_top1 == x.target)
        hits5 = sum(1 for x in group if x.top5_hit)
        nlls = [x.nll for x in group]
        lo1, hi1 = _wilson_interval(hits, n)
        per_mask_len.append({
            "run": run_name, "checkpoint": checkpoint_tag, "step": step, "aggregation": "per_mask_len",
            "band": band, "mask_len": mask_len, "codebook": k, "gap_pos": pos, "n": n,
            "acc_top1": hits / n, "acc_top1_wilson_lo": lo1, "acc_top1_wilson_hi": hi1,
            "acc_top5": hits5 / n, "nll_mean": float(np.mean(nlls)), "ppl": float(math.exp(min(np.mean(nlls), 20.0))),
            "entropy_mean": float(np.mean([x.entropy for x in group])),
        })

    # Merged over mask_len: (band, codebook, gap_pos), count-weighted.
    by_merged: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for rec in per_mask_len:
        by_merged[(rec["band"], rec["codebook"], rec["gap_pos"])].append(rec)
    merged: List[Dict[str, Any]] = []
    for (band, k, pos), group in sorted(by_merged.items()):
        n_total = sum(r["n"] for r in group)
        acc1 = sum(r["acc_top1"] * r["n"] for r in group) / n_total
        acc5 = sum(r["acc_top5"] * r["n"] for r in group) / n_total
        nll_mean = sum(r["nll_mean"] * r["n"] for r in group) / n_total
        hits1 = round(acc1 * n_total)
        lo1, hi1 = _wilson_interval(hits1, n_total)
        merged.append({
            "run": run_name, "checkpoint": checkpoint_tag, "step": step, "aggregation": "merged_over_mask_len",
            "band": band, "codebook": k, "gap_pos": pos, "n": n_total,
            "contributing_mask_lens": sorted(r["mask_len"] for r in group),
            "acc_top1": acc1, "acc_top1_wilson_lo": lo1, "acc_top1_wilson_hi": hi1,
            "acc_top5": acc5, "nll_mean": nll_mean, "ppl": float(math.exp(min(nll_mean, 20.0))),
        })

    rows: List[Dict[str, Any]] = []
    for rec in per_mask_len:
        for metric in ("acc_top1", "acc_top5", "nll_mean"):
            rows.append({
                "experiment": "exp01_h2_codebook_position", "run": run_name, "checkpoint": checkpoint_tag,
                "step": step, "band": rec["band"], "mask_len": rec["mask_len"], "method": "model",
                "metric": metric, "value": rec[metric], "codebook": rec["codebook"], "gap_pos": rec["gap_pos"],
                "n": rec["n"],
            })
    return per_mask_len + merged, rows


def h2_consistency_check(
    per_mask_len: List[Dict[str, Any]], reference_scalars: Optional[Dict[str, float]], historical_scalars: Dict[str, float],
) -> Dict[str, Any]:
    """Reduce per-(band,mask_len,codebook,gap_pos) cells up to group/band/combined using the
    exact pooling verified against train.py: group = unweighted mean over (codebook,gap_pos)
    cells; band = unweighted mean over the 4 mask_len groups; combined = mean of the two bands.
    Reported as deltas, never asserted equal (fp32 batch-of-1 here vs fp16 autocast batch-of-8
    in run_validation / the historical run)."""
    by_group: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for rec in per_mask_len:
        by_group[(rec["band"], rec["mask_len"])].append(rec)

    comparisons: List[Dict[str, Any]] = []
    band_losses: Dict[str, float] = {}
    band_accs: Dict[str, float] = {}
    for band in ("high_activity", "low_activity"):
        band_key = "high" if band == "high_activity" else "low"
        group_losses, group_accs = [], []
        for mask_len in (1, 2, 3, 4):
            group = by_group.get((band, mask_len), [])
            if not group:
                continue
            group_loss = float(np.mean([r["nll_mean"] for r in group]))
            group_acc = float(np.mean([r["acc_top1"] for r in group]))
            group_losses.append(group_loss)
            group_accs.append(group_acc)
            for tag, h2_val in ((f"val/{band_key}_len_{mask_len}_loss", group_loss), (f"val/{band_key}_len_{mask_len}_acc_top1", group_acc)):
                comparisons.append(_compare(tag, h2_val, reference_scalars, historical_scalars))
        if group_losses:
            band_losses[band_key] = float(np.mean(group_losses))
            band_accs[band_key] = float(np.mean(group_accs))
            comparisons.append(_compare(f"val/{band_key}_loss", band_losses[band_key], reference_scalars, historical_scalars))
            comparisons.append(_compare(f"val/{band_key}_acc_top1", band_accs[band_key], reference_scalars, historical_scalars))
    if "high" in band_losses and "low" in band_losses:
        combined = float(np.mean([band_losses["high"], band_losses["low"]]))
        comparisons.append(_compare("val/combined_loss", combined, reference_scalars, historical_scalars))

    max_abs_delta_ref = max((abs(c["delta_vs_reference"]) for c in comparisons if c["delta_vs_reference"] is not None), default=None)
    max_abs_delta_hist = max((abs(c["delta_vs_historical"]) for c in comparisons if c["delta_vs_historical"] is not None), default=None)
    ref_hist_agreement = max(
        (abs(c["delta_vs_reference"] - c["delta_vs_historical"]) for c in comparisons
         if c["delta_vs_reference"] is not None and c["delta_vs_historical"] is not None),
        default=None,
    )
    if max_abs_delta_ref is not None and max_abs_delta_ref > 0.05:
        if ref_hist_agreement is not None and ref_hist_agreement < 1e-3:
            note = (
                f"LARGER THAN EXPECTED (max |delta|={max_abs_delta_ref:.3f} nats): H2's fp32 "
                "batch-of-1 cells differ from BOTH the in-process fp16 reference and the "
                "historical fp16 metrics.csv, while reference and historical agree with each "
                f"other to <{ref_hist_agreement:.1e} nats. That rules out a batch-size or "
                "dataset-construction bug in H2's reduction -- this is a genuine fp16-autocast-"
                "vs-fp32 evaluation-precision effect on this checkpoint, large enough to be "
                "worth reporting on its own (comparable in scale to the data-scaling headline "
                "effect sizes), not something to wave away as noise."
            )
        else:
            note = (
                f"LARGER THAN EXPECTED (max |delta|={max_abs_delta_ref:.3f} nats) and the "
                "in-process reference does not closely match the historical metrics.csv value "
                "either -- investigate before trusting this consistency check (possible dataset "
                "reconstruction or environment mismatch, not simply fp16 vs fp32)."
            )
    else:
        note = "Deltas are within the ~1e-2 nat range expected from fp16-vs-fp32 evaluation precision."
    return {
        "comparisons": comparisons,
        "max_abs_delta_vs_reference": max_abs_delta_ref,
        "max_abs_delta_vs_historical": max_abs_delta_hist,
        "reference_vs_historical_agreement": ref_hist_agreement,
        "note": note,
    }


def _compare(tag: str, h2_value: float, reference_scalars: Optional[Dict[str, float]], historical_scalars: Dict[str, float]) -> Dict[str, Any]:
    """Compare an H2-recomputed loss with the reference and historically logged scalar for `tag`."""
    ref_val = reference_scalars.get(tag) if reference_scalars else None
    hist_val = historical_scalars.get(tag)
    return {
        "tag": tag,
        "h2_reduction": h2_value,
        "in_process_reference": ref_val,
        "historical_metrics_csv": hist_val,
        "delta_vs_reference": (h2_value - ref_val) if ref_val is not None else None,
        "delta_vs_historical": (h2_value - hist_val) if hist_val is not None else None,
    }


def plot_h2(per_mask_len: List[Dict[str, Any]], merged: List[Dict[str, Any]], figures_dir: Path, run_name: str, step: int) -> None:
    """Figure for H2: error structure across codebooks and gap positions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
    bands = ["high_activity", "low_activity"]
    mask_lens = [1, 2, 3, 4]

    for metric, cmap, higher_is_worse in (("acc_top1", "Blues", False), ("nll_mean", "Oranges", True)):
        fig = plt.figure(figsize=(13, 6))
        gs = GridSpec(2, 4, width_ratios=mask_lens, figure=fig, wspace=0.35, hspace=0.45)
        vals_all = [r[metric] for r in per_mask_len]
        vmin, vmax = min(vals_all), max(vals_all)
        im = None
        for row, band in enumerate(bands):
            for col, mask_len in enumerate(mask_lens):
                ax = fig.add_subplot(gs[row, col])
                grid = np.full((8, mask_len), np.nan)
                for r in per_mask_len:
                    if r["band"] == band and r["mask_len"] == mask_len:
                        grid[r["codebook"], r["gap_pos"]] = r[metric]
                im = ax.pcolormesh(grid, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=2)
                for k in range(8):
                    for pos in range(mask_len):
                        v = grid[k, pos]
                        frac = (v - vmin) / max(vmax - vmin, 1e-9)
                        ink = "white" if frac > 0.6 else "black"
                        ax.text(pos + 0.5, k + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=6, color=ink)
                ax.set_xticks(np.arange(mask_len) + 0.5)
                ax.set_xticklabels(range(mask_len), fontsize=7)
                ax.set_yticks(np.arange(8) + 0.5)
                ax.set_yticklabels(range(8), fontsize=7)
                ax.invert_yaxis()
                if col == 0:
                    ax.set_ylabel(f"{band}\ncodebook", fontsize=8)
                if row == 1:
                    ax.set_xlabel("gap position", fontsize=8)
                ax.set_title(f"mask_len={mask_len}", fontsize=8)
        fig.colorbar(im, ax=fig.get_axes(), label=metric, fraction=0.02, pad=0.02)
        fig.suptitle(
            f"H2 per-codebook x per-gap-position {metric} -- {run_name} best_val (step {step:,})\n"
            "every cell n=8 held-out examples",
            fontsize=10,
        )
        _save_figure(fig, figures_dir / f"h2_heatmap_by_mask_len_{metric}")

    for metric, cmap in (("acc_top1", "Blues"), ("nll_mean", "Oranges")):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
        vals_all = [r[metric] for r in merged]
        vmin, vmax = min(vals_all), max(vals_all)
        im = None
        for ax, band in zip(axes, bands):
            grid = np.full((8, 4), np.nan)
            n_grid = np.zeros((8, 4), dtype=int)
            for r in merged:
                if r["band"] == band:
                    grid[r["codebook"], r["gap_pos"]] = r[metric]
                    n_grid[r["codebook"], r["gap_pos"]] = r["n"]
            im = ax.pcolormesh(grid, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=2)
            for k in range(8):
                for pos in range(4):
                    v = grid[k, pos]
                    frac = (v - vmin) / max(vmax - vmin, 1e-9)
                    ink = "white" if frac > 0.6 else "black"
                    ax.text(pos + 0.5, k + 0.5, f"{v:.2f}\nn={n_grid[k, pos]}", ha="center", va="center", fontsize=6, color=ink)
            ax.set_xticks(np.arange(4) + 0.5)
            ax.set_xticklabels(range(4), fontsize=8)
            ax.set_yticks(np.arange(8) + 0.5)
            ax.set_yticklabels(range(8), fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("gap position (merged over mask_len >= pos+1)")
            ax.set_title(band, fontsize=9)
        axes[0].set_ylabel("codebook")
        fig.colorbar(im, ax=axes, label=metric, fraction=0.025, pad=0.02)
        fig.suptitle(
            f"H2 merged {metric} -- {run_name} best_val (step {step:,})\n"
            "gap position p pools all mask_len > p groups; n shown per cell",
            fontsize=10,
        )
        _save_figure(fig, figures_dir / f"h2_heatmap_merged_{metric}")


# ---------------------------------------------------------------------------
# H3 -- cross-attention vs. context distance
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def cross_attention_capture(model: EncoderDecoderCodebookInfiller, layers: Sequence[int]):
    """Yields store: Dict[int, torch.Tensor], filled on the next forward pass with
    store[layer] = [1, n_heads, gap_len, ctx_len] (average_attn_weights=False).

    nn.TransformerDecoderLayer._mha_block hardcodes need_weights=False when calling
    self.multihead_attn(...), so a plain forward hook sees output[1] is None. Fix: a forward
    PRE-hook (with_kwargs=True) that mutates kwargs to force need_weights=True and
    average_attn_weights=False, paired with a forward hook reading output[1]. All handles are
    removed in `finally` so an exception mid-forward can never leave the model permanently
    instrumented (which would silently change later H1/H2 forward-pass numerics)."""
    store: Dict[int, torch.Tensor] = {}
    handles = []

    def make_pre_hook():
        """Build a forward pre-hook that forces the attention module to return (unaveraged, per-head) weights."""
        def pre_hook(module, args, kwargs):
            """Request per-head attention weights from the module."""
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return args, kwargs

        return pre_hook

    def make_hook(layer_idx):
        """Build a forward hook capturing the cross-attention weights of layer `layer_idx`."""
        def hook(module, inputs, output):
            """Store the module's attention weights (fails loudly if the pre-hook did not take effect)."""
            attn = output[1]
            assert attn is not None, (
                "cross-attention weights are None -- the need_weights pre-hook did not take "
                f"(torch={torch.__version__}); check nn.TransformerDecoderLayer._mha_block"
            )
            n_heads = module.num_heads
            assert attn.dim() == 4 and attn.shape[0] == 1 and attn.shape[1] == n_heads, (
                f"unexpected attention shape {tuple(attn.shape)}"
            )
            store[layer_idx] = attn.detach().float().cpu()

        return hook

    try:
        for i in layers:
            mha = model.dec.layers[i].multihead_attn
            handles.append(mha.register_forward_pre_hook(make_pre_hook(), with_kwargs=True))
            handles.append(mha.register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in handles:
            h.remove()
        for i in layers:
            mha = model.dec.layers[i].multihead_attn
            assert not mha._forward_pre_hooks, "cross-attention hook cleanup failed"


def run_h3(
    trainer, run_name: str, checkpoint_tag: str, step: int, grouped_examples: Dict[str, List[Any]],
    eps_bug_clean: bool, examples_per_band: int = 2, mask_len: int = 4, layers: Sequence[int] = (0, 1, 2, 3),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """H3 (context usage): record the decoder cross-attention weights per layer and summarise attention by distance from the gap. Returns (rows, summary, aux data for plotting)."""
    assert not trainer.model.training
    T = trainer.cfg.seq_len
    maps: Dict[Tuple[str, int], Dict[int, np.ndarray]] = {}
    example_meta: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for band in ("high_activity", "low_activity"):
        key = f"{band}_len_{mask_len}"
        examples = grouped_examples.get(key, [])[:examples_per_band]
        for ex_id, example in enumerate(examples):
            gap_start = example.mask_start
            gap_end = gap_start + example.mask_len
            left_ctx_len = gap_start
            right_ctx_len = T - gap_end
            C = left_ctx_len + right_ctx_len
            example_meta.append({
                "band": band, "example_id": ex_id, "window_start": example.window_start,
                "mask_start": gap_start, "mask_mean_activity": example.mask_mean_activity,
                "left_ctx_len": left_ctx_len, "right_ctx_len": right_ctx_len,
            })

            with cross_attention_capture(trainer.model, layers) as store:
                _ = _forward_gap_logits(trainer, example)
            maps[(band, ex_id)] = {i: store[i][0].numpy() for i in layers}  # [n_heads, L, C]

            # memory column j -> original frame f(j); signed distance d(j) from nearest gap edge.
            col_idx = np.arange(C)
            frame_idx = np.where(col_idx < left_ctx_len, col_idx, gap_end + (col_idx - left_ctx_len))
            dist = np.where(
                frame_idx < gap_start, frame_idx - gap_start, frame_idx - gap_end + 1
            )  # negative = left of gap, positive = right of gap
            assert len(set(frame_idx.tolist())) == C
            assert not set(range(gap_start, gap_end)) & set(frame_idx.tolist())

            ctx_bins = len(DIST_BIN_EDGES)
            for layer_idx, attn in maps[(band, ex_id)].items():  # attn: [n_heads, L, C]
                n_heads, L, _ = attn.shape
                ctx_entropy_ceiling = math.log(C)
                for head in range(n_heads):
                    for q_row in range(L):
                        a = attn[head, q_row]  # [C], sums to 1
                        ent = float(-(a[a > 0] * np.log(a[a > 0])).sum())
                        com = float((a * np.abs(dist)).sum())
                        mass_left = float(a[dist < 0].sum())
                        mass_right = float(a[dist > 0].sum())
                        rows.append({
                            "experiment": "exp01_h3_cross_attention", "run": run_name, "checkpoint": checkpoint_tag,
                            "step": step, "band": band, "method": "model", "metric": "attn_entropy_ratio",
                            "value": ent / ctx_entropy_ceiling, "layer": layer_idx, "head": head,
                        })
                # Binned, averaged over heads and query rows for this example.
                abs_dist = np.abs(dist)
                mean_attn = attn.mean(axis=(0, 1))  # [C]
                prev_edge = 0
                for edge in DIST_BIN_EDGES:
                    in_bin = (abs_dist >= prev_edge) & (abs_dist < edge)
                    n_cols = int(in_bin.sum())
                    if n_cols > 0:
                        mean_w = float(mean_attn[in_bin].mean())
                        rows.append({
                            "experiment": "exp01_h3_cross_attention", "run": run_name, "checkpoint": checkpoint_tag,
                            "step": step, "band": band, "method": "model", "metric": "mean_weight_by_distance",
                            "value": mean_w, "layer": layer_idx, "distance_bin": f"[{prev_edge},{edge})",
                            "n": n_cols,
                        })
                    prev_edge = edge

    records = [{
        "run": run_name, "checkpoint": checkpoint_tag, "step": step, "eps_bug_clean": eps_bug_clean,
        "examples": example_meta, "seq_len": T, "mask_len": mask_len, "layers": list(layers),
    }]
    aux = {"maps": maps, "example_meta": example_meta}
    return records, rows, aux


def plot_h3(aux: Dict[str, Any], figures_dir: Path, run_name: str, step: int) -> None:
    """Figure for H3: cross-attention maps and attention mass versus distance from the gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
    maps = aux["maps"]
    example_meta = aux["example_meta"]
    if not maps:
        return
    layers = sorted({layer for m in maps.values() for layer in m.keys()})
    bands = sorted({band for band, _ in maps.keys()})
    meta_by_key = {(m["band"], m["example_id"]): m for m in example_meta}

    # --- main: attention weight vs signed distance, one panel per (layer, band) ---
    fig, axes = plt.subplots(len(layers), len(bands), figsize=(11, 2.8 * len(layers)), sharex=True)
    if len(layers) == 1:
        axes = axes.reshape(1, -1)
    for row, layer in enumerate(layers):
        for col, band in enumerate(bands):
            ax = axes[row][col]
            examples = [k for k in maps if k[0] == band]
            for band_key, ex_id in examples:
                attn = maps[(band_key, ex_id)][layer]  # [n_heads, L, C]
                meta = meta_by_key[(band_key, ex_id)]
                left_ctx_len, right_ctx_len = meta["left_ctx_len"], meta["right_ctx_len"]
                C = left_ctx_len + right_ctx_len
                col_idx = np.arange(C)
                dist = np.where(col_idx < left_ctx_len, col_idx - left_ctx_len, col_idx - left_ctx_len + 1)
                order = np.argsort(dist)
                mean_over_heads_rows = attn.mean(axis=(0, 1))[order]
                ax.plot(dist[order], mean_over_heads_rows, alpha=0.8, label=f"ex{ex_id}")
                ax.axhline(1.0 / C, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xscale("symlog", linthresh=4)
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(band, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"layer {layer}\nmean attn weight", fontsize=8)
            if row == len(layers) - 1:
                ax.set_xlabel("signed distance from gap edge (frames)")
            ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
            _style_axes(ax)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"H3 cross-attention vs. distance -- {run_name} best_val (step {step:,})\ndashed line = uniform (1/context_len)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, figures_dir / "h3_attention_vs_distance")

    # --- concentration summary: cumulative mass vs. |distance|, per layer x band ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    for layer in layers:
        for band in bands:
            examples = [k for k in maps if k[0] == band]
            if not examples:
                continue
            curves = []
            C_ref = None
            for band_key, ex_id in examples:
                attn = maps[(band_key, ex_id)][layer]
                meta = meta_by_key[(band_key, ex_id)]
                left_ctx_len, right_ctx_len = meta["left_ctx_len"], meta["right_ctx_len"]
                C = left_ctx_len + right_ctx_len
                C_ref = C
                col_idx = np.arange(C)
                dist = np.where(col_idx < left_ctx_len, col_idx - left_ctx_len, col_idx - left_ctx_len + 1)
                mean_w = attn.mean(axis=(0, 1))
                order = np.argsort(np.abs(dist))
                cum = np.cumsum(mean_w[order])
                curves.append((np.abs(dist)[order], cum))
            if curves:
                xs, cum = curves[0]
                ax.plot(xs + 1, cum, label=f"layer {layer} / {band}", alpha=0.85)
    if C_ref:
        xs_ref = np.arange(1, C_ref + 1)
        ax.plot(xs_ref, np.minimum(2 * xs_ref, C_ref) / C_ref, color="gray", linestyle="--", label="uniform")
    ax.set_xscale("log")
    ax.set_xlabel("|distance| from gap edge (frames)")
    ax.set_ylabel("cumulative attention mass")
    ax.set_title("(a) cumulative mass vs. distance")
    ax.legend(fontsize=6, frameon=False)
    _style_axes(ax)

    ax = axes[1]
    width = 0.35
    xs = np.arange(len(layers))
    for i, band in enumerate(bands):
        ratios = []
        for layer in layers:
            vals = []
            for band_key, ex_id in [k for k in maps if k[0] == band]:
                attn = maps[(band_key, ex_id)][layer]
                meta = meta_by_key[(band_key, ex_id)]
                C = meta["left_ctx_len"] + meta["right_ctx_len"]
                for head in range(attn.shape[0]):
                    for q_row in range(attn.shape[1]):
                        a = attn[head, q_row]
                        ent = -(a[a > 0] * np.log(a[a > 0])).sum()
                        vals.append(ent / math.log(C))
            ratios.append(np.mean(vals) if vals else np.nan)
        ax.bar(xs + (i - 0.5) * width, ratios, width, label=band)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("attention entropy / ln(context_len)")
    ax.set_title("(b) attention entropy ratio (1.0 = uniform)")
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)

    ax = axes[2]
    for i, band in enumerate(bands):
        coms = []
        for layer in layers:
            vals = []
            for band_key, ex_id in [k for k in maps if k[0] == band]:
                attn = maps[(band_key, ex_id)][layer]
                meta = meta_by_key[(band_key, ex_id)]
                left_ctx_len, right_ctx_len = meta["left_ctx_len"], meta["right_ctx_len"]
                C = left_ctx_len + right_ctx_len
                col_idx = np.arange(C)
                dist = np.where(col_idx < left_ctx_len, col_idx - left_ctx_len, col_idx - left_ctx_len + 1)
                mean_w = attn.mean(axis=(0, 1))
                vals.append(float((mean_w * np.abs(dist)).sum()))
            coms.append(np.mean(vals) if vals else np.nan)
        ax.plot(xs, coms, marker="o", label=band)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set_ylabel("center-of-mass |distance| (frames)")
    ax.set_title("(c) attention center of mass")
    ax.legend(fontsize=7, frameon=False)
    _style_axes(ax)

    fig.suptitle(f"H3 attention concentration -- {run_name} best_val (step {step:,})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, figures_dir / "h3_attention_concentration")

    # --- per-example qualitative attention maps ---
    for (band, ex_id), layer_maps in maps.items():
        meta = meta_by_key[(band, ex_id)]
        left_ctx_len = meta["left_ctx_len"]
        n_heads, L, C = next(iter(layer_maps.values())).shape
        fig, axes2 = plt.subplots(len(layers), n_heads, figsize=(2.6 * n_heads, 2.3 * len(layers)))
        vmax = max(m.max() for m in layer_maps.values())
        im = None
        for row, layer in enumerate(layers):
            attn = layer_maps[layer]
            for head in range(n_heads):
                ax = axes2[row][head] if len(layers) > 1 else axes2[head]
                im = ax.imshow(attn[head], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
                ax.axvline(left_ctx_len - 0.5, color="red", linestyle=":", linewidth=1)
                if row == 0:
                    ax.set_title(f"head {head}", fontsize=8)
                if head == 0:
                    ax.set_ylabel(f"layer {layer}\ngap pos", fontsize=7)
        fig.colorbar(im, ax=fig.get_axes(), fraction=0.02, pad=0.02)
        fig.suptitle(
            f"H3 attention map -- {run_name} {band} ex{ex_id} (step {step:,})\n"
            f"mask_start={meta['mask_start']}, left_ctx={meta['left_ctx_len']}, right_ctx={meta['right_ctx_len']}, "
            f"activity={meta['mask_mean_activity']:.3f}",
            fontsize=9,
        )
        _save_figure(fig, figures_dir / f"h3_attention_map_{band}_ex{ex_id}")


# ---------------------------------------------------------------------------
# H4 -- training convergence check (CSV only, never the raw TensorBoard file)
# ---------------------------------------------------------------------------


def _read_metrics_csv(path: Path, run_alias: str, tag: str) -> List[Tuple[int, float]]:
    """Read (step, value) points for one run alias and tag from an exported metrics CSV."""
    points: List[Tuple[int, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["run"] == run_alias and row["tag"] == tag:
                points.append((int(row["step"]), float(row["value"])))
    points.sort(key=lambda p: p[0])
    return points


def run_h4(
    run_name: str, checkpoint_step: Optional[int], eps_bug_clean: bool, tail_points: int = 20,
) -> Tuple[Dict[str, Any], List[Tuple[int, float]], List[Tuple[int, float]]]:
    """H4 (convergence): minimum, last value and tail slope of held-out loss from the logged metrics, versus the checkpoint step."""
    if run_name not in METRICS_SOURCES:
        return {"run": run_name, "error": "no metrics.csv source registered for this run"}, [], []
    path, alias = METRICS_SOURCES[run_name]
    if not path.exists():
        return {"run": run_name, "error": f"metrics.csv not found: {path}"}, [], []

    val_points = _read_metrics_csv(path, alias, "val/combined_loss")
    train_points = _read_metrics_csv(path, alias, "train/avg_nll")
    if not val_points:
        return {"run": run_name, "error": f"no val/combined_loss rows for alias={alias} in {path}"}, [], []

    steps = np.array([s for s, _ in val_points])
    values = np.array([v for _, v in val_points])
    min_idx = int(np.argmin(values))

    tail = val_points[-tail_points:]
    tail_steps = np.array([s for s, _ in tail], dtype=np.float64)
    tail_vals = np.array([v for _, v in tail])
    slope, intercept = np.polyfit(tail_steps, tail_vals, 1)
    pred = slope * tail_steps + intercept
    ss_res = float(((tail_vals - pred) ** 2).sum())
    ss_tot = float(((tail_vals - tail_vals.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    slope_per_100k = float(slope * 100_000)

    checkpoint_value = None
    if checkpoint_step is not None:
        exact = [v for s, v in val_points if s == checkpoint_step]
        checkpoint_value = exact[0] if exact else None

    train_val_gap = None
    if checkpoint_step is not None and train_points:
        train_exact = [v for s, v in train_points if s == checkpoint_step]
        if train_exact and checkpoint_value is not None:
            train_val_gap = float(checkpoint_value - train_exact[0])

    plateaued = bool(abs(slope_per_100k) < 0.01 and values[-1] <= values.min() * 1.01)
    record = {
        "run": run_name, "eps_bug_clean": eps_bug_clean,
        "n_points": len(val_points), "first_step": int(steps[0]), "first_value": float(values[0]),
        "last_step": int(steps[-1]), "last_value": float(values[-1]),
        "min_step": int(steps[min_idx]), "min_value": float(values[min_idx]),
        "checkpoint_step": checkpoint_step, "checkpoint_value": checkpoint_value,
        "best_step_fraction_of_total": float(steps[min_idx] / steps[-1]) if steps[-1] > 0 else None,
        "tail_slope_nats_per_100k_steps": slope_per_100k, "tail_fit_r2": r2, "tail_n_points": len(tail),
        "plateau_criterion": "|OLS slope over last N val points| * 1e5 < 0.01 nats AND last value within 1% of running min",
        "plateaued": plateaued, "still_decreasing": not plateaued,
        "train_val_gap_at_checkpoint": train_val_gap,
    }
    return record, val_points, train_points


def plot_h4(records: List[Dict[str, Any]], curves: Dict[str, Tuple[List, List]], figures_dir: Path) -> None:
    """Figure for H4: held-out loss curves with the evaluated checkpoint marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.get_cmap("tab10")

    for i, rec in enumerate(records):
        if "error" in rec:
            continue
        run_name = rec["run"]
        val_points, train_points = curves[run_name]
        vs, vv = zip(*val_points)
        ax = axes[0]
        ax.plot(vs, vv, marker="o", markersize=3, color=colors(i), label=f"{run_name} val/combined_loss")
        if train_points and len(records) == 1:
            ts, tv = zip(*train_points)
            ax.plot(ts, tv, linestyle="--", color=colors(i), alpha=0.6, label=f"{run_name} train/avg_nll")
        if rec["checkpoint_step"] is not None and rec["checkpoint_value"] is not None:
            ax.scatter([rec["checkpoint_step"]], [rec["checkpoint_value"]], marker="*", s=180, color=colors(i), zorder=5)
            ax.annotate(f"best_val (step {rec['checkpoint_step']:,})", (rec["checkpoint_step"], rec["checkpoint_value"]), fontsize=7, xytext=(5, 8), textcoords="offset points")
        ax.set_xscale("log")

        ax2 = axes[1]
        cutoff = vs[-1] * 0.75
        tail_v = [(s, v) for s, v in val_points if s >= cutoff]
        ts2, vv2 = zip(*tail_v)
        ax2.plot(ts2, vv2, marker="o", markersize=3, color=colors(i), label=f"{run_name}")
        tail = rec["tail_n_points"]
        tail_pts = val_points[-tail:]
        tx = np.array([s for s, _ in tail_pts], dtype=np.float64)
        slope = rec["tail_slope_nats_per_100k_steps"] / 100_000
        intercept = tail_pts[-1][1] - slope * tail_pts[-1][0]
        ax2.plot(tx, slope * tx + intercept, linestyle=":", color=colors(i))
        ax2.axhline(rec["min_value"], color=colors(i), linestyle="--", linewidth=0.7, alpha=0.5)
        ax2.annotate(
            f"slope={rec['tail_slope_nats_per_100k_steps']:.4f} nats/100k (R2={rec['tail_fit_r2']:.2f})",
            (tx[0], vv2[0]), fontsize=7,
        )

    axes[0].set_xlabel("step (log scale)")
    axes[0].set_ylabel("loss (nats)")
    axes[0].set_title("(a) full run")
    axes[0].legend(fontsize=7, frameon=False)
    _style_axes(axes[0])
    axes[1].set_xlabel("step")
    axes[1].set_title("(b) tail (last 25%), with OLS fit")
    axes[1].legend(fontsize=7, frameon=False)
    _style_axes(axes[1])

    fig.suptitle("H4 training convergence check (source: already-extracted metrics.csv, not raw TensorBoard)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_figure(fig, figures_dir / "h4_convergence")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    """CLI: runs, checkpoint tag, which diagnostics (h1-h4) and H3 sampling options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS), choices=list(RUN_NAMES))
    parser.add_argument("--checkpoint", default="best_val", choices=("best_val", "best", "final"))
    parser.add_argument("--diagnostics", nargs="+", default=list(DIAGNOSTIC_NAMES), choices=list(DIAGNOSTIC_NAMES))
    parser.add_argument("--h3-examples-per-band", type=int, default=2)
    parser.add_argument("--h3-mask-len", type=int, default=4)
    parser.add_argument(
        "--device", default=None,
        help="Override the run's own config device (e.g. cuda:0, cuda:1, cpu). "
        "Default: whatever the run's YAML config says.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the selected diagnostics on each requested run, writing CSV/JSON results and figures under phase_4/results."""
    args = parse_args(argv)
    _check_disk_space()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    h1_all_records, h1_all_rows = [], []
    h2_all_records, h2_all_rows, h2_all_consistency = [], [], []
    h3_all_records, h3_all_rows = [], []
    h4_all_records: List[Dict[str, Any]] = []
    h4_curves: Dict[str, Tuple[List, List]] = {}

    # H4 first: pure CSV, no GPU, fails fast if a metrics file moved.
    if "h4" in args.diagnostics:
        for run_name in args.runs:
            eps_clean = run_name in EPS_BUG_CLEAN
            checkpoint_step = 950_000 if run_name == "phase1_songs17" and args.checkpoint == "best_val" else None
            record, val_points, train_points = run_h4(run_name, checkpoint_step, eps_clean)
            h4_all_records.append(record)
            h4_curves[run_name] = (val_points, train_points)
            print(f"[h4] {run_name}: {record}")
        c.save_json(RESULTS_DIR / "h4_convergence.json", h4_all_records)
        write_tidy_csv(
            RESULTS_DIR / "h4_curve.csv",
            [{"run": r, "tag": "val/combined_loss", "step": s, "value": v} for r, (vp, _) in h4_curves.items() for s, v in vp]
            + [{"run": r, "tag": "train/avg_nll", "step": s, "value": v} for r, (_, tp) in h4_curves.items() for s, v in tp],
            ["run", "tag", "step", "value"],
        )
        if any("error" not in r for r in h4_all_records):
            plot_h4([r for r in h4_all_records if "error" not in r], h4_curves, FIGURES_DIR)

    needs_trainer = any(d in args.diagnostics for d in ("h1", "h2", "h3"))
    run_metadata: List[Dict[str, Any]] = []
    if needs_trainer:
        for run_name in args.runs:
            eps_clean = run_name in EPS_BUG_CLEAN
            if not eps_clean:
                print(
                    f"\n*** WARNING: {run_name} is NOT in the eps-bug-clean set (RERUN_PLAN.md Section 3). "
                    "H1/H2/H3 results from this checkpoint cannot be trusted as architecture/objective "
                    "evidence -- an observed pattern may be an artifact of the noisy training trajectory, "
                    "not a real property of the model. Proceeding anyway since explicitly requested. ***\n"
                )

            print(f"\n=== building trainer for {run_name} ({args.checkpoint}) ===")
            trainer = c.build_trainer(run_name, args.checkpoint, device=args.device)
            step = trainer.global_step
            assert isinstance(trainer.model, EncoderDecoderCodebookInfiller), (
                "H3's attention hooks assume EncoderDecoderCodebookInfiller"
            )
            assert trainer.K == 8 and trainer.bins == BINS
            run_metadata.append({
                "run": run_name, "checkpoint": args.checkpoint, "step": step, "eps_bug_clean": eps_clean,
                "d_model": trainer.cfg.d_model, "n_layers": trainer.cfg.n_layers, "n_heads": trainer.cfg.n_heads,
                "seq_len": trainer.cfg.seq_len, "mask_len_min": trainer.cfg.mask_len_min, "mask_len_max": trainer.cfg.mask_len_max,
                "best_val_loss": getattr(trainer, "best_val_loss", None), "torch_version": torch.__version__,
                "device": str(trainer.device),
            })

            grouped_examples, holdout_ranges = c.build_real_validation_examples(trainer)
            n_groups = {k: len(v) for k, v in grouped_examples.items() if "_len_" in k}
            print(f"  held-out groups: {n_groups}")

            marginals = compute_token_marginals(trainer, grouped_examples, holdout_ranges)

            reference_scalars = None
            if "h2" in args.diagnostics:
                trainer.cfg.validation_audio_metrics_enabled = False
                dataloaders, group_specs = c.build_validation_dataloaders(trainer, grouped_examples)
                trainer.validation_audio_targets = {}
                c.run_validation_safe(trainer, dataloaders, group_specs, step=step)
                trainer.model.eval()  # run_validation ends with model.train() -- MUST re-assert eval
                assert not trainer.model.training
                reference_scalars = trainer.writer.as_dict()

            cells = None
            if "h1" in args.diagnostics or "h2" in args.diagnostics:
                print("  collecting fp32 gap predictions (H1/H2 shared pass)...")
                cells = collect_gap_predictions(trainer, grouped_examples, marginals)
                print(f"  collected {len(cells)} (codebook, gap_pos) cells")

            if "h1" in args.diagnostics:
                h1_records, h1_rows = run_h1(run_name, args.checkpoint, step, cells, marginals, eps_clean)
                h1_all_records.extend(h1_records)
                h1_all_rows.extend(h1_rows)
                plot_h1(h1_records, FIGURES_DIR, run_name, step)

            if "h2" in args.diagnostics:
                h2_records, h2_rows = run_h2(run_name, args.checkpoint, step, cells, eps_clean)
                h2_all_records.extend(h2_records)
                h2_all_rows.extend(h2_rows)
                per_mask_len = [r for r in h2_records if r["aggregation"] == "per_mask_len"]
                merged = [r for r in h2_records if r["aggregation"] == "merged_over_mask_len"]
                historical_record = next((r for r in h4_all_records if r.get("run") == run_name), None)
                historical_scalars = {}
                if run_name in METRICS_SOURCES:
                    path, alias = METRICS_SOURCES[run_name]
                    if path.exists():
                        with open(path, newline="", encoding="utf-8") as f:
                            for row in csv.DictReader(f):
                                if row["run"] == alias and row["tag"].startswith("val/") and int(row["step"]) == step:
                                    historical_scalars[row["tag"]] = float(row["value"])
                consistency = h2_consistency_check(per_mask_len, reference_scalars, historical_scalars)
                consistency["run"] = run_name
                h2_all_consistency.append(consistency)
                print(f"  h2 consistency: max_abs_delta_vs_reference={consistency['max_abs_delta_vs_reference']}, "
                      f"max_abs_delta_vs_historical={consistency['max_abs_delta_vs_historical']}")
                print(f"  h2 consistency note: {consistency['note']}")
                plot_h2(per_mask_len, merged, FIGURES_DIR, run_name, step)

            if "h3" in args.diagnostics:
                assert not trainer.model.training
                h3_records, h3_rows, h3_aux = run_h3(
                    trainer, run_name, args.checkpoint, step, grouped_examples, eps_clean,
                    examples_per_band=args.h3_examples_per_band, mask_len=args.h3_mask_len,
                )
                h3_all_records.extend(h3_records)
                h3_all_rows.extend(h3_rows)
                plot_h3(h3_aux, FIGURES_DIR, run_name, step)

            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        c.save_json(RESULTS_DIR / "run_metadata.json", run_metadata)

    if h1_all_records:
        c.save_json(RESULTS_DIR / "h1_mode_averaging.json", h1_all_records)
    if h2_all_records:
        c.save_json(RESULTS_DIR / "h2_codebook_position.json", h2_all_records)
        c.save_json(RESULTS_DIR / "h2_consistency_check.json", h2_all_consistency)
        write_tidy_csv(
            RESULTS_DIR / "h2_cells.csv", h2_all_records,
            ["run", "checkpoint", "step", "aggregation", "band", "mask_len", "codebook", "gap_pos", "n",
             "acc_top1", "acc_top1_wilson_lo", "acc_top1_wilson_hi", "acc_top5", "nll_mean", "ppl"],
        )
    if h3_all_records:
        c.save_json(RESULTS_DIR / "h3_cross_attention.json", h3_all_records)
        write_tidy_csv(
            RESULTS_DIR / "h3_attention_by_distance.csv",
            [r for r in h3_all_rows if r["metric"] == "mean_weight_by_distance"],
            ["run", "band", "layer", "distance_bin", "n", "value"],
        )

    all_rows = h1_all_rows + h2_all_rows + h3_all_rows
    if all_rows:
        append_summary_rows(all_rows)

    print(f"\nWrote outputs under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
