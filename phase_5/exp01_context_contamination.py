#!/usr/bin/env python3
"""Phase 5 / Experiment 1 -- context contamination at gaps (supervisor Q3, train/inference consistency).

Training builds every example from tokens of CLEAN audio and lays a mask over the gap frames, so the
context the model sees is exactly what the clean audio encodes to. A real gap is different: the gap's
samples are zeroed in the audio and the whole gapped file is encoded, so
  (1) the EnCodec encoder (convolutions + LSTM) sees the silence, and frames next to the gap can
      encode to different tokens ("encoder contamination");
  (2) real gaps are not aligned to the 320-sample frame grid, and the gap -> frame mapping rounds the
      start and end (`Trainer._resolve_gap_frames`), so frames that are partly silent can sit OUTSIDE
      the mask: they are given to the model as context and kept unchanged in the output
      ("mask coverage").
This script measures both, on the 4 real gaps of the key song and on the held-out validation gaps
(statistics), for the single-song checkpoints. All audio comes from the original ungapped source
(data/nuvole_bianche.mp3), so clean and gapped versions differ ONLY by the zeroed samples.

Conditions (same window, same gap, scored against the clean audio):
  A  clean      : tokens of clean audio, rounded mask                 (= how training/validation look)
  B  aligned    : gap frames zeroed exactly (frame-aligned), rounded mask (encoder contamination only)
  C  deployed   : zeroed span not frame-aligned (real gaps: the real spans; synthetic: random sub-frame
                  offsets that round back to the same mask), rounded mask   (= what inference does today)
  E  pred-only  : C's predictions written into the CLEAN context (isolates the effect on predictions
                  from the effect of keeping contaminated context tokens in the output)
  D  coverage   : as C, but the mask covers every frame the zeroing touches (floor/ceil)
  Z  unfilled   : as C, no inpainting (the gapped audio itself), reference only

Outputs: phase_5/results/exp01_context_contamination/ (CSV/JSON) and figures/ (PNG+PDF).

    uv run python phase_5/exp01_context_contamination.py --device cuda:1
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "phase_0"))
sys.path.insert(0, str(REPO_ROOT / "phase_5"))
import common as c  # noqa: E402
import plot_style as ps  # noqa: E402
from audio_infill.train import reallike_span, touched_frames  # noqa: E402

OUT = REPO_ROOT / "phase_5" / "results" / "exp01_context_contamination"
FIG = OUT / "figures"
ANNOTATION = REPO_ROOT / "data/processed/nuvole_bianche_short_gaps/nuvole_bianche_multigap_4x_15ms_20ms_30ms_50ms.json"
SOURCE = REPO_ROOT / "data" / "nuvole_bianche.mp3"
DEFAULT_RUNS = ("baseline", "phase1_songs01", "phase2_regularization", "phase3_regularization_seed43")
CONDITIONS = ("A_clean", "B_aligned", "C_deployed", "E_pred_only", "D_coverage", "Z_unfilled")
COND_LABEL = {"A_clean": "A\nclean", "B_aligned": "B\naligned", "C_deployed": "C\ndeployed",
              "E_pred_only": "E\npred. only", "D_coverage": "D\nfull mask", "Z_unfilled": "Z\nunfilled"}
COND_KEY = ("A clean context (training-like)   B frame-aligned zeroing   C deployed today (unaligned zeroing, rounded mask)\n"
            "E C's predictions written into clean context   D mask covers every silent frame   Z gap not inpainted")
PROFILE_RADIUS = 64
MARGIN = 16


# ---------------------------------------------------------------------------
# encoding helpers (model-independent, cached across runs)
# ---------------------------------------------------------------------------

class Encodings:
    """Cache of EnCodec encodings of the ungapped source: the clean codes and codes of the source with given sample spans zeroed."""
    def __init__(self, encoder, base_audio: np.ndarray):
        """Encode the clean base audio once and prepare the cache for zeroed variants."""
        self.encoder = encoder
        self.base = base_audio
        self.cache: Dict[Tuple, torch.Tensor] = {}
        self.clean = self._encode(base_audio)

    def _encode(self, audio: np.ndarray) -> torch.Tensor:
        """Encode a mono float array to long codes [K, frames]."""
        codes, _ = self.encoder.encode(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0))
        return codes.long()

    def zeroed(self, spans: Tuple[Tuple[int, int], ...]) -> torch.Tensor:
        """Codes of the source with the given (start, end) sample spans zeroed (cached)."""
        key = tuple(spans)
        if key not in self.cache:
            a = self.base.copy()
            for s, e in spans:
                a[s:e] = 0.0
            self.cache[key] = self._encode(a)
        return self.cache[key]


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(trainer, x_win: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """fp32, eval, batch of 1 -- the same call Trainer.inpaint_all_gaps makes. Returns log-probs [K,T,V]."""
    dev = trainer.device
    xb = x_win.clone()
    xb[:, mask] = trainer.mask_token
    m = mask.unsqueeze(0).to(dev)
    seg, left, right = trainer._build_model_boundary_tensors(m)
    logits = trainer.model(xb.unsqueeze(0).to(dev), seg, left, right)[0]
    return F.log_softmax(logits.float(), dim=-1)


def score(trainer, enc_win, clean_win, mask, eval_frames, crop, region, fill=True, out_win=None) -> Dict[str, float]:
    """enc_win: context tokens of the condition; clean_win: clean tokens (targets/reference).
    mask: frames the model fills; eval_frames: frames scored at token level (the rounded mask);
    crop: (lo, hi) frames decoded; region: (lo, hi) frames whose samples form the 'gap region'."""
    out: Dict[str, float] = {}
    filled = (enc_win if out_win is None else out_win).clone()
    if fill:
        logp = predict(trainer, enc_win, mask).cpu()
        pred = logp.argmax(-1)
        filled[:, mask] = pred[:, mask]
        tgt = clean_win[:, eval_frames]
        lp = logp[:, eval_frames, :]
        out["nll"] = float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())
        out["acc_top1"] = float((pred[:, eval_frames] == tgt).float().mean())
    lo, hi = crop
    hop = trainer.cfg.target_sr // trainer.encoder.frame_rate
    pred_audio = trainer._decode_validation_window_audio(filled[:, lo:hi])
    ref_audio = trainer._decode_validation_window_audio(clean_win[:, lo:hi])
    n = min(len(pred_audio), len(ref_audio))
    pred_audio, ref_audio = pred_audio[:n], ref_audio[:n]
    r0, r1 = (region[0] - lo) * hop, (region[1] - lo) * hop
    out["si_sdr_crop"] = c.si_sdr_db(pred_audio, ref_audio)
    out["si_sdr_gap"] = c.si_sdr_db(pred_audio[r0:r1], ref_audio[r0:r1])
    out["spectral_convergence_crop"] = trainer._spectral_convergence(pred_audio, ref_audio)
    out["snr_gap"] = c.snr_db(pred_audio[r0:r1], ref_audio[r0:r1])
    out["log_mel_l1_crop"] = c.log_mel_l1(pred_audio, ref_audio, trainer.cfg.target_sr)
    return out


def profile_rows(kind: str, clean: torch.Tensor, zeroed: torch.Tensor, g0: int, g1: int, frames: int) -> List[Dict]:
    """Per (signed distance from the rounded mask edge, codebook): does the zeroed encoding differ from
    the clean one? Distance -1 = frame just before the mask, +1 = frame just after."""
    rows = []
    for f in range(max(0, g0 - PROFILE_RADIUS), min(frames, g1 + PROFILE_RADIUS)):
        if g0 <= f < g1:
            continue
        d = f - g0 if f < g0 else f - (g1 - 1)
        diff = (clean[:, f] != zeroed[:, f]).numpy()
        for k in range(clean.shape[0]):
            rows.append({"kind": kind, "distance": d, "codebook": k, "mismatch": int(diff[k])})
    return rows


def far_mismatch(clean: torch.Tensor, zeroed: torch.Tensor, spans_frames: List[Tuple[int, int]]) -> float:
    """Fraction of tokens that differ more than PROFILE_RADIUS frames away from every gap."""
    keep = torch.ones(clean.shape[1], dtype=torch.bool)
    for a, b in spans_frames:
        keep[max(0, a - PROFILE_RADIUS):b + PROFILE_RADIUS] = False
    return float((clean[:, keep] != zeroed[:, keep]).float().mean())


# ---------------------------------------------------------------------------
# per-run evaluation
# ---------------------------------------------------------------------------

def run_one(run: str, device: str, enc_holder: Dict, ann: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Evaluate conditions A-E/Z for one run on the real gaps and held-out validation gaps. Returns (example records, effects, coverage rows, metadata)."""
    trainer = c.build_trainer(run, "best_val", device=device)
    trainer.model.eval()
    hop = trainer.cfg.target_sr // trainer.encoder.frame_rate
    if "enc" not in enc_holder:
        base = c.load_source_wav_matching_annotation(str(SOURCE), trainer.cfg.target_sr)
        enc_holder["enc"] = Encodings(trainer.encoder, base)
    E: Encodings = enc_holder["enc"]
    enc_holder.setdefault("first_run", run)
    frames = min(E.clean.shape[1], trainer.frames)
    clean = E.clean[:, :frames]
    records, profile, coverage = [], [], []
    meta = {"run": run, "step": trainer.global_step, "hop_samples": hop, "frames": frames}

    # --- the 4 real gaps --------------------------------------------------------------------
    real_spans = [(int(g["gap_start_sample"]), int(g["gap_end_sample"])) for g in ann["gaps"]]
    real_zeroed = E.zeroed(tuple(real_spans))[:, :frames]
    agree = float((real_zeroed[:, :frames] == trainer.codes[:, :frames]).float().mean())
    meta["agreement_trainer_codes_vs_reencoded_gapped_source"] = agree
    meta["real_far_mismatch_rate"] = far_mismatch(clean, real_zeroed, [touched_frames(s, e, hop) for s, e in real_spans])
    for gi, ((s, e), (g0, g1)) in enumerate(zip(real_spans, trainer.gaps_f)):
        t0, t1 = touched_frames(s, e, hop)
        zero_in_mask = max(0, min(e, g1 * hop) - max(s, g0 * hop))
        coverage.append({"run": run, "gap": gi, "gap_ms": 1000 * (e - s) / trainer.cfg.target_sr,
                         "zero_start": s, "zero_end": e, "mask_g0": g0, "mask_g1": g1, "touched_f0": t0, "touched_f1": t1,
                         "zeroed_samples": e - s, "zeroed_outside_mask": (e - s) - zero_in_mask,
                         "frac_zeroed_outside_mask": ((e - s) - zero_in_mask) / (e - s),
                         "masked_samples_not_zeroed": (g1 - g0) * hop - zero_in_mask})
        if run == enc_holder["first_run"]:  # token-level profile is model-independent: compute once
            profile += profile_rows("real", clean, real_zeroed, g0, g1, frames)
        L = g1 - g0
        budget = max(0, trainer.cfg.max_len - L - 16)
        w0 = max(0, g0 - budget // 2)
        w1 = min(frames, g1 + (budget - budget // 2))
        cw, zw = clean[:, w0:w1], real_zeroed[:, w0:w1]
        T = w1 - w0
        mask_r = torch.zeros(T, dtype=torch.bool); mask_r[g0 - w0:g1 - w0] = True
        mask_c = torch.zeros(T, dtype=torch.bool); mask_c[t0 - w0:t1 - w0] = True
        eval_f = mask_r.clone()
        lo = max(0, min(t0, g0) - w0 - MARGIN); hi = min(T, max(t1, g1) - w0 + MARGIN)
        region = (min(t0, g0) - w0, max(t1, g1) - w0)
        conds = {"A_clean": (cw, mask_r, True, None), "C_deployed": (zw, mask_r, True, None),
                 "E_pred_only": (zw, mask_r, True, cw),
                 "D_coverage": (zw, mask_c, True, None), "Z_unfilled": (zw, mask_r, False, None)}
        for name, (xw, m, fill, ow) in conds.items():
            r = score(trainer, xw, cw, m, eval_f, (lo, hi), region, fill, ow)
            records.append({"run": run, "set": "real", "example": f"real_gap_{gi}", "band": "real",
                            "mask_len": L, "filled_len": int(m.sum()), "condition": name, **r})

    # --- held-out synthetic gaps (the run's own validation set) -----------------------------------
    grouped, _ = c.build_real_validation_examples(trainer)
    for key, examples in grouped.items():
        if "_len_" not in key:
            continue
        band = "high" if key.startswith("high") else "low"
        for ei, ex in enumerate(examples):
            ws, ms, L = int(ex.window_start), int(ex.mask_start), int(ex.mask_len)
            T = int(ex.y.shape[1])
            if ws + T > frames:
                continue
            g0, g1 = ws + ms, ws + ms + L
            s_al, e_al = g0 * hop, g1 * hop
            s_rl, e_rl = reallike_span(g0, g1, hop)
            t0, t1 = touched_frames(s_rl, e_rl, hop)
            z_al = E.zeroed(((s_al, e_al),))
            z_rl = E.zeroed(((s_rl, e_rl),))
            if run == enc_holder["first_run"]:
                profile += profile_rows("synthetic_aligned", clean, z_al[:, :frames], g0, g1, frames)
                profile += profile_rows("synthetic_reallike", clean, z_rl[:, :frames], g0, g1, frames)
            cw, aw, rw = clean[:, ws:ws + T], z_al[:, ws:ws + T], z_rl[:, ws:ws + T]
            mask_r = torch.zeros(T, dtype=torch.bool); mask_r[ms:ms + L] = True
            mask_c = torch.zeros(T, dtype=torch.bool); mask_c[max(0, t0 - ws):min(T, t1 - ws)] = True
            lo = max(0, min(t0, g0) - ws - MARGIN); hi = min(T, max(t1, g1) - ws + MARGIN)
            region = (max(0, min(t0, g0) - ws), min(T, max(t1, g1) - ws))
            conds = {"A_clean": (cw, mask_r, True, None), "B_aligned": (aw, mask_r, True, None),
                     "C_deployed": (rw, mask_r, True, None), "E_pred_only": (rw, mask_r, True, cw),
                     "D_coverage": (rw, mask_c, True, None), "Z_unfilled": (rw, mask_r, False, None)}
            for name, (xw, m, fill, ow) in conds.items():
                r = score(trainer, xw, cw, m, mask_r, (lo, hi), region, fill, ow)
                records.append({"run": run, "set": "synthetic", "example": f"{key}_{ei}", "band": band,
                                "mask_len": L, "filled_len": int(m.sum()), "condition": name, **r})
    del trainer
    torch.cuda.empty_cache()
    return records, profile, coverage, meta


# ---------------------------------------------------------------------------
# statistics + figures
# ---------------------------------------------------------------------------

def mean_ci(x: np.ndarray) -> Tuple[float, float]:
    """Mean and 95% normal-approximation CI half-width, ignoring non-finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def paired_effects(records: List[Dict]) -> List[Dict]:
    """Per run, set, condition: mean of (condition - A_clean) over paired examples, with 95% CI."""
    by = defaultdict(dict)
    for r in records:
        by[(r["run"], r["set"], r["example"])][r["condition"]] = r
    out = []
    runs = sorted({r["run"] for r in records})
    for run in runs:
        for st in ("synthetic", "real"):
            keys = [k for k in by if k[0] == run and k[1] == st]
            for cond in CONDITIONS[1:]:
                for metric in ("nll", "acc_top1", "si_sdr_gap", "si_sdr_crop", "spectral_convergence_crop", "snr_gap", "log_mel_l1_crop"):
                    d = [by[k][cond][metric] - by[k]["A_clean"][metric] for k in keys
                         if cond in by[k] and metric in by[k][cond] and metric in by[k]["A_clean"]]
                    if not d:
                        continue
                    m, ci = mean_ci(np.array(d))
                    out.append({"run": run, "set": st, "condition": cond, "metric": metric, "n": len(d),
                                "mean_diff_vs_A": m, "ci95": ci})
    return out


def fig_profile(prof_agg: List[Dict]) -> None:
    """prof_agg rows: kind, distance, codebook (1-based), n, mismatch_rate."""
    ps.apply_style()
    agg = defaultdict(list)
    for r in prof_agg:
        agg[(r["kind"], int(r["distance"]))].append(float(r["mismatch_rate"]))
    kinds = [("synthetic_aligned", "B: zeroing aligned to the gap frames (encoder only)"),
             ("synthetic_reallike", "C: zeroing not frame-aligned (like the real gaps)"),
             ("real", "the 4 real gaps")]
    fig, axes = plt_subplots(1, 2, (12.5, 4.0), width_ratios=[1.35, 1])
    ax = axes[0]
    for i, (kind, label) in enumerate(kinds):
        ds = sorted(d for (k, d) in agg if k == kind and -8 <= d <= 64)
        if not ds:
            continue
        ys = [100 * np.mean(agg[(kind, d)]) for d in ds]
        for side in (-1, 1):
            xs = [d for d in ds if np.sign(d) == side]
            yy = [y for d, y in zip(ds, ys) if np.sign(d) == side]
            ax.plot(xs, yy, color=ps.CAT[i], marker=ps.MARKERS[i], linestyle=ps.LINESTYLES[i],
                    label=label if side == -1 else None, markersize=5)
    ax.axvspan(-0.5, 0.5, color=ps.GRID, zorder=0)
    ax.annotate("masked gap", xy=(0, 2), xytext=(-7.5, 12), fontsize=8, color=ps.INK_2,
                arrowprops={"arrowstyle": "-", "color": ps.MUTED, "linewidth": 0.8})
    ax.set_xlabel("frames from the masked gap (negative = before, positive = after)")
    ax.set_ylabel("context tokens changed by the zeroing (%)")
    ax.set_title("How far the gap's silence corrupts the context tokens")
    ax.legend(loc="upper right")
    ax.set_xlim(-8.5, 64.5)
    sec = ax.secondary_xaxis("top", functions=(lambda f: f * 13.333, lambda t: t / 13.333))
    sec.set_xlabel("ms from the gap", fontsize=8.5, color=ps.INK_2)
    # codebook x distance heatmap for the real-like synthetic case
    ax = axes[1]
    kind = "synthetic_reallike"
    cb = defaultdict(list)
    for r in prof_agg:
        if r["kind"] == kind and -4 <= int(r["distance"]) <= 16:
            cb[(int(r["codebook"]) - 1, int(r["distance"]))].append(float(r["mismatch_rate"]))
    ds = sorted({d for (_, d) in cb})
    mat = np.array([[100 * np.mean(cb[(k, d)]) for d in ds] for k in range(8)])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(ds)), [str(d) for d in ds])
    ax.set_yticks(range(8), [str(k + 1) for k in range(8)])
    ax.set_xlabel("frames from the masked gap")
    ax.set_ylabel("codebook")
    ax.set_title("Per codebook, case C (% tokens changed)")
    ax.grid(False)
    for (k, j), v in np.ndenumerate(mat):
        ax.text(j, k, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=ps.INK if v < 55 else "#ffffff")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5, colors=ps.INK_2)
    ps.save(fig, FIG, "contamination_profile")


def fig_coverage(coverage: List[Dict], sr: int, hop: int) -> None:
    """Figure of the gap, its rounded frame mask and the frames actually touched by the zeroing, per real gap."""
    ps.apply_style()
    cov = [r for r in coverage if r["run"] == coverage[0]["run"]]
    fig, axes = plt_subplots(len(cov), 1, (11, 1.15 * len(cov) + 0.8), sharex=False)
    for ax, r in zip(np.atleast_1d(axes), cov):
        f0 = min(r["mask_g0"], r["touched_f0"]) - 1
        f1 = max(r["mask_g1"], r["touched_f1"]) + 1
        to_ms = lambda smp: 1000.0 * (smp - f0 * hop) / sr  # noqa: E731
        for f in range(f0, f1 + 1):
            ax.axvline(to_ms(f * hop), color=ps.AXIS, linewidth=0.8, zorder=1)
        ax.add_patch(plt_rect((to_ms(r["touched_f0"] * hop), 0.05), to_ms(r["touched_f1"] * hop) - to_ms(r["touched_f0"] * hop), 0.9,
                              fill=False, edgecolor=ps.CAT[2], linewidth=1.8, linestyle="--", zorder=4))
        ax.add_patch(plt_rect((to_ms(r["mask_g0"] * hop), 0.2), to_ms(r["mask_g1"] * hop) - to_ms(r["mask_g0"] * hop), 0.6,
                              color=ps.CAT[0], alpha=0.85, zorder=2))
        ax.add_patch(plt_rect((to_ms(r["zero_start"]), 0.35), to_ms(r["zero_end"]) - to_ms(r["zero_start"]), 0.3,
                              color=ps.CAT[1], zorder=3))
        ax.set_xlim(to_ms(f0 * hop), to_ms(f1 * hop))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"real gap {r['gap']} ({r['gap_ms']:.0f} ms): {100 * r['frac_zeroed_outside_mask']:.0f}% of the silent samples "
                     f"lie outside the mask", fontsize=9.5, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("time (ms); vertical lines = 13.3 ms EnCodec frame boundaries")
    from matplotlib.patches import Patch
    handles = [Patch(color=ps.CAT[1], label="zeroed (silent) samples"),
               Patch(color=ps.CAT[0], alpha=0.85, label="masked frames today (start/end rounded)"),
               Patch(fill=False, edgecolor=ps.CAT[2], linestyle="--", linewidth=1.8, label="frames touched by the silence (full coverage)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ps.save(fig, FIG, "real_gap_mask_coverage")


def fig_effects(effects: List[Dict], st: str, name: str, title: str) -> None:
    """Figure of each contamination condition's effect (vs clean) on NLL, SI-SDR and log-mel L1."""
    ps.apply_style()
    metrics = [("nll", "token NLL on the gap (nats)\nlower is better"),
               ("si_sdr_crop", "SI-SDR, gap +/- 16 frames (dB)\nhigher is better"),
               ("log_mel_l1_crop", "log-mel L1, gap +/- 16 frames\nlower is better")]
    conds = [cd for cd in CONDITIONS[1:] if any(e["condition"] == cd and e["set"] == st for e in effects)]
    runs = sorted({e["run"] for e in effects})
    fig, axes = plt_subplots(1, 3, (13.5, 4.1))
    greys = ["#52514e", "#898781", "#6d6c68", "#a8a69f"]
    for ax, (metric, ylabel) in zip(axes, metrics):
        cs = [cd for cd in conds if any(e["condition"] == cd and e["metric"] == metric and e["set"] == st for e in effects)]
        for ri, run in enumerate(runs):
            for ci_, cond in enumerate(cs):
                e = [x for x in effects if x["run"] == run and x["condition"] == cond and x["metric"] == metric and x["set"] == st]
                if not e:
                    continue
                e = e[0]
                xpos = ci_ + (ri - (len(runs) - 1) / 2) * 0.13
                ax.errorbar(xpos, e["mean_diff_vs_A"], yerr=e["ci95"] if np.isfinite(e["ci95"]) else None,
                            fmt=ps.MARKERS[ri % 4], color=greys[ri % 4], ecolor=greys[ri % 4], elinewidth=1.2, capsize=2.5,
                            markersize=6, label=run if (ci_ == 0 and metric == "nll") else None)
        for ci_, cond in enumerate(cs):
            vals = [x["mean_diff_vs_A"] for x in effects if x["condition"] == cond and x["metric"] == metric and x["set"] == st]
            if vals:
                ax.plot([ci_ - 0.3, ci_ + 0.3], [np.mean(vals)] * 2, color=ps.CAT[0], linewidth=2.5, zorder=5,
                        label="mean over runs" if (ci_ == 0 and metric == "nll") else None)
        ax.axhline(0, color=ps.INK_2, linewidth=1.0)
        ax.set_xticks(range(len(cs)), [COND_LABEL[cd] for cd in cs], fontsize=8.5)
        ax.set_ylabel(f"difference vs A (clean context)\n{ylabel}", fontsize=8.5)
    fig.suptitle(title, fontsize=11, color=ps.INK)
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.1, COND_KEY, ha="center", va="top", fontsize=8.3, color=ps.INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    ps.save(fig, FIG, name)


def fig_real_gaps(records: List[Dict], coverage: List[Dict]) -> None:
    """Figure of SI-SDR per real gap for the main conditions (A clean, C deployed, D coverage, Z unfilled)."""
    ps.apply_style()
    conds = ["A_clean", "C_deployed", "D_coverage", "Z_unfilled"]
    colors = {"A_clean": ps.CAT[0], "C_deployed": ps.CAT[1], "D_coverage": ps.CAT[2], "Z_unfilled": ps.INK_2}
    labels = {"A_clean": "A clean context (training-like)", "C_deployed": "C deployed today",
              "D_coverage": "D full-coverage mask", "Z_unfilled": "Z not inpainted (silence left)"}
    gaps = sorted({int(r["example"].split("_")[-1]) for r in records if r["set"] == "real"})
    ms = {r["gap"]: r["gap_ms"] for r in coverage}
    fig, axes = plt_subplots(1, 2, (12.5, 4.0))
    for ax, metric, ylabel in ((axes[0], "si_sdr_crop", "SI-SDR, gap +/- 16 frames (dB)\nhigher is better"),
                               (axes[1], "log_mel_l1_crop", "log-mel L1, gap +/- 16 frames\nlower is better")):
        for j, cond in enumerate(conds):
            xs, ys, lo, hi = [], [], [], []
            for g in gaps:
                v = [r[metric] for r in records if r["set"] == "real" and r["example"] == f"real_gap_{g}" and r["condition"] == cond]
                if not v:
                    continue
                xs.append(g + (j - 1.5) * 0.17); ys.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=ps.MARKERS[j], color=colors[cond], ecolor=colors[cond],
                        capsize=2.5, elinewidth=1.2, markersize=7, label=labels[cond] if metric == "si_sdr_crop" else None)
        ax.set_xticks(gaps, [f"gap {g}\n{ms[g]:.0f} ms" for g in gaps])
        ax.set_ylabel(ylabel)
    n_ck = len({r["run"] for r in records})
    axes[0].set_title(f"Real gaps: mean over the {n_ck} checkpoints (bars = min/max)", fontsize=10, loc="left")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ps.save(fig, FIG, "real_gaps_quality")


def plt_subplots(nr, nc, size, **kw):
    """plt.subplots wrapper that accepts width_ratios and a (w, h) size tuple."""
    import matplotlib.pyplot as plt
    gridspec_kw = {"width_ratios": kw.pop("width_ratios")} if "width_ratios" in kw else None
    return plt.subplots(nr, nc, figsize=size, gridspec_kw=gridspec_kw, **kw)


def plt_rect(xy, w, h, **kw):
    """Matplotlib Rectangle helper (lazy import)."""
    from matplotlib.patches import Rectangle
    return Rectangle(xy, w, h, **kw)


def write_csv(path: Path, rows: List[Dict]) -> None:
    """Write dict rows to CSV using the union of keys in first-seen order."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """CLI: run (or, with --figures-only, redraw from saved CSVs) the experiment for each run, write CSV/JSON results and figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from the saved CSVs")
    args = ap.parse_args()
    if args.figures_only:
        make_figures()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    ann = json.loads(ANNOTATION.read_text())
    enc_holder: Dict = {}
    records, profile, coverage, metas = [], [], [], []
    for run in args.runs:
        print(f"=== {run} ===", flush=True)
        r, p, cv, m = run_one(run, args.device, enc_holder, ann)
        records += r; profile += p; coverage += cv; metas.append(m)
        print(f"  {len(r)} records; trainer.codes vs re-encoded gapped source agreement {m['agreement_trainer_codes_vs_reencoded_gapped_source']:.4f}", flush=True)
    effects = paired_effects(records)
    write_csv(OUT / "records.csv", records)
    write_csv(OUT / "paired_effects.csv", effects)
    write_csv(OUT / "real_gap_coverage.csv", coverage)
    agg = defaultdict(list)
    for r in profile:
        agg[(r["kind"], r["distance"], r["codebook"])].append(r["mismatch"])
    write_csv(OUT / "contamination_profile.csv", [{"kind": k, "distance": d, "codebook": cb + 1, "n": len(v),
                                                    "mismatch_rate": float(np.mean(v))} for (k, d, cb), v in sorted(agg.items())])
    (OUT / "run_metadata.json").write_text(json.dumps(metas, indent=1))
    make_figures()
    print(f"Wrote {OUT}")


def _read(path: Path) -> List[Dict]:
    """Read a results CSV, converting numeric-looking cells to float."""
    def conv(v):
        """Convert a cell to float when possible, else keep the string."""
        try:
            return float(v)
        except ValueError:
            return v
    with path.open(newline="") as fh:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def make_figures() -> None:
    """All figures from the saved CSVs (so layout changes do not need the GPU run)."""
    records, effects = _read(OUT / "records.csv"), _read(OUT / "paired_effects.csv")
    coverage, prof = _read(OUT / "real_gap_coverage.csv"), _read(OUT / "contamination_profile.csv")
    for r in coverage:
        for k in ("gap", "zero_start", "zero_end", "mask_g0", "mask_g1", "touched_f0", "touched_f1"):
            r[k] = int(r[k])
    hop = json.loads((OUT / "run_metadata.json").read_text())[0]["hop_samples"]
    fig_profile(prof)
    fig_coverage(coverage, 24000, hop)
    n_held = len({r["example"] for r in records if r["set"] == "synthetic"})
    fig_effects(effects, "synthetic", "effect_heldout_gaps",
                f"Held-out gaps ({n_held} per checkpoint): change vs clean context, per checkpoint (95% CI)")
    fig_effects(effects, "real", "effect_real_gaps_vs_clean",
                "Real gaps (4 per checkpoint): change vs clean context, per checkpoint (95% CI)")
    fig_real_gaps(records, coverage)


if __name__ == "__main__":
    main()
