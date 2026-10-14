#!/usr/bin/env python3
"""Phase 6 evaluation -- does the final recipe do what it was built to do?

The Phase 6 run (`phase6_final_songs17_seed43`) changed four things at once against the songs17
baseline (`phase3_songs17_seed43/44`, same data, same 300k-step schedule): zeroed-gap context in
training, codebook-conditioned (depth_ar) heads, no STFT loss, and splice inference. Its own
validation curves are NOT comparable to the baseline's (fp32 + zeroed context + 128 examples vs
fp16 + clean context + 64 examples), so every checkpoint is re-scored here under ONE protocol, on
ONE shared example set:

  * 128 held-out gaps of the pivot song (16 per activity band x gap length 1-4), built with the
    Phase 6 run's validation construction. The held-out regions are chosen by activity, not by
    seed, and the script asserts that every run's holdout ranges are identical (no leakage).
  * All tokens come from the original ungapped source (`data/nuvole_bianche.mp3`), as in Phase 5:
      A  clean context   : clean-audio tokens, gap masked                 (how the baselines trained)
      C  deployed        : the whole song re-encoded with a real-like zeroed span (what inference sees)
  * fp32, batch of 1, greedy codebook decoding (inference path). Token NLL is the teacher-forced
    joint NLL for depth_ar (exact), the plain NLL for independent heads.
  * Audio is scored two ways:
      token-domain (as in training-time validation): decoded fill vs decoded clean tokens;
      waveform-domain (what a listener gets): output vs the ORIGINAL audio, for
        decode_all (old inference: the whole window re-decoded) and splice (Phase 6 inference).

Points tested (numbers in results/eval/FINDINGS.md, one figure each, PNG + PDF):
  P1  Deployment quality vs baseline at a matched budget: Phase 6 final vs songs17 seed 43/44 final
      (and best_val vs best_val), condition C, splice output. Rule fixed in advance: an effect is
      claimed only if the paired 95% CI excludes 0 AND the difference to the mean of the two
      baseline seeds exceeds the seed-43-vs-44 difference on the same metric.
  P2  Context robustness: C - A per checkpoint. Phase 6 was trained on C-like context, so its C - A
      should be <= the baselines', and its C score should beat theirs.
  P3  Codebook structure (H2 -> H6): NLL per codebook (1..8) and its slope. Conditioned heads should
      flatten the slope / lower later-codebook NLL relative to the baseline.
  P4  Splice: splice vs decode_all (waveform domain), held-out and the 4 real gaps, with zero-fill
      floor and oracle-token ceiling. Phase 5 predicts +2-4 dB around the gap.
  P5  Checkpoint selection inside Phase 6: best_val (val NLL) vs best_audio (crop SI-SDR) vs final
      vs 100k/200k snapshots under the deployment protocol.
  P6  Validation-protocol confound: the training-time curves (Phase 6 zeroed/fp32 vs baseline
      clean/fp16) next to the common-protocol numbers, to show how much of the logged SI-SDR gap
      was protocol.

    uv run python phase_6/eval_final.py --device cuda:1        # full run (~1 h; whole-song encodes cached)
    uv run python phase_6/eval_final.py --report-only          # tables + figures from saved CSVs
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase_0"))
sys.path.insert(0, str(REPO / "phase_5"))
sys.path.insert(0, str(REPO / "phase_6"))
import common as c  # noqa: E402
import plot_style as ps  # noqa: E402
from audio_infill.train import reallike_span, splice_gap_audio, touched_frames  # noqa: E402

OUT = REPO / "phase_6" / "results" / "eval"
FIG = OUT / "figures"
CACHE = REPO / "outputs" / "phase6_eval_cache.pt"  # whole-song encodings (outputs/ is gitignored)
ANNOTATION = REPO / "data/processed/nuvole_bianche_short_gaps/nuvole_bianche_multigap_4x_15ms_20ms_30ms_50ms.json"
SOURCE = REPO / "data" / "nuvole_bianche.mp3"
MARGIN = 16  # frames around the gap for crop/region metrics (= decoded_loss_margin_frames)
P6 = "phase6_final_songs17_seed43"
S43, S44, S42 = "phase3_songs17_seed43", "phase3_songs17_seed44", "phase1_songs17"
# (run, checkpoint tags). The first tag is used to build the trainer, the others swap weights in.
PLAN: Dict[str, Tuple[str, ...]] = {
    P6: ("final", "best_val", "best_audio", "step_100000", "step_200000"),
    S43: ("final", "best_val"),
    S44: ("final", "best_val"),
    S42: ("best_val",),  # seed 42, 1M-step schedule (best at 950k): long-budget reference only
}
LABEL = {P6: "Phase 6", S43: "songs17 s43", S44: "songs17 s44", S42: "songs17 s42 (1M)"}
COLOR = {P6: ps.CAT[0], S43: ps.CAT[1], S44: ps.CAT[2], S42: ps.CAT[3]}
MARK = {P6: "o", S43: "s", S44: "^", S42: "D"}
K = 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mean_ci(x: Sequence[float]) -> Tuple[float, float, int]:
    a = np.asarray([v for v in x if v is not None and math.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    half = 1.96 * a.std(ddof=1) / math.sqrt(a.size) if a.size > 1 else float("nan")
    return float(a.mean()), float(half), int(a.size)


@torch.no_grad()
def predict(trainer, x_win: torch.Tensor, y_win: torch.Tensor, mask: torch.Tensor):
    """fp32, batch of 1, the inference call. Returns (greedy tokens [K,T], per-codebook NLL [K] on the mask,
    teacher-forced argmax tokens [K,T]: for depth_ar each codebook's argmax given the TRUE lower codebooks,
    which separates exposure bias from per-codebook skill; identical to greedy for independent heads)."""
    dev = trainer.device
    xb = x_win.clone()
    xb[:, mask] = trainer.mask_token
    xb = xb.unsqueeze(0).to(dev)
    seg, left, right = trainer._build_model_boundary_tensors(mask.unsqueeze(0).to(dev))
    greedy = trainer.model(xb, seg, left, right)[0]
    pred = greedy.argmax(-1).cpu()
    if trainer.cfg.codebook_conditioning == "depth_ar":
        tf = trainer.model(xb, seg, left, right, gap_targets=y_win.unsqueeze(0).to(dev))[0]
    else:
        tf = greedy
    logp = F.log_softmax(tf.float(), dim=-1)[:, mask.to(dev), :]  # [K, L, V]
    tgt = y_win[:, mask].to(dev)
    nll_k = (-logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).mean(dim=1).cpu().numpy()
    return pred, nll_k, tf.argmax(-1).cpu()


def decode(trainer, codes: torch.Tensor) -> np.ndarray:
    return trainer._decode_validation_window_audio(codes)


class Encodings:
    """Whole-song encodings of the clean source and of copies with zeroed spans, cached on disk."""

    def __init__(self, encoder, base: np.ndarray):
        self.encoder = encoder
        self.base = base
        self.disk = torch.load(CACHE, weights_only=False) if CACHE.exists() else {}
        if "clean" not in self.disk:
            self.disk["clean"] = self._encode(base)
        self.clean = self.disk["clean"]
        self.dirty = False

    def _encode(self, audio: np.ndarray) -> torch.Tensor:
        codes, _ = self.encoder.encode(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0))
        return codes.to(torch.int16)

    def window(self, spans: Tuple[Tuple[int, int], ...], w0: int, w1: int) -> torch.Tensor:
        key = (spans, w0, w1)
        if key not in self.disk:
            a = self.base.copy()
            for s, e in spans:
                a[s:e] = 0.0
            self.disk[key] = self._encode(a)[:, w0:w1].clone()
            self.dirty = True
        return self.disk[key].long()

    def save(self):
        if self.dirty:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.disk, CACHE)
            self.dirty = False


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def score_example(trainer, run, tag, step, ex_meta, x_cond, clean_win, mask, crop, audio_in, rs, zspan, ref_cache):
    """One (checkpoint, example, condition). Returns a record dict."""
    hop = trainer.cfg.target_sr // trainer.encoder.frame_rate
    sr = trainer.cfg.target_sr
    pred, nll_k, pred_tf = predict(trainer, x_cond, clean_win, mask)
    filled = x_cond.clone()
    filled[:, mask] = pred[:, mask]
    tgt = clean_win[:, mask]
    acc_k = (pred[:, mask] == tgt).float().mean(dim=1).numpy()
    acc_tf_k = (pred_tf[:, mask] == tgt).float().mean(dim=1).numpy()
    rec = dict(ex_meta, run=run, tag=tag, step=step, nll=float(nll_k.mean()), acc_top1=float(acc_k.mean()),
               acc_top1_tf=float(acc_tf_k.mean()))
    for k in range(K):
        rec[f"nll_k{k}"] = float(nll_k[k])
        rec[f"acc_k{k}"] = float(acc_k[k])
        rec[f"acc_tf_k{k}"] = float(acc_tf_k[k])
    lo, hi = crop
    m0 = int(mask.nonzero()[0])
    m1 = int(mask.nonzero()[-1]) + 1
    # token domain (training-time validation style): decoded fill vs decoded clean tokens
    pred_crop = decode(trainer, filled[:, lo:hi])
    key = (ex_meta["example"], "ref")
    if key not in ref_cache:
        ref_cache[key] = (decode(trainer, clean_win[:, lo:hi]), decode(trainer, clean_win[:, m0:m1]))
    ref_crop, ref_gap = ref_cache[key]
    pred_gap = decode(trainer, filled[:, m0:m1])
    rec["tok_crop_si_sdr"] = c.si_sdr_db(pred_crop[: len(ref_crop)], ref_crop[: len(pred_crop)])
    rec["tok_gap_si_sdr"] = c.si_sdr_db(pred_gap[: len(ref_gap)], ref_gap[: len(pred_gap)])
    # waveform domain vs the ORIGINAL audio (only meaningful for deployed context)
    if audio_in is not None:
        rec.update(wave_metrics(pred_crop, audio_in, trainer, rs, zspan, sr, hop, lo, hi))
    return rec


def wave_metrics(pred_crop, audio_in, trainer, rs, zspan, sr, hop, lo, hi, prefix=""):
    """decode_all vs splice vs original audio over the region [rs, rs + (hi-lo)*hop)."""
    base = trainer._p6_base
    n = (hi - lo) * hop
    ref = base[rs:rs + n]
    inp = audio_in[rs:rs + n]
    dec = np.asarray(pred_crop[:n], dtype=np.float32)
    n = min(len(ref), len(dec))
    ref, inp, dec = ref[:n], inp[:n], dec[:n]
    zs, ze = zspan[0] - rs, zspan[1] - rs
    fade = int(round(0.0025 * sr))
    spl = splice_gap_audio(inp, dec, 0, zs, ze, fade)
    out = {}
    for name, sig in (("decode_all", dec), ("splice", spl)):
        out[f"{prefix}wav_region_si_sdr_{name}"] = c.si_sdr_db(sig, ref)
        out[f"{prefix}wav_region_log_mel_{name}"] = c.log_mel_l1(sig, ref, sr)
    out[f"{prefix}wav_gapspan_si_sdr"] = c.si_sdr_db(spl[zs:ze], ref[zs:ze])
    out[f"{prefix}wav_gapspan_snr"] = c.snr_db(spl[zs:ze], ref[zs:ze])
    return out


def floors(trainer, ex_meta, clean_win, audio_in, rs, zspan, crop, sr, hop):
    """Model-independent references: zero fill (the gapped input) and oracle tokens (clean tokens decoded + spliced)."""
    lo, hi = crop
    base = trainer._p6_base
    n = (hi - lo) * hop
    ref = base[rs:rs + n]
    inp = audio_in[rs:rs + n]
    zs, ze = zspan[0] - rs, zspan[1] - rs
    rows = []
    rows.append(dict(ex_meta, run="zero_fill", tag="-", step=0, condition="C",
                     wav_region_si_sdr_splice=c.si_sdr_db(inp, ref), wav_region_log_mel_splice=c.log_mel_l1(inp, ref, sr),
                     wav_gapspan_snr=c.snr_db(inp[zs:ze], ref[zs:ze])))
    oracle = decode(trainer, clean_win[:, lo:hi])
    rec = dict(ex_meta, run="oracle_tokens", tag="-", step=0, condition="C")
    rec.update(wave_metrics(oracle, audio_in, trainer, rs, zspan, sr, hop, lo, hi))
    rows.append(rec)
    return rows


def run_eval(device: str, limit: Optional[int] = None, plan: Optional[Dict[str, Tuple[str, ...]]] = None) -> None:
    plan = plan or PLAN
    ann = json.loads(ANNOTATION.read_text())
    records: List[Dict] = []
    meta: Dict = {"plan": {r: list(t) for r, t in (plan or PLAN).items()}, "checkpoint_steps": {}}

    # shared example set + encodings, from the Phase 6 trainer
    tr6 = c.build_trainer(P6, plan[P6][0], device=device)
    hop = tr6.cfg.target_sr // tr6.encoder.frame_rate
    sr = tr6.cfg.target_sr
    base = c.load_source_wav_matching_annotation(str(SOURCE), sr)
    E = Encodings(tr6.encoder, base)
    frames = min(E.clean.shape[1], tr6.frames)
    clean = E.clean[:, :frames].long()
    grouped, holdout6 = c.build_real_validation_examples(tr6)
    examples = []
    for key, items in grouped.items():
        if "_len_" not in key:
            continue
        for i, ex in enumerate(items):
            ws, ms, L = int(ex.window_start), int(ex.mask_start), int(ex.mask_len)
            T = int(ex.y.shape[1])
            if ws + T > frames:
                continue
            g0, g1 = ws + ms, ws + ms + L
            zs, ze = reallike_span(g0, g1, hop)
            lo, hi = max(0, ms - MARGIN), min(T, ms + L + MARGIN)
            examples.append(dict(example=f"{key}_{i}", band="high" if key.startswith("high") else "low",
                                 mask_len=L, ws=ws, ms=ms, T=T, zs=zs, ze=ze, lo=lo, hi=hi))
    if limit:
        examples = examples[::max(1, len(examples) // limit)][:limit]
    print(f"{len(examples)} held-out examples; encoding zeroed songs (cached in {CACHE}) ...", flush=True)
    for i, e in enumerate(examples):
        E.window(((e["zs"], e["ze"]),), e["ws"], e["ws"] + e["T"])
        if i % 16 == 15:
            E.save()
            print(f"  encoded {i + 1}/{len(examples)}", flush=True)
    real_spans = tuple((int(g["gap_start_sample"]), int(g["gap_end_sample"])) for g in ann["gaps"])
    real_windows = []
    for gi, ((s, e), (g0, g1)) in enumerate(zip(real_spans, tr6.gaps_f)):
        L = g1 - g0
        budget = max(0, tr6.cfg.max_len - L - 16)
        w0, w1 = max(0, g0 - budget // 2), min(frames, g1 + (budget - budget // 2))
        real_windows.append(dict(example=f"real_gap_{gi}", band="real", mask_len=L, ws=w0, ms=g0 - w0, T=w1 - w0,
                                 zs=s, ze=e, lo=max(0, g0 - w0 - MARGIN), hi=min(w1 - w0, g1 - w0 + MARGIN)))
        E.window(real_spans, w0, w1)
    E.save()
    audio_real = base.copy()
    for s, e in real_spans:
        audio_real[s:e] = 0.0
    meta["n_heldout"] = len(examples)
    meta["holdout_ranges"] = [list(r) for r in holdout6]
    meta["agreement_source_clean_vs_trainer_codes"] = float((clean[:, :frames] == tr6.codes[:, :frames]).float().mean())

    for run, tags in plan.items():
        tr = tr6 if run == P6 else c.build_trainer(run, tags[0], device=device)
        tr._p6_base = base
        if run != P6:
            _, holdout = c.build_real_validation_examples(tr)
            assert [tuple(r) for r in holdout] == [tuple(r) for r in holdout6], f"holdout ranges differ for {run}"
        for tag in tags:
            path = c.RUNS[run]["checkpoints"][tag]
            ck = torch.load(path, map_location=tr.device, weights_only=False)
            tr.model.load_state_dict(ck["model"])
            tr.model.eval()
            step = int(ck["step"])
            meta["checkpoint_steps"][f"{run}/{tag}"] = step
            print(f"{run}/{tag} (step {step:,})", flush=True)
            ref_cache: Dict = {}
            for e in examples:
                ws, T, ms, L = e["ws"], e["T"], e["ms"], e["mask_len"]
                cw = clean[:, ws:ws + T]
                zw = E.window(((e["zs"], e["ze"]),), ws, ws + T)
                mask = torch.zeros(T, dtype=torch.bool)
                mask[ms:ms + L] = True
                exm = {k: e[k] for k in ("example", "band", "mask_len")}
                exm["set"] = "heldout"
                audio_in = base.copy()
                audio_in[e["zs"]:e["ze"]] = 0.0
                rs = (ws + e["lo"]) * hop
                for cond, xw, ain in (("A", cw, None), ("C", zw, audio_in)):
                    rec = score_example(tr, run, tag, step, dict(exm, condition=cond), xw, cw, mask,
                                        (e["lo"], e["hi"]), ain, rs, (e["zs"], e["ze"]), ref_cache)
                    records.append(rec)
                if run == P6 and tag == tags[0]:
                    records += floors(tr, dict(exm), cw, audio_in, rs, (e["zs"], e["ze"]), (e["lo"], e["hi"]), sr, hop)
            for e in real_windows:
                ws, T, ms, L = e["ws"], e["T"], e["ms"], e["mask_len"]
                cw = clean[:, ws:ws + T]
                zw = E.window(real_spans, ws, ws + T)
                mask = torch.zeros(T, dtype=torch.bool)
                mask[ms:ms + L] = True
                exm = {"example": e["example"], "band": "real", "mask_len": L, "set": "real"}
                rs = (ws + e["lo"]) * hop
                rec = score_example(tr, run, tag, step, dict(exm, condition="C"), zw, cw, mask, (e["lo"], e["hi"]),
                                    audio_real, rs, (e["zs"], e["ze"]), ref_cache)
                records.append(rec)
                if run == P6 and tag == tags[0]:
                    records += floors(tr, dict(exm), cw, audio_real, rs, (e["zs"], e["ze"]), (e["lo"], e["hi"]), sr, hop)
            write_csv(OUT / "records.csv", records)
        if run != P6:
            del tr
            torch.cuda.empty_cache()
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    write_csv(OUT / "records.csv", records)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> List[Dict]:
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v) if v not in ("", None) else None
                except ValueError:
                    d[k] = v
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

METRICS_DEPLOY = [  # (metric, condition, label, higher_is_better)
    ("nll", "C", "token NLL (nats)", False),
    ("acc_top1", "C", "top-1 accuracy", True),
    ("wav_gapspan_si_sdr", "C", "gap SI-SDR vs original (dB)", True),
    ("wav_region_si_sdr_splice", "C", "region SI-SDR, splice (dB)", True),
    ("wav_region_log_mel_splice", "C", "region log-mel L1, splice", False),
    ("tok_gap_si_sdr", "C", "token-domain gap SI-SDR (dB)", True),
    ("tok_crop_si_sdr", "C", "token-domain crop SI-SDR (dB)", True),
]


def index(records, set_="heldout"):
    """{(run, tag, condition): {example: record}}"""
    idx: Dict = defaultdict(dict)
    for r in records:
        if r.get("set") == set_:
            idx[(r["run"], r["tag"], r["condition"])][r["example"]] = r
    return idx


def paired(idx, a, b, metric, band=None):
    """mean/CI of a - b over shared examples. a, b: (run, tag, cond) or a list of keys (averaged per example)."""
    def val(keys, ex):
        keys = keys if isinstance(keys, list) else [keys]
        vs = [idx[k][ex].get(metric) for k in keys if ex in idx[k]]
        vs = [v for v in vs if v is not None and math.isfinite(v)]
        return float(np.mean(vs)) if len(vs) == len(keys) else None
    ka = a if isinstance(a, list) else [a]
    kb = b if isinstance(b, list) else [b]
    exs = set.intersection(*[set(idx[k]) for k in ka + kb])
    if band:
        exs = {e for e in exs if idx[ka[0]][e]["band"] == band}
    d = [val(a, e) - val(b, e) for e in sorted(exs) if val(a, e) is not None and val(b, e) is not None]
    return mean_ci(d)


def summarize(records) -> Dict:
    idx = index(records)
    S: Dict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "abs": []}
    keys = sorted({(r["run"], r["tag"]) for r in records if r.get("set") == "heldout" and r["run"] in PLAN})
    for run, tag in keys:
        for cond in ("A", "C"):
            if (run, tag, cond) not in idx:
                continue
            for m in ["nll", "acc_top1", "acc_top1_tf", "tok_crop_si_sdr", "tok_gap_si_sdr", "wav_gapspan_si_sdr",
                      "wav_region_si_sdr_splice", "wav_region_si_sdr_decode_all", "wav_region_log_mel_splice",
                      "wav_region_log_mel_decode_all"]:
                for band in (None, "high", "low"):
                    vals = [r.get(m) for r in idx[(run, tag, cond)].values() if band is None or r["band"] == band]
                    mu, ci, n = mean_ci(vals)
                    if n:
                        S["abs"].append(dict(run=run, tag=tag, condition=cond, metric=m, band=band or "all", mean=mu, ci95=ci, n=n))
    # P1
    for tag in ("final", "best_val"):
        b43, b44, p6 = (S43, tag, "C"), (S44, tag, "C"), (P6, tag, "C")
        if not all(k in idx for k in (b43, b44, p6)):
            continue
        for m, cond, label, hib in METRICS_DEPLOY:
            row = dict(comparison=f"Phase 6 {tag} - songs17 {tag}", metric=m, label=label, higher_is_better=hib)
            for name, ref in (("vs_s43", b43), ("vs_s44", b44), ("vs_mean", [b43, b44])):
                mu, ci, n = paired(idx, p6, ref, m)
                row[f"{name}_mean"], row[f"{name}_ci95"], row["n"] = mu, ci, n
            mu, ci, _ = paired(idx, b44, b43, m)
            row["seed_diff_s44_minus_s43"], row["seed_diff_ci95"] = mu, ci
            excl = abs(row["vs_mean_mean"]) > row["vs_mean_ci95"]
            beyond_seed = abs(row["vs_mean_mean"]) > abs(row["seed_diff_s44_minus_s43"])
            better = (row["vs_mean_mean"] > 0) == hib
            row["verdict"] = ("Phase 6 better" if better else "Phase 6 worse") if (excl and beyond_seed) else (
                "within seed spread" if excl else "no detectable difference")
            S["P1"].append(row)
    # P2: C - A per checkpoint
    for run, tag in keys:
        if (run, tag, "A") in idx and (run, tag, "C") in idx:
            for m in ("nll", "acc_top1", "tok_crop_si_sdr", "tok_gap_si_sdr"):
                mu, ci, n = paired(idx, (run, tag, "C"), (run, tag, "A"), m)
                S["P2"].append(dict(run=run, tag=tag, metric=m, c_minus_a=mu, ci95=ci, n=n))
    # P3: per codebook NLL (condition A and C), slope
    for run, tag in keys:
        for cond in ("A", "C"):
            if (run, tag, cond) not in idx:
                continue
            for band in ("high", "low"):
                rs_ = [r for r in idx[(run, tag, cond)].values() if r["band"] == band]
                nk = [float(np.mean([r[f"nll_k{k}"] for r in rs_])) for k in range(K)]
                ak = [float(np.mean([r[f"acc_k{k}"] for r in rs_])) for k in range(K)]
                atf = [float(np.mean([r[f"acc_tf_k{k}"] for r in rs_])) for k in range(K)]
                slope = float(np.polyfit(np.arange(K), nk, 1)[0])
                S["P3"].append(dict(run=run, tag=tag, condition=cond, band=band, slope=slope,
                                    **{f"nll_k{k}": nk[k] for k in range(K)}, **{f"acc_k{k}": ak[k] for k in range(K)},
                                    **{f"acc_tf_k{k}": atf[k] for k in range(K)}))
    # P4: splice - decode_all (held-out), per checkpoint
    for run, tag in keys:
        if (run, tag, "C") in idx:
            for m in ("wav_region_si_sdr", "wav_region_log_mel"):
                vals = [r[f"{m}_splice"] - r[f"{m}_decode_all"] for r in idx[(run, tag, "C")].values()
                        if r.get(f"{m}_splice") is not None and r.get(f"{m}_decode_all") is not None]
                mu, ci, n = mean_ci(vals)
                S["P4"].append(dict(run=run, tag=tag, metric=m, splice_minus_decode_all=mu, ci95=ci, n=n))
    # P5: Phase 6 checkpoints, deployment metrics
    steps = {}
    for r in records:
        if r["run"] == P6:
            steps[r["tag"]] = int(r["step"])
    for tag in PLAN[P6]:
        if (P6, tag, "C") not in idx:
            continue
        row = dict(tag=tag, step=steps.get(tag))
        for m, cond, label, hib in METRICS_DEPLOY:
            mu, ci, n = mean_ci([r.get(m) for r in idx[(P6, tag, cond)].values()])
            row[m], row[f"{m}_ci95"] = mu, ci
        S["P5"].append(row)
    return S


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_p1(S, records):
    import matplotlib.pyplot as plt
    rows = [r for r in S["abs"] if r["condition"] == "C" and r["band"] == "all"]
    show = [(P6, "final"), (S43, "final"), (S44, "final"), (P6, "best_val"), (S43, "best_val"), (S44, "best_val"), (S42, "best_val")]
    metrics = [m for m in METRICS_DEPLOY if m[0] in ("nll", "acc_top1", "wav_gapspan_si_sdr", "wav_region_si_sdr_splice")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.9))
    for ax, (m, _, label, hib) in zip(axes, metrics):
        for i, (run, tag) in enumerate(show):
            r = next((x for x in rows if x["run"] == run and x["tag"] == tag and x["metric"] == m), None)
            if r is None:
                continue
            ax.errorbar(i, r["mean"], yerr=r["ci95"], fmt=MARK[run], color=COLOR[run], capsize=3,
                         mfc=COLOR[run] if tag == "final" else "white")
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels([f"{LABEL[r]}\n{t}" for r, t in show], rotation=60, ha="right", fontsize=7.5)
        ax.set_title(label + (" (higher better)" if hib else " (lower better)"), fontsize=9)
        ax.axvline(2.5, color=ps.AXIS, lw=0.8)
    fig.suptitle("P1 - deployment protocol (zeroed-gap context, fp32, splice), 128 held-out gaps, mean +- 95% CI\n"
                 "filled = end of the 300k schedule (final), hollow = checkpoint selected on validation NLL", fontsize=10)
    fig.tight_layout()
    ps.save(fig, FIG, "p1_deployment_comparison")

    rows = [r for r in S["P1"] if r["comparison"].startswith("Phase 6 final")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.5, 1.05 * len(rows) + 1.2))
    for ax, r in zip(np.atleast_1d(axes), rows):
        sign = 1 if r["higher_is_better"] else -1  # plot "improvement" so right = better for every metric
        sd = abs(r["seed_diff_s44_minus_s43"])
        ax.axvspan(-sd, sd, color=ps.CAT[1], alpha=0.18, lw=0)
        ax.errorbar(sign * r["vs_mean_mean"], 0, xerr=r["vs_mean_ci95"], fmt="o", color=ps.CAT[0], capsize=3)
        ax.axvline(0, color=ps.INK_2, lw=0.8)
        ax.set_yticks([0])
        ax.set_yticklabels([r["label"]])
        ax.set_ylim(-0.6, 0.6)
        ax.grid(axis="y", visible=False)
        half = max(abs(sign * r["vs_mean_mean"]) + r["vs_mean_ci95"], sd) * 1.25 + 1e-9
        ax.set_xlim(-half, half)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.text(1.0, 0.5, r["verdict"], transform=ax.transAxes, ha="left", va="center", fontsize=8, color=ps.INK_2)
    np.atleast_1d(axes)[-1].set_xlabel("Phase 6 final - mean(songs17 s43, s44 final); sign flipped so right = better")
    fig.suptitle("P1 - paired difference vs baseline (95% CI); shaded band = |s44 - s43| (seed spread); own scale per row", fontsize=9.5)
    fig.tight_layout()
    ps.save(fig, FIG, "p1_paired_difference")


def fig_p2(S):
    import matplotlib.pyplot as plt
    show = [(P6, "final"), (S43, "final"), (S44, "final"), (P6, "best_val"), (S43, "best_val"), (S44, "best_val"), (S42, "best_val")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, label) in zip(axes, (("nll", "token NLL, C - A (nats)"), ("acc_top1", "top-1, C - A"),
                                    ("tok_crop_si_sdr", "token-domain crop SI-SDR, C - A (dB)"))):
        for i, (run, tag) in enumerate(show):
            r = next((x for x in S["P2"] if x["run"] == run and x["tag"] == tag and x["metric"] == m), None)
            if r:
                ax.errorbar(i, r["c_minus_a"], yerr=r["ci95"], fmt=MARK[run], color=COLOR[run], capsize=3,
                            mfc=COLOR[run] if tag == "final" else "white")
        ax.axhline(0, color=ps.INK_2, lw=0.8)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels([f"{LABEL[r]}\n{t}" for r, t in show], rotation=60, ha="right", fontsize=7.5)
        ax.set_title(label, fontsize=9.5)
    fig.suptitle("P2 - sensitivity to deployed (zeroed-gap) vs clean context, same examples, mean +- 95% CI", fontsize=10)
    fig.tight_layout()
    ps.save(fig, FIG, "p2_context_sensitivity")


def fig_p3(S):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for col, band in enumerate(("high", "low")):
        ax = axes[0, col]
        for run in (P6, S43, S44):
            r = next((x for x in S["P3"] if x["run"] == run and x["tag"] == "final" and x["condition"] == "A" and x["band"] == band), None)
            if r:
                ax.plot(range(1, K + 1), [r[f"nll_k{k}"] for k in range(K)], marker=MARK[run], color=COLOR[run],
                        label=f"{LABEL[run]} final (slope {r['slope']:+.2f}/codebook)")
        ax.set_xlabel("codebook (RVQ level)")
        ax.set_ylabel("NLL (nats)")
        ax.set_title(f"NLL per codebook, {band}-activity gaps (clean context)", fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax = axes[1, col]
        for run in (P6, S43, S44):
            r = next((x for x in S["P3"] if x["run"] == run and x["tag"] == "final" and x["condition"] == "C" and x["band"] == band), None)
            if not r:
                continue
            ax.plot(range(1, K + 1), [r[f"acc_k{k}"] for k in range(K)], marker=MARK[run], color=COLOR[run],
                    label=f"{LABEL[run]} final, greedy")
            if run == P6:
                ax.plot(range(1, K + 1), [r[f"acc_tf_k{k}"] for k in range(K)], marker=MARK[run], color=COLOR[run], ls="--",
                        mfc="white", label=f"{LABEL[run]} final, true lower codebooks")
        ax.set_xlabel("codebook (RVQ level)")
        ax.set_ylabel("top-1 accuracy")
        ax.set_title(f"top-1 per codebook, {band}-activity gaps (deployed context)", fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("P3 - codebook structure: do conditioned heads flatten the H2 slope, and how much does greedy decoding lose?", fontsize=10)
    fig.tight_layout()
    ps.save(fig, FIG, "p3_per_codebook_nll")


def fig_p4(S, records):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    idx = index(records)
    ax = axes[0]
    show = [(P6, "final"), (S43, "final"), (S44, "final"), (P6, "best_val"), (S43, "best_val"), (S44, "best_val")]
    for i, (run, tag) in enumerate(show):
        for j, (name, off, mk) in enumerate((("decode_all", -0.15, "x"), ("splice", 0.15, MARK[run]))):
            mu, ci, _ = mean_ci([r.get(f"wav_region_si_sdr_{name}") for r in idx[(run, tag, "C")].values()])
            ax.errorbar(i + off, mu, yerr=ci, fmt=mk, color=COLOR[run], capsize=3, mfc=COLOR[run] if name == "splice" else "none")
    for ref, ls in (("zero_fill", ":"), ("oracle_tokens", "--")):
        mu, _, _ = mean_ci([r.get("wav_region_si_sdr_splice") for r in records if r["run"] == ref and r["set"] == "heldout"])
        ax.axhline(mu, color=ps.MUTED, ls=ls, lw=1, label=f"{ref.replace('_', ' ')} (splice)")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([f"{LABEL[r]}\n{t}" for r, t in show], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("SI-SDR vs original audio, gap +- 16 frames (dB)")
    ax.set_title("held-out gaps: x = decode_all, filled marker = splice", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    reals = sorted({r["example"] for r in records if r.get("set") == "real"})
    for gi, ex in enumerate(reals):
        for run, off in ((P6, -0.2), (S43, 0.0), (S44, 0.2)):
            r = next((x for x in records if x.get("set") == "real" and x["run"] == run and x["tag"] == "final" and x["example"] == ex), None)
            if r:
                ax.plot([gi + off, gi + off], [r["wav_region_si_sdr_decode_all"], r["wav_region_si_sdr_splice"]], color=COLOR[run], lw=1.2)
                ax.plot(gi + off, r["wav_region_si_sdr_decode_all"], "x", color=COLOR[run])
                ax.plot(gi + off, r["wav_region_si_sdr_splice"], MARK[run], color=COLOR[run], label=LABEL[run] if gi == 0 else None)
    ax.set_xticks(range(len(reals)))
    ax.set_xticklabels([r.replace("real_gap_", "gap ") for r in reals])
    ax.set_ylabel("SI-SDR vs original (dB)")
    ax.set_title("4 real gaps, final checkpoints: x = decode_all -> marker = splice", fontsize=9.5)
    ax.legend(fontsize=7.5)
    fig.suptitle("P4 - splice inference vs re-decoding the whole window", fontsize=10)
    fig.tight_layout()
    ps.save(fig, FIG, "p4_splice")


def fig_p5(S):
    import matplotlib.pyplot as plt
    rows = sorted(S["P5"], key=lambda r: r["step"])
    metrics = [("nll", "token NLL (C)"), ("wav_gapspan_si_sdr", "gap SI-SDR vs original (dB)"),
               ("wav_region_si_sdr_splice", "region SI-SDR, splice (dB)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (m, label) in zip(axes, metrics):
        ax.errorbar([r["step"] / 1000 for r in rows], [r[m] for r in rows], yerr=[r[f"{m}_ci95"] for r in rows],
                    fmt="o-", color=ps.CAT[0], capsize=3)
        for r in rows:
            ax.annotate(r["tag"], (r["step"] / 1000, r[m]), fontsize=7, xytext=(4, 4), textcoords="offset points", color=ps.INK_2)
        ax.set_xlabel("training step (k)")
        ax.set_title(label, fontsize=9.5)
    fig.suptitle("P5 - Phase 6 checkpoints under the deployment protocol (which selection rule picks the best audio?)", fontsize=10)
    fig.tight_layout()
    ps.save(fig, FIG, "p5_checkpoint_selection")


def fig_p6():
    import matplotlib.pyplot as plt
    import monitor
    s = monitor.scalars()
    base = defaultdict(dict)
    with open(REPO / "phase_3/results/seed_check/metrics.csv") as f:
        for x in csv.DictReader(f):
            base[(x["run"], x["tag"])][int(x["step"])] = float(x["value"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, tag, label in ((axes[0], "val/combined_loss", "validation NLL"), (axes[1], "val/si_sdr_db_gap_only", "validation gap-only SI-SDR (dB)")):
        p6 = sorted({a: b for a, b, _ in s.get(tag, [])}.items())
        ax.plot([a / 1000 for a, _ in p6], [b for _, b in p6], marker="o", ms=3, color=COLOR[P6], label="Phase 6 (zeroed context, fp32, 128 ex.)")
        for run, key in ((S43, "phase1_songs17_seed43"), (S44, "phase1_songs17_seed44")):
            pts = sorted(base[(key, tag)].items())
            ax.plot([a / 1000 for a, _ in pts], [b for _, b in pts], marker=MARK[run], ms=3, ls="--", color=COLOR[run],
                    label=f"{LABEL[run]} (clean context, fp16, 64 ex.)")
        ax.set_xlabel("training step (k)")
        ax.set_title(label, fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("P6 - training-time validation curves; the two protocols differ (see P1/P2 for the common protocol)", fontsize=10)
    fig.tight_layout()
    ps.save(fig, FIG, "p6_training_curves")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def write_findings(S, records, meta):
    L = ["# Phase 6 evaluation findings (generated by `phase_6/eval_final.py`; do not edit by hand)", ""]
    L.append(f"Common protocol: {meta.get('n_heldout', '?')} held-out gaps of the pivot song (16 per band x gap length 1-4), "
             "identical for every checkpoint (holdout ranges asserted equal across runs); tokens from the original "
             "ungapped source; fp32, batch 1, greedy codebook decoding. Condition A = clean context, C = whole song "
             "re-encoded with a real-like zeroed span (deployment). Mean +- 95% CI over examples.")
    L.append("")
    L.append("Checkpoint steps: " + ", ".join(f"`{k}` {v:,}" for k, v in meta.get("checkpoint_steps", {}).items()))
    L.append("")
    L.append("## P1 - Phase 6 vs songs17 baseline under the deployment protocol")
    L.append("")
    L.append("Paired differences (Phase 6 minus baseline) over the same examples. Rule fixed in advance: an effect is "
             "claimed only if the CI vs the mean of seeds 43/44 excludes 0 AND the difference exceeds the seed-43-vs-44 difference.")
    L.append("")
    L.append("| comparison | metric | vs s43 | vs s44 | vs mean(s43,s44) | seed diff s44-s43 | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for r in S["P1"]:
        L.append(f"| {r['comparison']} | {r['label']} | {f(r['vs_s43_mean'])} +- {f(r['vs_s43_ci95'])} | "
                 f"{f(r['vs_s44_mean'])} +- {f(r['vs_s44_ci95'])} | {f(r['vs_mean_mean'])} +- {f(r['vs_mean_ci95'])} | "
                 f"{f(r['seed_diff_s44_minus_s43'])} | {r['verdict']} |")
    L.append("")
    L.append("Absolute values, condition C, all bands:")
    L.append("")
    L.append("| checkpoint | step | NLL | top-1 | gap SI-SDR vs original | region SI-SDR splice | region SI-SDR decode_all | token gap SI-SDR |")
    L.append("|---|---|---|---|---|---|---|---|")
    ab = {(r["run"], r["tag"], r["condition"], r["metric"], r["band"]): r for r in S["abs"]}
    for run, tags in PLAN.items():
        for tag in tags:
            g = lambda m: ab.get((run, tag, "C", m, "all"))  # noqa: E731
            if g("nll") is None:
                continue
            cell = lambda m: f"{f(g(m)['mean'])} +- {f(g(m)['ci95'])}" if g(m) else "-"  # noqa: E731
            L.append(f"| {LABEL[run]} {tag} | {meta['checkpoint_steps'].get(f'{run}/{tag}', 0):,} | {cell('nll')} | {cell('acc_top1')} | "
                     f"{cell('wav_gapspan_si_sdr')} | {cell('wav_region_si_sdr_splice')} | {cell('wav_region_si_sdr_decode_all')} | {cell('tok_gap_si_sdr')} |")
    for ref in ("zero_fill", "oracle_tokens"):
        vals = [r.get("wav_region_si_sdr_splice") for r in records if r["run"] == ref and r.get("set") == "heldout"]
        gs = [r.get("wav_gapspan_si_sdr") for r in records if r["run"] == ref and r.get("set") == "heldout"]
        L.append(f"| {ref} (reference) | - | - | - | {f(mean_ci(gs)[0])} | {f(mean_ci(vals)[0])} | - | - |")
    L.append("")
    L.append("## P2 - sensitivity to the deployed (zeroed-gap) context: C minus A")
    L.append("")
    L.append("| checkpoint | NLL C-A | top-1 C-A | token crop SI-SDR C-A (dB) | token gap SI-SDR C-A (dB) |")
    L.append("|---|---|---|---|---|")
    p2 = {(r["run"], r["tag"], r["metric"]): r for r in S["P2"]}
    for run, tags in PLAN.items():
        for tag in tags:
            g = lambda m: p2.get((run, tag, m))  # noqa: E731
            if g("nll") is None:
                continue
            L.append(f"| {LABEL[run]} {tag} | " + " | ".join(f"{f(g(m)['c_minus_a'])} +- {f(g(m)['ci95'])}" for m in
                                                             ("nll", "acc_top1", "tok_crop_si_sdr", "tok_gap_si_sdr")) + " |")
    L.append("")
    L.append("## P3 - per-codebook NLL (condition A) and slope per codebook")
    L.append("")
    L.append("| checkpoint | band | NLL cb1 | NLL cb8 | slope (nats/codebook) | top-1 cb1 | top-1 cb8 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in S["P3"]:
        if r["condition"] == "A":
            L.append(f"| {LABEL[r['run']]} {r['tag']} | {r['band']} | {f(r['nll_k0'])} | {f(r['nll_k7'])} | {r['slope']:+.3f} | "
                     f"{f(r['acc_k0'])} | {f(r['acc_k7'])} |")
    L.append("")
    L.append("Exposure bias of the conditioned heads: top-1 accuracy per codebook with the model's OWN previous codebooks "
             "(greedy, what inference does) vs with the TRUE previous codebooks (teacher-forced argmax). Equal for independent heads.")
    L.append("")
    L.append("| checkpoint | band | condition | greedy top-1 (mean over codebooks) | teacher-forced top-1 | greedy cb1..8 | teacher-forced cb1..8 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in S["P3"]:
        if r["condition"] == "C":
            g = [r[f"acc_k{k}"] for k in range(K)]
            t = [r[f"acc_tf_k{k}"] for k in range(K)]
            L.append(f"| {LABEL[r['run']]} {r['tag']} | {r['band']} | C | {f(float(np.mean(g)))} | {f(float(np.mean(t)))} | "
                     + " ".join(f"{v:.2f}" for v in g) + " | " + " ".join(f"{v:.2f}" for v in t) + " |")
    L.append("")
    L.append("## P4 - splice vs decode_all (waveform domain, vs original audio, gap +- 16 frames)")
    L.append("")
    L.append("| checkpoint | SI-SDR splice - decode_all (dB) | log-mel L1 splice - decode_all |")
    L.append("|---|---|---|")
    p4 = {(r["run"], r["tag"], r["metric"]): r for r in S["P4"]}
    for run, tags in PLAN.items():
        for tag in tags:
            a, b = p4.get((run, tag, "wav_region_si_sdr")), p4.get((run, tag, "wav_region_log_mel"))
            if a:
                L.append(f"| {LABEL[run]} {tag} | {f(a['splice_minus_decode_all'], 2)} +- {f(a['ci95'], 2)} | "
                         f"{f(b['splice_minus_decode_all'])} +- {f(b['ci95'])} |")
    L.append("")
    L.append("Real gaps (final checkpoints), SI-SDR vs original over gap +- 16 frames, decode_all -> splice; gap-span SNR:")
    L.append("")
    L.append("| gap | " + " | ".join(LABEL[r] for r in (P6, S43, S44)) + " |")
    L.append("|---|---|---|---|")
    for ex in sorted({r["example"] for r in records if r.get("set") == "real"}):
        cells = []
        for run in (P6, S43, S44):
            r = next((x for x in records if x.get("set") == "real" and x["run"] == run and x["tag"] == "final" and x["example"] == ex), None)
            cells.append(f"{f(r['wav_region_si_sdr_decode_all'], 1)} -> {f(r['wav_region_si_sdr_splice'], 1)} (SNR {f(r['wav_gapspan_snr'], 1)})" if r else "-")
        L.append(f"| {ex} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## P5 - Phase 6 checkpoint selection (deployment protocol)")
    L.append("")
    L.append("| checkpoint | step | NLL (C) | gap SI-SDR vs original | region SI-SDR splice |")
    L.append("|---|---|---|---|---|")
    for r in sorted(S["P5"], key=lambda r: r["step"]):
        L.append(f"| {r['tag']} | {r['step']:,} | {f(r['nll'])} +- {f(r['nll_ci95'])} | {f(r['wav_gapspan_si_sdr'], 2)} +- "
                 f"{f(r['wav_gapspan_si_sdr_ci95'], 2)} | {f(r['wav_region_si_sdr_splice'], 2)} +- {f(r['wav_region_si_sdr_splice_ci95'], 2)} |")
    L.append("")
    L.append("## P6 - training-time curves")
    L.append("")
    L.append("`figures/p6_training_curves`: Phase 6 logged validation on zeroed context in fp32 (128 examples); the "
             "baselines logged clean context in fp16 (64 examples). Compare runs with P1/P2, not with these curves.")
    L.append("")
    L.append("Figures (PNG + PDF): " + ", ".join(f"`figures/{n}`" for n in (
        "p1_deployment_comparison", "p1_paired_difference", "p2_context_sensitivity", "p3_per_codebook_nll",
        "p4_splice", "p5_checkpoint_selection", "p6_training_curves")))
    (OUT / "FINDINGS.md").write_text("\n".join(L) + "\n")


def report() -> None:
    ps.apply_style()
    records = read_csv(OUT / "records.csv")
    meta = json.loads((OUT / "run_metadata.json").read_text())
    S = summarize(records)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    for name in ("P1", "P2", "P3", "P4", "P5"):
        write_csv(OUT / f"{name.lower()}.csv", S[name])
    fig_p1(S, records)
    fig_p2(S)
    fig_p3(S)
    fig_p4(S, records)
    fig_p5(S)
    fig_p6()
    write_findings(S, records, meta)
    print((OUT / "FINDINGS.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="8 examples, 2 checkpoints, output to results/eval_smoke (pipeline check)")
    args = ap.parse_args()
    global OUT, FIG
    if args.smoke:
        OUT = REPO / "phase_6" / "results" / "eval_smoke"
        FIG = OUT / "figures"
        if not args.report_only:
            run_eval(args.device, limit=8, plan={P6: ("final", "best_val"), S43: ("final",), S44: ("final",)})
    elif not args.report_only:
        run_eval(args.device)
    report()


if __name__ == "__main__":
    main()
