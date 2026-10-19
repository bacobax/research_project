#!/usr/bin/env python3
"""Demo audio samples for the supervisor: the same ~2 s window, filled by several models.

For every chosen gap this writes one WAV per version (24 kHz mono, gap +- 1 s, one common gain per gap
so all versions play at the same level): the original audio, the zeroed input, and the splice output
(only the gap's samples replaced, 2.5 ms raised-cosine cross-fade) of each model in MODELS.

Gaps:
  * the 4 real carved gaps (15/20/30/50 ms), filled by the repo's own inference path
    (`Trainer.inpaint_all_gaps` with `inpaint_output_mode = "splice"`, the real gapped input);
  * 6 of the 128 held-out gaps of `phase_6/eval_final.py` (its gap sampling and condition C protocol:
    whole song re-encoded with a real-like zeroed span, fp32, greedy codebook decoding). Selection is
    not by hand: gaps are split into a short (1-2 frames) and a long (3-4 frames) band, and in each band
    the best, median and worst gap by gap-only SI-SDR averaged over all models are kept.

Trainers are rebuilt with `phase_0/common.py::build_trainer` (never writes into outputs/runs/).
Output: phase_6/results/demo_samples/{README.md, index.csv, gap_<id>/*.wav, figures/*.png|pdf}.

    uv run python scripts/make_demo_samples.py --device cuda:1
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
for sub in ("phase_0", "phase_5", "phase_6"):
    sys.path.insert(0, str(REPO / sub))
import common as c  # noqa: E402
import eval_final as ef  # noqa: E402
import plot_style as ps  # noqa: E402
from audio_infill.train import splice_gap_audio  # noqa: E402

OUT = REPO / "phase_6" / "results" / "demo_samples"
# (file label, run, checkpoint tag). Early best_val checkpoints (Phase 6 30k, s43 20k) are left out on purpose.
MODELS: List[Tuple[str, str, str]] = [
    ("phase6_final", ef.P6, "final"),
    ("phase6_best_audio", ef.P6, "best_audio"),
    ("songs17_s43", ef.S43, "final"),
    ("songs17_s44", ef.S44, "final"),
    ("songs17_1M", ef.S42, "best_val"),
    ("single_song", "phase1_songs01", "final"),
]
BANDS = {"short": (1, 2), "long": (3, 4)}  # held-out gap length in frames (75 Hz: 13-27 ms, 40-53 ms)
CROP_S = 1.0
TARGET_RMS_DBFS = -23.0
PEAK_LIMIT = 10 ** (-1.0 / 20)  # -1 dBFS
EDGE_FADE_S = 0.01  # short fade at the crop edges so playback does not click


def finite(x) -> bool:
    """True for a finite number (blank CSV cells and None are not)."""
    return isinstance(x, (int, float)) and math.isfinite(x)


def scores(out: np.ndarray, base: np.ndarray, zspan: Tuple[int, int], region: Tuple[int, int]) -> Tuple[float, float]:
    """(gap-only SI-SDR over the zeroed span, region SI-SDR over gap +- 16 frames), both vs the original."""
    (zs, ze), (r0, r1) = zspan, region
    return c.si_sdr_db(out[zs:ze], base[zs:ze]), c.si_sdr_db(out[r0:r1], base[r0:r1])


@torch.no_grad()
def fill_real(tr, base: np.ndarray, spans, gaps_f, hop: int, sr: int) -> List[Dict]:
    """Real gaps through Trainer.inpaint_all_gaps (splice). Returns per gap {crop (gap +- 1 s), gap, region}."""
    tr.cfg.inpaint_output_mode = "splice"
    out = tr.inpaint_all_gaps()
    res = []
    for (zs, ze), (g0, g1) in zip(spans, gaps_f):
        region = ((g0 - ef.MARGIN) * hop, (g1 + ef.MARGIN) * hop)
        g, r = scores(out, base, (zs, ze), region)
        c0, c1 = crop_bounds(zs, ze, sr, len(base))
        res.append(dict(crop=out[c0:c1].copy(), gap=g, region=r))
    return res


@torch.no_grad()
def fill_heldout(tr, base: np.ndarray, E, clean: torch.Tensor, e: Dict, hop: int, sr: int) -> Dict:
    """One held-out gap under eval_final's condition C; returns the gap +- 1 s crop of the splice output."""
    ws, T, ms, L = e["ws"], e["T"], e["ms"], e["mask_len"]
    cw = clean[:, ws:ws + T]
    zw = E.window(((e["zs"], e["ze"]),), ws, ws + T)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[ms:ms + L] = True
    pred = ef.predict(tr, zw, cw, mask)[0]
    filled = zw.clone()
    filled[:, mask] = pred[:, mask]
    dec = np.asarray(ef.decode(tr, filled[:, e["lo"]:e["hi"]]), dtype=np.float32)
    rs = (ws + e["lo"]) * hop
    c0, c1 = crop_bounds(e["zs"], e["ze"], sr, len(base))
    inp = base[c0:c1].copy()
    inp[e["zs"] - c0:e["ze"] - c0] = 0.0
    spl = splice_gap_audio(inp, dec, rs - c0, e["zs"] - c0, e["ze"] - c0, int(round(0.0025 * sr)))
    n = min((e["hi"] - e["lo"]) * hop, len(dec))
    g, r = scores(spl, base[c0:c1], (e["zs"] - c0, e["ze"] - c0), (rs - c0, rs - c0 + n))
    return dict(crop=spl, gap=g, region=r)


def crop_bounds(zs: int, ze: int, sr: int, n: int) -> Tuple[int, int]:
    return max(0, zs - int(CROP_S * sr)), min(n, ze + int(CROP_S * sr))


def pick_heldout(examples: List[Dict], per_model: Dict[str, Dict[str, Dict]]) -> List[Tuple[Dict, str, float]]:
    """Best / median / worst gap per length band by gap SI-SDR averaged over all models."""
    chosen = []
    for band, lens in BANDS.items():
        ranked = []
        for e in examples:
            if e["mask_len"] not in lens:
                continue
            vals = [per_model[m][e["example"]]["gap"] for m, _, _ in MODELS]
            if all(finite(v) for v in vals):
                ranked.append((float(np.mean(vals)), e))
        ranked.sort(key=lambda t: t[0], reverse=True)
        for pick, i in (("best", 0), ("median", len(ranked) // 2), ("worst", len(ranked) - 1)):
            chosen.append((ranked[i][1], f"{band}_{pick}", ranked[i][0]))
    return chosen


def write_versions(gap_dir: Path, versions: Dict[str, np.ndarray], sr: int) -> float:
    """One common gain per gap (original at TARGET_RMS_DBFS, no version above -1 dBFS); 16-bit WAVs. Returns the gain."""
    rms = float(np.sqrt(np.mean(versions["original"].astype(np.float64) ** 2))) + 1e-12
    gain = 10 ** (TARGET_RMS_DBFS / 20) / rms
    peak = max(float(np.max(np.abs(v))) for v in versions.values()) + 1e-12
    gain = min(gain, PEAK_LIMIT / peak)
    nf = int(EDGE_FADE_S * sr)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    gap_dir.mkdir(parents=True, exist_ok=True)
    for name, v in versions.items():
        y = (v * gain).astype(np.float32)
        y[:nf] *= ramp
        y[-nf:] *= ramp[::-1]
        sf.write(str(gap_dir / f"{name}.wav"), y, sr, subtype="PCM_16")
    return gain


def fig_gap(gap_id: str, title: str, versions: Dict[str, np.ndarray], zs: int, ze: int, sr: int) -> None:
    """Waveform (gap +- 60 ms) and log spectrogram (gap +- 300 ms) of original, zeroed and phase6_final."""
    import matplotlib.pyplot as plt
    rows = [("original", ps.INK), ("zeroed", ps.MUTED), ("phase6_final", ps.CAT[0])]
    fig, axes = plt.subplots(3, 2, figsize=(11, 6.6), gridspec_kw=dict(width_ratios=[1, 1.4]))
    w = int(0.06 * sr)
    s = int(0.30 * sr)
    n_fft, hop = 512, 64
    win = np.hanning(n_fft)
    specs = {}
    for name, _ in rows:
        y = versions[name][max(0, zs - s):ze + s]
        frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * win
        specs[name] = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)).T + 1e-6)
    vmax = max(v.max() for v in specs.values())
    ylim = max(float(np.max(np.abs(versions[n][zs - w:ze + w]))) for n, _ in rows) * 1.1 + 1e-6
    for i, (name, col) in enumerate(rows):
        ax = axes[i, 0]
        y = versions[name][zs - w:ze + w]
        t = (np.arange(len(y)) - w) / sr * 1000
        ax.plot(t, y, color=col, lw=0.8)
        ax.axvspan(0, (ze - zs) / sr * 1000, color=ps.CAT[3], alpha=0.18, lw=0)
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(name, fontsize=9)
        ax = axes[i, 1]
        S = specs[name]
        t0 = -(zs - max(0, zs - s)) / sr * 1000
        ext = [t0 + n_fft / 2 / sr * 1000, t0 + (n_fft / 2 + hop * S.shape[1]) / sr * 1000, 0, sr / 2000]
        ax.imshow(S, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=vmax - 80, vmax=vmax)
        for x in (0, (ze - zs) / sr * 1000):
            ax.axvline(x, color="white", lw=0.7, ls="--")
        ax.set_ylim(0, 8)
        ax.set_ylabel("kHz", fontsize=8)
        ax.grid(False)
    axes[-1, 0].set_xlabel("time from gap start (ms); shaded = zeroed span")
    axes[-1, 1].set_xlabel("time from gap start (ms); dashed = zeroed span, 0-8 kHz")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    ps.save(fig, OUT / "figures", f"gap_{gap_id}")


def fig_scores(rows: List[Dict]) -> None:
    """Gap-only and region SI-SDR of every version, per demo gap (the numbers in index.csv)."""
    import matplotlib.pyplot as plt
    gaps = list(dict.fromkeys(r["gap_id"] for r in rows))
    style = {"zeroed": (ps.MUTED, "x"), "phase6_final": (ps.CAT[0], "o"), "phase6_best_audio": (ps.CAT[0], "D"),
             "songs17_s43": (ps.CAT[1], "s"), "songs17_s44": (ps.CAT[2], "^"), "songs17_1M": (ps.INK_2, "v"),
             "single_song": (ps.INK_2, "P")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(gaps) + 1.8), sharey=True)
    for ax, key, label in ((axes[0], "gap_si_sdr_db", "gap-only SI-SDR vs original (dB)"),
                           (axes[1], "region_si_sdr_db", "region SI-SDR vs original, gap +- 16 frames (dB)")):
        for j, (model, (col, mk)) in enumerate(style.items()):
            pts = [(r[key], gaps.index(r["gap_id"])) for r in rows if r["model"] == model and finite(r[key])]
            if pts:
                off = (j - 3) * 0.08
                ax.scatter([p[0] for p in pts], [p[1] + off for p in pts], color=col, marker=mk, s=26,
                           facecolors=col if model != "phase6_best_audio" else "white", label=model, zorder=3)
        ax.set_title(label, fontsize=9.5)
    axes[0].set_yticks(range(len(gaps)))
    axes[0].set_yticklabels(gaps, fontsize=8)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    if "zeroed" not in labels:
        h1, l1 = axes[1].get_legend_handles_labels()
        handles, labels = h1, l1
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle("Demo gaps: score of every version (zeroed has no gap-only score: its gap is silence)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ps.save(fig, OUT / "figures", "scores")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--scores-figure-only", action="store_true", help="redraw figures/scores from an existing index.csv")
    args = ap.parse_args(argv)
    ps.apply_style()
    if args.scores_figure_only:
        fig_scores(ef.read_csv(OUT / "index.csv"))
        return

    tr6 = c.build_trainer(ef.P6, "final", device=args.device)
    sr = tr6.cfg.target_sr
    hop = sr // tr6.encoder.frame_rate
    base = c.load_source_wav_matching_annotation(str(ef.SOURCE), sr)
    E = ef.Encodings(tr6.encoder, base)
    frames = min(E.clean.shape[1], tr6.frames)
    clean = E.clean[:, :frames].long()
    examples, holdout6 = ef.heldout_examples(tr6, frames)
    ann = json.loads(ef.ANNOTATION.read_text())
    spans = [(int(g["gap_start_sample"]), int(g["gap_end_sample"])) for g in ann["gaps"]]
    gaps_f = list(tr6.gaps_f)
    gapped = tr6.wav.squeeze(0).numpy().astype(np.float32)
    real_ref = base.copy()
    for s, e in spans:
        real_ref[s:e] = 0.0
    n_cmp = min(len(gapped), len(base))
    print(f"{len(examples)} held-out gaps; real gapped input vs zeroed source: max |diff| "
          f"{float(np.max(np.abs(gapped[:n_cmp] - real_ref[:n_cmp]))):.2e}", flush=True)

    real: Dict[str, List[Dict]] = {}
    held: Dict[str, Dict[str, Dict]] = {}
    steps: Dict[str, int] = {}
    holdout_same: Dict[str, bool] = {}
    tr, tr_run = tr6, ef.P6
    for label, run, tag in MODELS:
        if run != tr_run:
            del tr
            torch.cuda.empty_cache()
            tr, tr_run = c.build_trainer(run, tag, device=args.device), run
        ck = torch.load(c.RUNS[run]["checkpoints"][tag], map_location=tr.device, weights_only=False)
        tr.model.load_state_dict(ck["model"])
        tr.model.eval()
        steps[label] = int(ck["step"])
        holdout_same[label] = [tuple(r) for r in c.build_real_validation_examples(tr)[1]] == [tuple(r) for r in holdout6]
        print(f"{label}: {run}/{tag} step {steps[label]:,} (same holdout as Phase 6: {holdout_same[label]})", flush=True)
        real[label] = fill_real(tr, base, spans, gaps_f, hop, sr)
        held[label] = {e["example"]: fill_heldout(tr, base, E, clean, e, hop, sr) for e in examples}
    E.save()

    # sanity check against the evaluation's own numbers (same protocol -> same scores)
    rec = {r["example"]: r for r in ef.read_csv(ef.OUT / "records.csv")
           if r["run"] == ef.P6 and r["tag"] == "final" and r.get("condition") == "C" and r.get("set") == "heldout"}
    diffs = [abs(held["phase6_final"][k]["gap"] - rec[k]["wav_gapspan_si_sdr"]) for k in rec
             if k in held["phase6_final"] and finite(rec[k]["wav_gapspan_si_sdr"]) and finite(held["phase6_final"][k]["gap"])]
    print(f"vs records.csv (phase6 final, gap SI-SDR): max |diff| {max(diffs):.3f} dB over {len(diffs)} gaps", flush=True)

    rows: List[Dict] = []
    meta = {"steps": steps, "holdout_same_as_phase6": holdout_same, "records_csv_max_abs_diff_db": max(diffs),
            "selection": {}, "gains": {}}

    def add(gap_id, kind, gap_ms, versions, metrics, zs, ze, pick=""):
        gain = write_versions(OUT / f"gap_{gap_id}", versions, sr)
        meta["gains"][gap_id] = gain
        for name in versions:
            run, tag = next(((r, t) for m, r, t in MODELS if m == name), ("-", "-"))
            g, r = metrics.get(name, (None, None))
            rows.append(dict(gap_id=gap_id, kind=kind, gap_ms=round(gap_ms, 1), model=name, run=run, checkpoint=tag,
                             step=steps.get(name, ""), gap_si_sdr_db="" if not finite(g) else round(g, 2),
                             region_si_sdr_db="" if not finite(r) else round(r, 2), pick=pick,
                             file=f"gap_{gap_id}/{name}.wav"))
        fig_gap(gap_id, f"gap {gap_id} ({kind}, {gap_ms:.1f} ms zeroed)", versions, zs - crop_lo(zs, ze), ze - crop_lo(zs, ze), sr)

    def crop_lo(zs, ze):
        return crop_bounds(zs, ze, sr, len(base))[0]

    for gi, ((zs, ze), (g0, g1)) in enumerate(zip(spans, gaps_f)):
        c0, c1 = crop_bounds(zs, ze, sr, len(base))
        region = ((g0 - ef.MARGIN) * hop, (g1 + ef.MARGIN) * hop)
        gap_ms = (ze - zs) / sr * 1000
        versions = {"original": base[c0:c1], "zeroed": gapped[c0:c1]}
        metrics = {"zeroed": (None, scores(gapped, base, (zs, ze), region)[1])}
        for label, _, _ in MODELS:
            versions[label] = real[label][gi]["crop"]
            metrics[label] = (real[label][gi]["gap"], real[label][gi]["region"])
        add(f"real{gi}_{round(gap_ms)}ms", "real", gap_ms, versions, metrics, zs, ze)

    for e, pick, mean_gap in pick_heldout(examples, held):
        zs, ze = e["zs"], e["ze"]
        c0, c1 = crop_bounds(zs, ze, sr, len(base))
        zeroed = base[c0:c1].copy()
        zeroed[zs - c0:ze - c0] = 0.0
        rs = (e["ws"] + e["lo"]) * hop
        n = (e["hi"] - e["lo"]) * hop
        versions = {"original": base[c0:c1], "zeroed": zeroed}
        metrics = {"zeroed": (None, c.si_sdr_db(zeroed[rs - c0:rs - c0 + n], base[rs:rs + n]))}
        for label, _, _ in MODELS:
            versions[label] = held[label][e["example"]]["crop"]
            metrics[label] = (held[label][e["example"]]["gap"], held[label][e["example"]]["region"])
        gap_id = f"heldout_{pick}"
        meta["selection"][gap_id] = dict(example=e["example"], activity_band=e["band"], mask_frames=e["mask_len"],
                                         mean_gap_si_sdr_over_models=mean_gap)
        add(gap_id, "heldout", (ze - zs) / sr * 1000, versions, metrics, zs, ze, pick=pick)

    # band-level context for the README: where the picked gaps sit in the 128-gap distribution
    for label, _, _ in MODELS:
        g = [v["gap"] for v in held[label].values() if finite(v["gap"])]
        r = [v["region"] for v in held[label].values() if finite(v["region"])]
        meta.setdefault("heldout_means", {})[label] = dict(gap_si_sdr=float(np.mean(g)), region_si_sdr=float(np.mean(r)))
    ef.write_csv(OUT / "index.csv", rows)
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    fig_scores(rows)
    print(f"wrote {len(rows)} wavs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
