#!/usr/bin/env python3
"""
Phase 0 / Experiment 4 -- The real gaps, end-to-end.

Question: forget synthetic validation masks -- how well does the model fill the 4 actual gaps
(15/20/30/50ms, at frame counts [4,2,2,1]) that were carved out of the song for this whole
project? This is the number a professor will actually ask about.

Ground truth comes from data/nuvole_bianche.mp3 (the pre-gapping source, still on disk),
re-loaded and resampled the same way audio_infill.make_gapped_dataset.py did, then encoded with
a fresh AudioEncoder at the same frames as trainer.gaps_f (both audios are sample-aligned since
the gapped version only zeroes the gap span in place -- see common.py's
load_source_wav_matching_annotation docstring). Scoring is against the encode-then-decode of the
TRUE audio, not the raw waveform directly, so codec-reconstruction error (already measured
separately as codec_ceiling in exp02/exp03) is not conflated with infill error.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

from audio_infill.train import AudioEncoder  # noqa: E402


def run_one(run_name: str, checkpoint_tag: str, true_codes_cache: dict) -> list:
    trainer = c.build_trainer(run_name, checkpoint_tag)
    step = trainer.global_step
    sr = trainer.cfg.target_sr
    margin_frames = trainer.cfg.decoded_loss_margin_frames
    stft_loss = c.build_stft_loss(trainer)
    rng = np.random.default_rng(1234)

    cache_key = (trainer.cfg.bandwidth, sr)
    if cache_key not in true_codes_cache:
        source_wav_path = c.REPO_ROOT / "data" / "nuvole_bianche.mp3"
        source_audio = c.load_source_wav_matching_annotation(str(source_wav_path), sr)
        source_wav_t = torch.from_numpy(source_audio).unsqueeze(0)
        fresh_encoder = AudioEncoder(trainer.cfg.bandwidth, trainer.device)
        true_codes, _ = fresh_encoder.encode(source_wav_t)
        true_codes_cache[cache_key] = true_codes
        print(f"  [encoded true source once: {tuple(true_codes.shape)} frames, cached for bandwidth={cache_key[0]}]")
    true_codes = true_codes_cache[cache_key]

    n_frames_common = min(true_codes.shape[1], trainer.frames)
    if true_codes.shape[1] != trainer.frames:
        print(
            f"  WARNING: true_codes frames ({true_codes.shape[1]}) != trainer.frames "
            f"({trainer.frames}); truncating comparisons to {n_frames_common}"
        )

    records = []
    rows = []
    for gap_index, (g0, g1) in enumerate(trainer.gaps_f):
        if g1 > n_frames_common:
            print(f"  skipping gap {gap_index}: [{g0},{g1}) exceeds common frame range {n_frames_common}")
            continue
        example = c.build_real_gap_example(trainer, true_codes, (g0, g1), gap_index)
        gap_ms = 1000.0 * (g1 - g0) / trainer.encoder.frame_rate

        # Decode only the gap+margin crop (matching Trainer._compute_decoded_domain_loss),
        # not the full seq_len window -- much cheaper, same numbers.
        target_crop = c.decode_gap_crop_audio(trainer, example.y, example.mask_start, example.mask_len, margin_frames)

        method_audio = {}
        method_filled = {}
        for name in c.BASELINE_NAMES:
            if name == "codec_ceiling":
                method_audio[name] = target_crop
                method_filled[name] = example.y
                continue
            filled = c.fill_baseline(name, example, rng)
            method_filled[name] = filled
            method_audio[name] = c.decode_gap_crop_audio(trainer, filled, example.mask_start, example.mask_len, margin_frames)
        filled_model = trainer._predict_validation_window(example)
        method_filled["model"] = filled_model
        method_audio["model"] = c.decode_gap_crop_audio(trainer, filled_model, example.mask_start, example.mask_len, margin_frames)

        for method, pred_crop in method_audio.items():
            audio_m = c.audio_metrics(pred_crop, target_crop, sr, stft_loss)
            token_m = c.token_accuracy(method_filled[method], example)
            record = {
                "run": run_name, "checkpoint": checkpoint_tag, "step": step,
                "gap_index": gap_index, "gap_len_frames": g1 - g0, "gap_len_ms": gap_ms,
                "method": method, **audio_m, **token_m,
            }
            records.append(record)
            for metric_name in list(audio_m.keys()) + ["acc_top1"]:
                rows.append({
                    "experiment": "exp04_real_gaps", "run": run_name, "checkpoint": checkpoint_tag,
                    "step": step, "band": "real_gap", "mask_len": g1 - g0, "method": method,
                    "metric": metric_name, "value": record[metric_name],
                })

    print(f"\n=== {run_name} / {checkpoint_tag} (step {step}) ===")
    for gap_index in sorted(set(r["gap_index"] for r in records)):
        gap_records = [r for r in records if r["gap_index"] == gap_index]
        gap_ms = gap_records[0]["gap_len_ms"]
        print(f"  gap {gap_index} ({gap_ms:.0f}ms):")
        for r in gap_records:
            print(f"    {r['method']:16s} acc_top1={r['acc_top1']:.3f}  SI-SDR(dB)={r['si_sdr_db']:8.2f}  spectral_convergence={r['stft_spectral_convergence']:.4f}")

    return records, rows


def main():
    all_records = []
    all_rows = []
    true_codes_cache: dict = {}
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            records, rows = run_one(run_name, checkpoint_tag, true_codes_cache)
            all_records.extend(records)
            all_rows.extend(rows)

    c.save_json(c.RESULTS_DIR / "exp04_real_gaps.json", all_records)
    c.append_summary_rows(all_rows)
    print(f"\nWrote {c.RESULTS_DIR / 'exp04_real_gaps.json'}")


if __name__ == "__main__":
    main()
