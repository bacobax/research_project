#!/usr/bin/env python3
"""
Phase 0 / Experiment 3 -- Decoded-audio metrics on synthetic validation masks.

Question: token top-1 accuracy is a weak proxy for "does the fill sound right" -- many
different token sequences decode to perceptually similar audio. This experiment decodes every
method's filled window (same examples/methods as exp02) and scores the audio restricted to the
gap span + margin (matching cfg.decoded_loss_margin_frames, the same window the training-time
decoded loss used) against ground truth, using multi-res STFT distance, SI-SDR, SNR, waveform
L1/MSE, and log-mel L1. Every method's numbers are reported alongside codec_ceiling (decode of
the true codes, unmasked) so the model's score reads relative to what's achievable in this
token space, not against zero.

Decodes only the gap+margin crop of codes (common.decode_gap_crop_audio), not the full
seq_len=512 window -- matching exactly what Trainer._compute_decoded_domain_loss does at
training time, and ~14x cheaper per call.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def run_one(run_name: str, checkpoint_tag: str) -> list:
    trainer = c.build_trainer(run_name, checkpoint_tag)
    step = trainer.global_step
    sr = trainer.cfg.target_sr
    margin_frames = trainer.cfg.decoded_loss_margin_frames
    stft_loss = c.build_stft_loss(trainer)
    rng = np.random.default_rng(1234)

    grouped_examples, _ = c.build_real_validation_examples(trainer)
    per_length_groups = {k: v for k, v in grouped_examples.items() if "_len_" in k}

    records = []
    rows = []
    t_start = time.time()
    n_done = 0
    n_total = sum(len(v) for v in per_length_groups.values())
    for group_name, examples in per_length_groups.items():
        band = "high_activity" if group_name.startswith("high_activity") else "low_activity"
        mask_len = int(group_name.rsplit("_", 1)[-1])

        for example in examples:
            target_crop = c.decode_gap_crop_audio(trainer, example.y, example.mask_start, example.mask_len, margin_frames)

            method_audio = {}
            for name in c.BASELINE_NAMES:
                if name == "codec_ceiling":
                    method_audio[name] = target_crop  # identity: filled codes == target codes
                    continue
                filled = c.fill_baseline(name, example, rng)
                method_audio[name] = c.decode_gap_crop_audio(trainer, filled, example.mask_start, example.mask_len, margin_frames)
            filled_model = trainer._predict_validation_window(example)
            method_audio["model"] = c.decode_gap_crop_audio(trainer, filled_model, example.mask_start, example.mask_len, margin_frames)

            for method, pred_crop in method_audio.items():
                metrics = c.audio_metrics(pred_crop, target_crop, sr, stft_loss)
                records.append({
                    "run": run_name, "checkpoint": checkpoint_tag, "step": step,
                    "group": group_name, "band": band, "mask_len": mask_len,
                    "method": method, **metrics,
                })
                for metric_name, value in metrics.items():
                    rows.append({
                        "experiment": "exp03_decoded_metrics", "run": run_name,
                        "checkpoint": checkpoint_tag, "step": step, "band": band,
                        "mask_len": mask_len, "method": method, "metric": metric_name,
                        "value": value,
                    })
            n_done += 1
            elapsed = time.time() - t_start
            print(f"  [{n_done}/{n_total} examples, {elapsed:.1f}s elapsed, {elapsed/n_done:.2f}s/example]", flush=True)

    print(f"\n=== {run_name} / {checkpoint_tag} (step {step}) ===")
    for method in c.BASELINE_NAMES + ["model"]:
        method_records = [r for r in records if r["method"] == method]
        mean_sc = np.mean([r["stft_spectral_convergence"] for r in method_records])
        mean_sisdr = np.mean([r["si_sdr_db"] for r in method_records])
        print(f"  {method:16s} mean spectral_convergence={mean_sc:.4f}  mean SI-SDR(dB)={mean_sisdr:8.2f}")

    return records, rows


def main():
    all_records = []
    all_rows = []
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            records, rows = run_one(run_name, checkpoint_tag)
            all_records.extend(records)
            all_rows.extend(rows)

    c.save_json(c.RESULTS_DIR / "exp03_decoded_metrics.json", all_records)
    c.append_summary_rows(all_rows)
    print(f"\nWrote {c.RESULTS_DIR / 'exp03_decoded_metrics.json'}")

    ceiling = [r for r in all_records if r["method"] == "codec_ceiling"]
    print(f"\nSanity: codec_ceiling mean wave_l1 = {np.mean([r['wave_l1'] for r in ceiling]):.6f} (expect ~0)")
    print(f"Sanity: codec_ceiling mean SI-SDR(dB) = {np.mean([r['si_sdr_db'] for r in ceiling]):.2f} (expect very high)")


if __name__ == "__main__":
    main()
