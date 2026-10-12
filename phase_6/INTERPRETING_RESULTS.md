# Phase 6 (final 17-song run): how to interpret the results

Shared definitions: [docs/interpreting_results.md](../docs/interpreting_results.md). Plan and change list: `README.md`.

## Status

The only committed result is `results/preflight_zeroed_context.json`. The 300k-step run `phase6_final_songs17_seed43` (`configs/train/phase6_final_songs17_seed43.yaml`,
output `outputs/runs/phase6_final/`, untracked) was still training when this guide was written (the log showed a `best_audio.pt` at step 40,000 with crop SI-SDR 10.705 dB). Nothing has been exported to `phase_6/results/` yet, so there are no final numbers, figures or audio here. This guide explains how to read them once they exist.

## What Phase 6 tests

One run that applies every fix suggested by Phases 0-5, compared with the `phase3_songs17_seed43` twin (same data, seed, 300k steps; differences confirmed by config diff):

| Change | Keys | Phase finding it answers |
|---|---|---|
| Training context encoded from audio with the gap zeroed, as inference does | `zeroed_context_prob: 1.0`, `zeroed_context_warmup_frames: 300` | Phase 5 (2-4 dB lost to contaminated context) |
| Head of codebook k conditioned on codebooks < k (teacher forced in training, greedy at inference) | `codebook_conditioning: depth_ar`, `depth_layers: 2` | Phase 4 H2/H6 |
| STFT/waveform loss disabled | `decoded_loss_enabled: false` | Phase 4 exp02/exp03 (no detectable effect) |
| fp32 validation, 16 examples per (band, length), extra checkpoint chosen on crop SI-SDR | `validation_autocast: false`, `validation_examples_per_length_band: 16`, `audio_selection_metric: si_sdr_db` | Phase 4 fp16 gap, Phase 5 aspect 8 |
| Splice only the gap samples (2.5 ms cross-fade) into the original waveform | `inpaint_output_mode: splice` | Phase 5 condition E |

It bundles five changes, so a good result cannot attribute the gain to any single one (no ablations are planned in Phase 6). Light logging: scalars only, no figures (`tb_log_figures: false`).

## The preflight file (`results/preflight_zeroed_context.json`)

Question: how well does a **windowed** re-encode of a training window (needed because re-encoding the whole song per step is too slow) reproduce the whole-song zeroed-gap tokens that inference sees? 3 songs x 16 random windows.
Keys: `clean_tokens_vs_whole_song_zeroed` (`before` 0.998, `after` 0.739: clean tokens match before the gap, but only 74% match after it); `agreement_with_whole_song_zeroed_encoding[W]` for warm-up W = 0, 75, 150, 300, 750, 1500 frames, each with `before`/`after` token agreement (before: 0.651, 0.734, 0.764, 0.798, 0.841, 0.882; after: 0.832, 0.841, 0.846, 0.855, 0.878, 0.903);
`encode_batch16_seconds_W150/W750/W1500` (0.074, 0.140, 0.224 s on the GPU). Reading: higher agreement = closer to inference; longer warm-up costs time. Even W = 1500 does not reach 1.0 (the encoder state has long memory).
Design consequence: training uses the clean tokens up to the first frame touched by the zeroed span (exact, the encoder is causal) and the windowed encoding after it; W = 300 was a compromise (80% / 85.5%). Agreement is a token-identity rate, not an audio-quality measure.

## How to read the run's outputs when exported

Scalars in TensorBoard (`outputs/runs/phase6_final/.../tb`; also summarised by `uv run python phase_6/monitor.py`):
- `val/*` is the **primary set with zeroed-gap context** (what inference sees); `val_clean/*` is the same examples with clean-token context (comparable to Phases 1-5). The difference `val` minus `val_clean` measures the remaining context-mismatch cost.
- Validation is every 10k steps, fp32, 16 examples per (band, length) cell (64 per band, 128 total, vs 64 total earlier). With `depth_ar`, NLL is teacher-forced (exact joint likelihood) while accuracy comes from the greedy pass, so NLL and accuracy are measured under slightly different decoders.
- Checkpoints: `best_val.pt` (min `val/combined_loss`), `best_audio.pt` (max **crop** `val/si_sdr_db`, the margin-included version, which is dominated by unchanged context and can be insensitive to fill quality; check `val/si_sdr_db_gap_only` before trusting it), `best.pt` (min training loss), periodic `step_*.pt` every 50k.
- The monitor compares with the songs17 seed-family at 300k: NLL 4.9569 +/- 0.0568, gap-only SI-SDR -4.49 +/- 2.96 dB (`phase_3/results/seed_check/family_summary.csv`). A difference smaller than about one std is not evidence.

## Caveats to apply when comparing with Phases 1-5

1. The validation set differs (16 vs 8 examples per cell, fp32 vs fp16, zeroed vs clean context), so NLL values are not directly comparable to Phase 3. Use `val_clean/*` plus a re-evaluation of the Phase 3 checkpoint under the same fp32 setup (Phase 4 showed precision effects up to 0.33 nats for some checkpoints).
2. `depth_ar` changes what "NLL" means for later codebooks (conditioned on earlier true tokens in teacher forcing), so NLL improvements on codebooks 2-8 are expected by construction; confirm with greedy accuracy and audio metrics.
3. Real-gap audio from this run uses splice mode: the context is the original waveform, so samples are not comparable to earlier phases' decoded-window crops that include contaminated context. Compare gap-only SI-SDR against a clean-context reference (Phase 5 condition E logic).
4. One seed, one run: use the Phase 3 seed spread as the noise floor.
