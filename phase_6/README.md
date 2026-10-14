# Phase 6 — one final run that accounts for what Phases 0–5 found

Plan: `~/.claude/plans/now-it-would-make-hidden-eich.md`. Run config: `configs/train/phase6_final_songs17_seed43.yaml`
(17 songs, seed 43, 300k steps, so it is comparable to the `songs17` seed family). Smoke config: `configs/train/phase6_smoke.yaml`.

| Change | Config key | Motivation |
|---|---|---|
| Context encoded from audio with the gap zeroed (as validation/inference do) | `zeroed_context_prob: 1.0`, `zeroed_context_warmup_frames: 300` | Phase 5 exp01 |
| Codebook k's head sees the tokens of codebooks < k (teacher forced in training, greedy at inference) | `codebook_conditioning: depth_ar`, `depth_layers: 2` | Phase 4 H2 / H6 |
| STFT/waveform loss off | `decoded_loss_enabled: false` | Phase 4 exp02/exp03 |
| fp32 validation, 16 examples per (band, length), extra checkpoint `best_audio.pt` selected on crop SI-SDR | `validation_autocast: false`, `validation_examples_per_length_band: 16`, `audio_selection_metric: si_sdr_db` | Phase 4 fp16 gap, Phase 5 #8 |
| Only the gap's samples replace the input audio (2.5 ms cross-fade) | `inpaint_output_mode: splice` | Phase 5 cond. E |

Defaults of every new key reproduce the previous behaviour.

## Zeroed-context details (pre-flight: `results/preflight_zeroed_context.json`)

`phase_6/preflight_zeroed_context.py`: 3 songs x 16 random windows, token agreement with the whole-song zeroed encoding
(what inference does).

| context tokens | before the gap | after the gap |
|---|---|---|
| clean whole-song tokens (today's training) | 99.8% | 73.9% |
| windowed re-encode, warm-up 0 / 75 / 300 / 1500 frames | 65 / 73 / 80 / 88% | 83 / 84 / 85.5 / 90% |

EnCodec's encoder state has a long memory, so a windowed encode never reproduces the whole-song state exactly.
Therefore training uses the clean tokens before the first frame touched by the zeroed span (exact, the encoder is causal)
and the windowed encoding of the frames from there on (closer to inference than clean tokens). Warm-up 300 was chosen as a
compromise: encoding a batch of 16 takes 0.07 s (W=150), 0.14 s (W=750), 0.22 s (W=1500) on `cuda:1`.

Validation: the primary `val/*` set has zeroed-gap context; the same examples with clean context are logged as `val_clean/*`.

## Results (`results/eval/FINDINGS.md`, figures in `results/eval/figures/`)

`eval_final.py` re-scores every checkpoint under one protocol (128 held-out gaps, fp32, deployed = zeroed-gap
context, same examples) because the training-time validation curves are not comparable (zeroed/fp32/128 examples
vs clean/fp16/64). Final checkpoints, Phase 6 minus the mean of songs17 seeds 43/44 (paired, 95% CI):

| metric | difference | beyond seed spread |
|---|---|---|
| token NLL | +0.085 +- 0.033 (worse) | yes |
| top-1 accuracy | -0.034 +- 0.014 (worse) | yes |
| gap SI-SDR vs original audio | -5.6 +- 1.8 dB (worse) | yes |
| region SI-SDR, spliced output | -0.96 +- 0.34 dB (worse) | yes |

- **P1** the combined recipe is not better than the baseline; slightly worse. (The 2-seed spread is a weak noise estimate.)
- **P2** training on zeroed context changed robustness little: NLL C-A 0.003 vs 0.024; the 2-3 dB decoded-audio loss from
  contaminated context is the same (-2.4 vs -2.7 dB).
- **P3** per-codebook NLL slope is flatter (+0.13 vs +0.17, high band) only because codebook 1 got worse; codebook 8 did not improve.
  Greedy decoding loses ~25% of top-1 vs teacher-forced (exposure bias), and even the teacher-forced accuracy only equals the baseline's greedy.
- **P4** splice gains +6 dB region SI-SDR over decoding the whole window (Phase 5 predicted 2-4 dB), for every checkpoint.
- **P5** selection: `best_val` (step 30k) is the worst Phase 6 checkpoint for audio; `best_audio` (crop SI-SDR, step 210k) is not the best either
  (that metric barely moves). Audio keeps improving after NLL stops: baseline s42 at 1M steps has gap SI-SDR -2.4 dB vs -6.2 dB at 300k.
- **Limits:** one Phase 6 seed, four changes at once (cannot attribute), 2 baseline seeds, held-out gaps from one song.
  Untested explanations: exposure bias of the depth heads, removal of the STFT loss at 17 songs (4A only tested single-song), noisier context.

## Status

Run finished 2026-10-13 17:58 (300k steps, `save_every` 50000); evaluation in `results/eval/`. Monitor script: `monitor.py`.
