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

## Status

Implemented and tested (`tests/test_codebook_conditioning.py`, `test_zeroed_context.py`, `test_inpaint_splice.py`, schema
test); smoke run passes (`best_val.pt`, `best_audio.pt`, splice crop samples). **The 300k-step run was launched 2026-10-12 12:26** (`outputs/runs/phase6_final/phase6_final_songs17_seed43`, `save_every: 50000`). Monitor with `uv run python phase_6/monitor.py` (or `watch -n 300 ...`). Observed speed ~2.7-2.9 it/s, i.e. ~28-31 h; the GPU is only ~12% utilised, so the run is CPU/latency-bound (the per-step EnCodec re-encode of 16 windows), on a shared machine.
