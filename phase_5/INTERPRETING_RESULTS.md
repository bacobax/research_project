# Phase 5 (train/inference consistency, supervisor Q3): how to interpret the results

Shared definitions: [docs/interpreting_results.md](../docs/interpreting_results.md). Procedures: `METHODS.md`; the Q3 aspect table and conclusions: `README.md`.
No training; four single-song `best_val.pt` checkpoints: `baseline` (step 20k), `phase1_songs01` (10k), `phase2_regularization` (25k), `phase3_regularization_seed43` (300k).
Inference-time differences are measured on two gap sets: **held-out** (64 validation gaps per checkpoint: `set = synthetic`) and the **4 real gaps** (`set = real`, illustrative only).

## Core idea

In training the context is tokens of clean audio with a mask on the gap. At a real gap the audio is zeroed and the whole gapped file is encoded; the causal EnCodec encoder then changes tokens after (and for unaligned zeroing, just before) the gap,
and the gap-to-frame rounding leaves part of the silence outside the mask. The experiments separate these two effects and separate "worse predictions" from "corrupted context left in the output".

### Conditions (`condition` column in `exp01_*/records.csv`, `paired_effects.csv`)

| Code | CSV label | Meaning |
|---|---|---|
| A | `A_clean` | clean tokens, rounded mask: the training/validation-like reference |
| B | `B_aligned` | gap frames zeroed exactly on the frame grid: pure encoder contamination |
| C | `C_deployed` | real (or real-like random sub-frame) zeroing, rounded mask: **what inference does today** |
| E | `E_pred_only` | C's predicted tokens written into the clean tokens: isolates the quality of the predictions from the contaminated context in the output |
| D | `D_coverage` | as C, but the mask covers every frame the zeroing touches (floor/ceil), up to 2 frames longer |
| Z | `Z_unfilled` | gapped audio decoded without inpainting (reference floor) |

All targets and references come from the clean audio. Effects are **paired differences to A on the same example**, with a 95% interval 1.96 sd/sqrt(n) (n = 64 held-out, n = 4 real).

## Files (`results/`)

**exp01_context_contamination/**
| File | Columns / content |
|---|---|
| `records.csv` (1,616 rows) | `run,set,example,band,mask_len,filled_len,condition,nll,acc_top1,si_sdr_crop,si_sdr_gap,spectral_convergence_crop,snr_gap,log_mel_l1_crop`. Real set has 4 gaps x 4 runs x 5 conditions (no B); synthetic has 64 x 4 x 6. `nll` is blank only for Z (no inpainting). `filled_len` equals `mask_len` except for D (longer in 212 of 272 rows). `mask_len` = rounded mask frames, `filled_len` = frames actually replaced (differs for D). `*_crop` metrics are over gap +/- 16 frames, `*_gap` over the frames touched by the zeroing (1-6 frames, noisy). SI-SDR dB higher = better, log-mel L1 and spectral convergence lower = better. |
| `paired_effects.csv` | `run,set,condition,metric,n,mean_diff_vs_A,ci95`. **The table to quote.** Interval covers 0 => no detectable effect. |
| `real_gap_coverage.csv` | per real gap: `gap_ms` (silenced duration 50/20/30/15), `zero_start/zero_end` (samples), `mask_g0/g1` (frames), `touched_f0/f1`, `zeroed_samples`, `zeroed_outside_mask`, `frac_zeroed_outside_mask` (0.20, 0.333, 0.111, 0.272), `masked_samples_not_zeroed` (0 except gap 3: 58). Identical across runs. |
| `contamination_profile.csv` | `kind` (`real`, `synthetic_aligned`, `synthetic_reallike`), signed `distance` (frames from the mask, -64..64, -1 = frame just before, +1 = just after), `codebook` (1-8), `n` (gaps; 4 real, 64 synthetic), `mismatch_rate` = share of context tokens that differ from the clean encoding. Model independent. |
| `run_metadata.json` | `agreement_trainer_codes_vs_reencoded_gapped_source` = 0.9938 (sanity: the re-encoding matches the pipeline's tokens), `real_far_mismatch_rate` = 0.018 (tokens more than 64 frames from any gap that still differ). |
| `figures/*` | see below |

**exp02_gap_position/**: `records.csv` (`run,band,gap_start,mask_len,window` = `train_512` (gap placed at `position` in a 512-frame window) or `inference` (the centred inference-style window, 493-496 frames), `position`, `window_len`, `nll`, `acc_top1`); `summary.csv` (`band` high/low/all, `comparison` e.g. `edge_8 minus centre_248 (512 window)`, `metric`, `n`, `mean_diff`, `ci95`; paired, pooled over checkpoints); `train_position_distribution.json` (histogram of the local gap start in 20,000 training draws; `share_within_32_of_centre` 0.127-0.131).

**exp03_consistency_checks/**: `checks.json` (per run: `iterations` per real gap with `identical_1_vs_10`; `precision_heldout`/`precision_real_gaps` with `argmax_agreement`, `mean_nll_fp16_minus_fp32`, `max_abs_example_dnll`; `activity_summary` incl. `real_gap_percentile_in_training`), `selection_alignment.csv` (per run: `n_points`, `spearman_nll_vs_sisdr`, `p_value`, `step_best_nll`, `step_best_sisdr`, `sisdr_at_best_nll`, `best_sisdr`, `sisdr_lost_by_selecting_on_nll`), `train_gap_activity_draws.json` (20,000 training gap activities per run).

## Verified key numbers

- Contamination (real-like synthetic and real): share of changed tokens at +1 frame 0.84 (synthetic, real-like), 0.86 (aligned), 0.94 (real); at +8 frames about 0.42-0.53; at +64 about 0.19-0.30. Before an **aligned** gap it is exactly 0 (causal encoder); before a real-like/real gap 0.44 / 0.91 at -1 because the zeroed span starts inside the preceding frame.
- Effect on predictions, C minus A, NLL (held-out): -0.027 to +0.022 nats, all intervals include 0. Real gaps: -0.04 to -0.12 with intervals +/-0.12-0.48 (n = 4).
- Effect on output audio, C minus A, crop SI-SDR: held-out -1.68 (baseline), -2.25, -2.04, -2.40 dB (intervals +/-0.4-0.6); real gaps -3.49, -4.20, -2.77, -3.37 dB (intervals +/-1.1-3.5).
- E minus A (held-out) -0.01 to +0.34 dB; real -0.12 to +0.35 dB: the loss disappears when the same predictions are put in clean context.
- D minus A: -2.8 to -3.4 dB held-out (-2.82, -3.30, -3.09, -3.36), -4.2 to -6.0 real. Widening the mask is worse.
- Gap position: edge_8 minus centre NLL +0.024 +/- 0.017; inference window minus centre -0.007 +/- 0.009 (all bands).
- Iterations: tokens identical after 1 and 10 passes at every real gap for all four checkpoints. Precision: argmax agreement 99.4-99.9% held-out for three checkpoints, 82.3% for `phase3_regularization_seed43` (95% at real gaps; NLL change +0.003).
- Selection: Spearman(NLL, gap SI-SDR) +0.78 (songs01, p = 0.008), +0.22, -0.53, -0.20, -0.08 for songs01, phase2, reg seed43, seed44, stft-off; SI-SDR lost 2.56, 1.95, 1.76, 1.40, 0.65 dB (upper bound).

Sign of the Spearman: positive means higher NLL goes with higher (better) SI-SDR, i.e. NLL and audio quality move in **opposite** directions, so a good NLL point is not a good audio point.

## Reading the figures (each as PNG and PDF)

- `contamination_profile`: left, share (%) of context tokens changed vs frames from the gap for real, real-like and aligned zeroing (secondary axis in ms); right, a codebook x distance heat map for case C. Zero before the gap for aligned zeroing, a long decaying tail after it.
- `real_gap_mask_coverage`: one panel per real gap, time axis in ms with vertical lines at 13.3 ms frame boundaries; shows the silent span against the rounded mask and the full-coverage mask. Title states the percentage of silent samples outside the mask.
- `effect_heldout_gaps`, `effect_real_gaps_vs_clean`: change vs A for B, C, E, D, Z with 95% bars, three panels (token NLL on the gap, crop SI-SDR, crop log-mel L1), x = condition, one grey marker shape per checkpoint, a coloured bar for the mean over checkpoints. Bars crossing 0 = no detectable effect. 0 on the y-axis means "as good as clean context". Real-gap plot has n = 4: bars are wide.
- `real_gaps_quality`: two panels (SI-SDR, log-mel L1), x = real gap (with its ms), series A, C, D, Z, mean over checkpoints with min/max bars (n = 4 checkpoints per point).
- `train_vs_inference_gap_position`: histogram of training local gap starts vs the inference position (centre). `quality_vs_gap_position`: NLL and accuracy of identical gaps at 10 positions plus the inference window, per band; flat lines = position does not matter.
- `selection_nll_vs_audio`: one panel per run, x = validation NLL, y = gap-only SI-SDR, points coloured by training step, a square marks the point chosen by min NLL, a triangle the best-SI-SDR point; panel title shows Spearman.
- `gap_activity_train_vs_real`: histogram of training gap activity with the 4 real gaps marked (percentiles 12, 26, 29, 65 per `README.md`).

## What to trust

- Strong (paired, n = 64, consistent across 4 checkpoints): the 2-4 dB audio loss from contaminated context and that it is absent in condition E; the null effect on token NLL; the null effect of position and iterations.
- Illustrative: all real-gap rows (n = 4) and `real_gap_percentile`.
- Caveats: the causal-encoder explanation is inferred from the profile, not checked in EnCodec's code; held-out zeroing is synthetic; single-song checkpoints only; the splice fix is supported by E but was not measured on raw waveforms in Phase 5 (Phase 6 implements it). Selection loss is optimistic (max of noisy points).
