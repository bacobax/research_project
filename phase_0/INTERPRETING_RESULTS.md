# Phase 0: how to interpret the results

Shared metric definitions: [docs/interpreting_results.md](../docs/interpreting_results.md).

## What Phase 0 asks

Two single-song runs (`baseline`: d=256, 4 layers; `bigmodel`: d=384, 6 layers) showed high train accuracy and about 1-10% validation accuracy.
Before accepting "inpainting does not generalise", Phase 0 evaluates saved checkpoints (no training) to answer three questions:

| Exp | Question | Logic |
|---|---|---|
| exp01 | Is validation built correctly? | Re-run `run_validation` on windows drawn from **training** territory ("train-region control"). If accuracy there matches the training loop's own train accuracy and is far above holdout accuracy, the pipeline is sound and the gap is a real generalisation gap. If control ~ holdout, validation is buggy. |
| exp02 | Does the model beat non-learned fillers in token space? | Same 64 validation examples, 5 fillers + model, token top-1. |
| exp03 | Do token differences matter in audio? | Decode each fill and score the gap+16-frame-margin crop. |
| exp04 | How well are the 4 real gaps filled? | Real gaps scored against the encode-decode of the ungapped source. |

Checkpoints: `baseline` best_val (step 20,000) and best (78,000); `bigmodel` best_val (20,000) and best (108,000). Post-divergence checkpoints (NaN weights) are excluded.
Nothing under `outputs/runs/**` is touched; scratch output goes to `results/scratch/`.

## Files

| File | Content |
|---|---|
| `results/FINDINGS.md` | Generated summary tables; every number is read back from the JSONs. Start here. |
| `results/summary.csv` | Long format `experiment,run,checkpoint,step,band,mask_len,method,metric,value`. `band` in {high, low} for exp01, {high_activity, low_activity} for exp02/03, `real_gap` for exp04; `mask_len` = frames (empty for exp01). Slice with pandas. **Note** `scripts/` for later phases also use `phase_0/common.py`; this file only contains Phase 0 rows. |
| `results/exp01_train_region_control.json` | One record per checkpoint: `train_acc_top1_reported` (nearest logged `train/acc_top1` in the real TensorBoard file), `holdout` and `control` dicts of `val/*` scalars, `holdout_combined_loss`, `control_combined_loss`, `holdout_ranges` (frame ranges excluded from training). |
| `results/exp02_baselines.json` | 192 rows: (run, checkpoint, group e.g. `high_activity_len_3`, method) -> `acc_top1` (mean over `n_examples` = 8). |
| `results/exp03_decoded_metrics.json` | 1,536 per-example rows (2 runs x 2 checkpoints x 8 groups x 8 examples x 6 methods): `wave_l1`, `wave_mse` (waveform units, lower better), `si_sdr_db`, `snr_db` (dB, higher better), `log_mel_l1` (80-mel log L1, lower), `stft_total`, `stft_spectral_convergence`, `stft_log_magnitude` (multi-res STFT loss and its parts, lower). |
| `results/exp04_real_gaps.json` | 4 gaps x 4 checkpoints x 6 methods rows: the exp03 metrics plus `gap_index`, `gap_len_frames`, `gap_len_ms`, `acc_top1`, `masked_tokens` (= 8 x frames). |
| `results/audio_samples/{baseline,bigmodel}/{best_val,latest}/gap{0-3}_step_<N>.wav` | Real-gap fills (see the glossary). Here `latest` is step 162,000 (baseline, which diverged to NaN after 78k; the file exists but treat it as a post-divergence artefact) and 108,000 (bigmodel). `results/audio/` is empty. |
| `results/scratch/` | Working directory for every phase's trainer reconstruction (songs.json, TensorBoard stubs, samples). **Not results**; ignore. |

### Methods (exp02-04)

`repeat_left` copies the last frame before the gap into every gap frame; `repeat_right` copies the first frame after; `nearest` copies the nearer side;
`random_tokens` draws uniformly; `codec_ceiling` returns the true tokens (what a perfect fill would score in this token space).
All fillers read only tokens outside the gap.

## How to read each result

**exp01.** Compare three numbers per row of `FINDINGS.md`: the training loop's `train_acc`, `control acc`, `holdout acc` (high/low band).
Verified in `FINDINGS.md`: baseline best (step 78,000) train 0.609, control 0.936/0.484, holdout 0.007/0.097; bigmodel best control 0.974/0.914 vs holdout 0.006/0.101.
Holdout combined loss 5.34 vs control 3.32 for baseline best_val (`exp01_train_region_control.json`). The control tracks train, not holdout, in all 4 cases, so the pipeline is sound.
Caveat: "control" accuracy is on training regions, hence a memorisation measure; "|control-train|" in FINDINGS is a loose tolerance, not a statistical test.
The exact reproduction of a logged TensorBoard number (baseline step 20,000 `val/high_acc_top1` = 0.009765625) is the strongest evidence of a faithful rebuild.

**exp02.** Mean top-1 over all bands and lengths: repeat_left 0.0547, repeat_right 0.0483, nearest 0.0539, random 0.0000 (chance 0.001), ceiling 1.0, model 0.0514-0.0607.
Interpretation: on held-out passages the model is statistically indistinguishable from copying a neighbouring frame (it beats `repeat_left` in 1 of 4 checkpoints). n = 8 per group and 8 groups, so differences of 0.01 are noise.

**exp03.** FINDINGS tables show mean SI-SDR and spectral convergence. Model 9.95-10.92 dB vs repeat/nearest 9.9-10.5 dB and random -3.26; ceiling 59.5 dB.
Read: the model is not better than trivial fillers. Caveats: (i) the crop includes a 16-frame margin of identical tokens, so all methods share most of the signal and SI-SDR differences are compressed
(later phases add a gap-only variant for this reason); (ii) the 60-73 dB "ceiling" is a numerical floor (eps 1e-9 and float rounding), not a meaningful quantity;
(iii) `random_tokens` spectral convergence 3.75 > 1 because noise energy exceeds the reference.

**exp04.** FINDINGS tables list SI-SDR per gap (rows in gap_index order 0..3; "gap (ms)" there is `gap_len_ms` = the **mask span** 40, 13, 27, 13 ms, not the 50/20/30/15 ms silenced).
The model is at 9-17 dB and above `repeat_left` in most (gap, checkpoint) cells (e.g. baseline best_val gap 0: 11.80 vs 6.05), but not all (bigmodel best_val gap 2: 11.45 vs 13.86; bigmodel gaps 0 at 9.3-9.6 dB).
This contrasts with exp03's no-better-than-copy result on synthetic masks; with 4 gaps this is not evidence of a real advantage.
Only 4 gaps, 8-24 tokens each: individual `acc_top1` values (0, 1/8...) are anecdotal. Context is tokens of the gapped file; target and prediction share it, so context contamination (Phase 5) does **not** penalise these scores,
unlike Phase 5's clean-context targets.

## What to trust

- Trust exp01 (pipeline correct) and the qualitative reading of exp02 (model not above copy baselines).
- Treat exp03/exp04 absolute dB values as indicative only: margin dilution, tiny n, one seed.
- Known documentation inconsistency: `phase_0/README.md` and `exp04_real_gaps.py` docstring describe the gaps as "15/20/30/50 ms at frames [4,2,2,1]". The data (JSON, annotations, Phase 5 coverage CSV) give frames [3,1,2,1] and durations by index 50/20/30/15 ms.
- `FINDINGS.md` is regenerated by `make_summary.py`; edit nothing by hand.
