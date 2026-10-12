# Phase 1 (data scaling): how to interpret the results

Shared metric definitions and validation design: [docs/interpreting_results.md](../docs/interpreting_results.md).

## Experiment and logic

Three runs differ only in the training data and budget; the validation set (pivot song held-out regions) is identical:

| run | songs | total minutes | total frames | steps | grad_clip | params |
|---|---|---|---|---|---|---|
| `phase1_songs01` | 1 (pivot) | 6.67 | 30,017 | 100,000 | 1.5 | 11.84 M |
| `phase1_songs04` | 4 | 20.58 | 92,593 | 300,000 | 1.0 (rerun config) | 11.84 M |
| `phase1_songs17` | 17 (all `dataset/` except `008_*`, a second recording of the pivot piece that would leak) | 75.27 | 338,735 | 1,000,000 | 1.0 | 11.84 M |

Hypothesis: more music lowers held-out loss and raises accuracy and audio quality. Conclusion is read from `val/combined_loss`, per-band accuracy and the audio metrics at each run's best-validation step.
Budgets are not matched (100k / 300k / 1M), so "more songs" is confounded with "more steps"; Phase 3 partially addresses this with a matched 300k cutoff.
Config differences visible in `configs/*.yaml`: `songs04` (rerun) and `songs17` use `decoded_loss_log_eps: 0.001` and `grad_clip: 1.0`; the `songs01` snapshot has the old 1e-7 default and clip 1.5 (`docs/RERUN_PLAN.md` documents the STFT epsilon issue; Phase 4 exp02 later showed the STFT term is only 1.5-6.5% of the gradient norm and the rerun did not change validation NLL).

## Files in `results/`

| File | Content |
|---|---|
| `README.md` | Exporter-generated summary. Its statement "Qualitative figures: not generated" is stale: `figures/qualitative_{high,low}_activity_len_4.{png,pdf}` exist. |
| `run_summary.csv` | One row per run: `num_songs`, `total_minutes`, `model_params_m`, `best_val_step` (step with minimum `val/combined_loss`), `best_combined_loss` (nats), `best_high_acc_top1`, `best_low_acc_top1` (at that step), `stop_step`, `stop_reason` (all `completed`), `runtime_seconds`. |
| `best_by_group.csv` | At the best step: per run x `activity_band` x `mask_len_frames` (1-4) with `gap_ms` (13.3/26.7/40/53.3): `loss`=`nll`, `ppl`, `acc_top1`, `acc_top5`. 24 rows. Each cell has 8 examples. |
| `metrics.csv` | Long format `run,category,tag,step,wall_time_utc,value`. Categories: `validation_metric` (`val/*`, every 10,000 steps: 140 points per tag over three runs: 10 + 30 + 100), `training_aggregate` (`train/avg_*`, every 1,000 steps; decoded-loss items every 4,000), `dataset_metadata` (`dataset/*`: song counts, per-song weights/starts, `holdout_blocked_fraction` 0.149, regime probabilities 0.35/0.30/0.20/0.15), `model_metadata`, `validation_metadata` (`validation/*`: counts and mean activity per group, e.g. high-band region activity 0.887, low 0.267). |
| `tables/*.tex` | LaTeX versions of the two summary CSVs. |
| `figures/*.png/pdf` | See below. |
| `configs/`, `song_manifests/` | Exact YAMLs and the list of songs (durations, frames, real-gap frame ranges) per run. |
| `provenance.json`, `source_state.patch` | Git commit, hashes of every raw artifact, uncommitted source diff, for reproducibility. |
| `audio_samples/<run>/{latest,best_val}/gap{0-3}_step_<N>.wav` | See glossary: songs01 steps 10,000/100,000; songs04 20,000/300,000; songs17 950,000/1,000,000. |

## Verified headline numbers (`run_summary.csv`)

| songs | best step | combined NLL | high top-1 | low top-1 |
|---|---|---|---|---|
| 1 | 10,000 | 5.1398 | 0.0075 | 0.0843 |
| 4 | 20,000 | 5.0697 | 0.0173 | 0.0986 |
| 17 | 950,000 | 4.8695 | 0.0612 | 0.1657 |

Further, from `metrics.csv`: songs01 val NLL rises to 6.17 by step 100,000 (overfits within 10k steps); songs04 ends at 5.36; songs17 ends at 4.94 (up from its 4.87 minimum).
Gap-only SI-SDR (dB) at the end of training: songs01 -8.4, songs04 -5.8, songs17 -0.9 (best of run 0.58), vs -9.1 / -9.8 / -11.6 at the first validation. Margin SI-SDR: 11.2 / 11.8 / 13.7.
Spectral convergence gap-only: 0.71 / 0.66 / 0.62.

## Reading the figures

All curves are one seed per condition; markers are validation points (every 10k steps).
- `validation_loss`: combined NLL vs step, one line per song count; a star marks each run's best step, an x its stop step. Lower is better. Songs01's curve is U-shaped (best at the first checkpoint); songs17's still improves slowly until 950k. Do not read the vertical gap between lines at a fixed step as an "effect of data" without noting the very different run lengths.
- `validation_accuracy`: two panels (high / low activity band) of top-1. High band stays near 1-6%; low band 8-17%. Values are shared-axis. Individual points are means over 32 examples x 8 codebooks and jump by a few percent.
- `training_loss`: `train/avg_loss` (token CE plus 0.1 x decoded loss after step 7000, which adds a small step) vs step. Lower training loss for fewer songs is expected (memorisation) and is **not** better quality; the gap between train and validation is the overfitting signal.
- `best_by_gap_length`: 2x2 grid. Rows = high/low activity, columns = NLL (left) and top-1 (right), x = gap duration in ms, one line per run at its own best step. Cells have n = 8, so the lines zig-zag; the consistent signal is the band difference and songs17 sitting lowest.
- `qualitative_{high,low}_activity_len_4`: 2 x 3 spectrograms (target decode on the top row, prediction on the bottom) for the same held-out 4-frame example at each run's best step. Judge visually whether harmonic lines continue across the gap; one example per panel, illustrative only.

## What to trust

- Directionally robust: songs17 is best on NLL, accuracy and gap-only SI-SDR; songs01 overfits quickly.
- Not established by Phase 1 alone: effect sizes. One seed; Phase 3 shows seed std 0.057 nats and about 3 dB for songs17 at 300k, comparable to the songs04/songs17 differences at matched budget.
- The best step is a minimum over noisy points (songs04's best is its second validation point); metrics at the best step are optimistically selected.
- High-band top-1 of 6% means the model is still near the "copy a neighbour" baseline from Phase 0 for loud material.
- Token NLL at validation was computed under fp16 autocast (Phase 4 found up to 0.33 nats difference vs fp32 for `phase1_songs17`).
