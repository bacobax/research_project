# Phase 3 (seed robustness and gap-length sweep): how to interpret the results

Shared metric definitions: [docs/interpreting_results.md](../docs/interpreting_results.md).

## Two sub-experiments

**A. Seed check (`results/seed_check/`)** tests whether "more data (Phase 1, 17 songs) beats regularization alone (Phase 2)" survives seed noise.
Six runs: families `songs17_data_scaling` and `regularization_ablation`, seeds 42 (the original runs, 1M and 600k steps) and 43, 44 (new, 300k steps, own cosine schedule; configs differ from their parents only in `seed` and `total_steps`).
Fairness device: every run is scored by its best `val/combined_loss` at or before **step 300,000** (`analyze_run_at_cutoff`), so the seed-42 runs are *not* represented by their headline minimum (4.87 at 950k for songs17).
Decision logic: compare the gap between family means with the seed-to-seed standard deviation (sample std, n = 3).

**B. Gap-length x activity sweep (`results/exp01_*.json`, `FINDINGS.md`, `summary.csv`, `figures/`)** re-evaluates the two headline `best_val.pt` checkpoints (songs17 step 950,000; regularization step 25,000)
on gaps of 1, 2, 4, 8, 16 frames (13.3-213 ms) in both bands, 8 examples per cell. Training used 1-4 frames only, so **8 and 16 frames are out of distribution** (column `ood_beyond_trained_range`); a decline there says how the model fails outside its training range, never that it generalises.

## Files

| File | Content and columns |
|---|---|
| `results/README.md` | Prose write-up. **Stale numbers**: it quotes regularization gap-only SI-SDR -8.41 +/- 1.24 dB and NLL 5.000 +/- 0.017; the generated `seed_check/family_summary.csv` says -8.55 +/- 0.90 dB and 5.0023 +/- 0.0198. Trust the CSV. The conclusions are unchanged. |
| `seed_check/per_run_at_cutoff.csv` | `run,family,seed,is_original,cutoff_step,best_step,best_combined_loss,best_high_acc_top1,best_low_acc_top1,best_si_sdr_db_gap_only,stop_step,stop_reason`. All metrics are taken **at the step of minimum combined NLL** (not each metric's own best). |
| `seed_check/family_summary.csv` | mean and sample std over the 3 seeds: `mean_/std_combined_loss`, `_high_acc_top1`, `_low_acc_top1`, `_si_sdr_db_gap_only`. |
| `seed_check/metrics.csv`, `provenance.json`, `tables/family_summary.tex`, `README.md` | all scalar points <= 300k for the six runs; source hashes; LaTeX; seed-check README. |
| `seed_check/figures/seed_check_combined_loss.{png,pdf}` | validation NLL vs step up to 300k, colour = family, linestyle = seed, star = each run's best-at-cutoff point, dotted vertical line = cutoff. |
| `exp01_gap_length_activity_sweep.json` | 20 rows (2 runs x 2 bands x 5 lengths): `loss,nll,ppl,acc_top1,acc_top5,si_sdr_db,si_sdr_db_gap_only,spectral_convergence,spectral_convergence_gap_only`, `gap_ms`, `ood_beyond_trained_range`. |
| `summary.csv` | The same in long form (`experiment,run,checkpoint,step,band,mask_len,method,metric,value`), `method` = `model` only (no baselines). |
| `FINDINGS.md` | Generated tables plus one-line trend descriptions. |
| `figures/gap_length_degradation.{png,pdf}` | 2x2 grid: rows = high/low activity, columns = top-1 accuracy and gap-only SI-SDR, x = gap length (ms), one line per run, dashed vertical line at the end of the trained range (53 ms, labelled OOD beyond). Spectral convergence is only in the JSON/FINDINGS. |
| `audio_samples/phase3_{songs17,regularization}_seed{43,44}/{best_val,latest}/gap{0-3}_step_<N>.wav` | Real-gap fills for the four new runs (the seed-42 originals are in Phases 1 and 2). |

## Verified numbers

Seed check (`per_run_at_cutoff.csv`, best NLL at <= 300k): songs17 seeds 42/43/44 = 4.908 (step 260k), 5.019 (step 20k), 4.944 (step 280k); regularization = 4.995 (25k), 5.025 (300k), 4.987 (25k).
Family means: songs17 4.9569 +/- 0.0568 nats, high top-1 0.0269 +/- 0.0205, SI-SDR gap-only -4.49 +/- 2.96 dB; regularization 5.0023 +/- 0.0198, 0.0183 +/- 0.0077, -8.55 +/- 0.90 dB.
Reading: the NLL gap between means (0.045) is smaller than songs17's own std (0.057): **not a seed-robust NLL claim**. The SI-SDR gap (about 4 dB) exceeds the regularization std but is comparable to songs17's std (one songs17 seed, 43, selected at step 20k, has -7.87 dB, similar to the regularized runs) so it is a real but noisy separation with n = 3.
Note one songs17 seed is selected at 20k: the minimum-NLL step is often early and does not reflect the long-run model.

Sweep (`FINDINGS.md`): gap-only SI-SDR songs17 high band 0.38 (13 ms), -4.29, -3.49 (53 ms), -12.54 (107 ms, OOD), -26.92 (213 ms, OOD); low band 1.82, 4.65, -1.63, -9.70, -19.76. Regularization high band -12.32, -15.60, -22.67, -23.29, -31.55; low band 3.67, 0.84, -7.62, -23.09, -24.18.

## How to read the figures

- Seed-check figure: curves that cross or overlap at 300k mean seeds differ more than families. Look at the vertical spread of same-colour lines (seed noise) against the separation between colours. The regularization seeds 43/44 show a mid-run worst point (around 150-275k) followed by partial recovery.
- Degradation figure: higher accuracy / SI-SDR is better; SI-SDR lines should fall left to right. Non-monotonic wiggles (e.g. songs17 high band -4.29 at 27 ms vs -3.49 at 53 ms) are within the noise of 8 examples. Accuracy columns in `FINDINGS.md` are not monotone either (low-band top-1 rises at 8 frames for both models); do not read that as improvement, it reflects easy quiet material and small n.
- The margin-included `si_sdr_db` in the JSON (e.g. 18.3 dB for songs17 high band 1 frame) is mostly unchanged context; use `si_sdr_db_gap_only`.

## What to trust

Trust: the existence of strong degradation beyond 53 ms; that songs17 is above regularization on gap-only SI-SDR at nearly every cell (a robust ordering, though per-cell n = 8); that NLL differences between families are within seed noise at 300k.
Do not trust: any single sweep cell, absolute SI-SDR at one gap length, or the OOD rows as generalisation evidence. The sweep has no seed replicates and no non-learned baseline.
