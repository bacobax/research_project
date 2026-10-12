# Phase 2 (regularization ablation): how to interpret the results

Shared metric definitions: [docs/interpreting_results.md](../docs/interpreting_results.md). File layout is identical to Phase 1 (see
[phase_1/INTERPRETING_RESULTS.md](../phase_1/INTERPRETING_RESULTS.md)); only the differences are described here.

## Experiment and logic

Single song (pivot only), one run `phase2_regularization`, compared with `phase1_songs01`. Changes (verified by diffing the two configs):
`d_model` 256 -> 128 (4.09 M vs 11.84 M parameters), dropout 0.1 -> 0.3, weight decay 0.01 -> 0.1, `total_steps` 100k -> 600k, validation every 25k (not 10k),
`grad_clip` 1.5 -> 1.0, `decoded_loss_log_eps` 1e-3 (rerun). Everything else (LR, batch, activity sampling, loss weights) is identical.
Hypothesis: if the one-song model overfits only because it has too much capacity/too little regularization, a smaller, regularized model should close the held-out gap.
Because several levers change together this tests "regularization as a bundle", not any single lever, and the longer budget and different validation cadence mean the best steps (10k vs 25k) are not on the same grid.

## Files

Same names as Phase 1, with these additions/changes:
- `run_summary.csv` has extra columns `d_model`, `n_layers`, `dropout`, `weight_decay`, `best_si_sdr_db` (margin) and `best_si_sdr_db_gap_only` taken **at the best-NLL step**.
- `figures/validation_audio_metrics.{png,pdf}`: left panel SI-SDR (dB), right panel spectral convergence, both vs step. Solid lines with circles = margin-included; dashed with squares = gap-only. Dotted grey line at 0 dB. Margin SI-SDR is about 11 dB throughout and barely moves (it is dominated by the unchanged context); the gap-only curve is the informative one (-9.9 to -6.7 dB over the run).
- `README.md` in this folder is exporter-generated; it also says qualitative figures were not generated although `qualitative_{high,low}_activity_len_4.*` exist.
- No `best_by_group` differences in format: cells are n = 8.

## Verified numbers (`run_summary.csv`, `metrics.csv`)

Best validation step 25,000: combined NLL 4.9951, margin SI-SDR 10.99 dB, gap-only SI-SDR -8.67 dB, high top-1 0.0101, low top-1 0.1064. Stop step 600,000, completed, 94,283 s.
Compared with `phase1_songs01` (NLL 5.1398 at 10k; gap-only about -9.1): the regularized run is 0.145 nats lower in NLL, with similar gap-only SI-SDR.
Trajectory: NLL minimum 4.995 at the first validation, maximum 5.498, ends 5.111 after a late three-checkpoint recovery (550k-600k: 5.181, 5.162, 5.111); gap-only SI-SDR improves to -7.1 at 600k (best of run -6.7) even though NLL is worse than at 25k.

## Reading the figures

- `validation_loss`: one line, star at best step (25k, the very first point). Lower is better. The shape is "best immediately, then worse, partial recovery" (overfitting plus drift), not a convergence curve. The late recovery is real in the log but the best checkpoint stays early.
- `validation_accuracy`: high band 0.5-3%, low band 7.6-13.6%; points are noisy.
- `training_loss`: token + decoded loss: compare with Phase 1's training curve only with care, since the models differ in size and the decoded loss term enters after step 7000.
- `best_by_gap_length`, `qualitative_*`: as in Phase 1.

## What to trust

- The comparison with `songs01` is single-seed and budget-mismatched; Phase 3 shows regularization-family seed std 0.02 nats, 0.9 dB, so a 0.145 nat difference to songs01 is plausible but songs01 itself has no seed replicate.
- Selecting on NLL picks step 25k, whereas gap-only SI-SDR keeps improving to 600k (Phase 5 aspect 8: Spearman +0.22 between NLL and gap SI-SDR for this run, p = 0.30).
- Do not describe "regularization closes the gap": validation NLL never gets below 4.995 and mostly rises afterwards (Phase 4 H4: last minus best = +0.12).
