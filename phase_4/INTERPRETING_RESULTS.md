# Phase 4 (why does it underfit?): how to interpret the results

Shared definitions: [docs/interpreting_results.md](../docs/interpreting_results.md). Procedures: `METHODS.md`. Headline text: `README.md` and the generated `results/FINDINGS.md`.
No training happens in exp01/exp02; exp03 post-processes the logs of one extra training run (`phase4_stft_off_seed43`, trained elsewhere).

## Hypotheses and how conclusions follow

| Exp | Hypothesis / question | Evidence that would support it | What was found (verified) |
|---|---|---|---|
| H1 | Per-codebook heads collapse to the context-free token marginal ("mode averaging") | model NLL not below a predictor that always outputs the train marginal; KL(model||marginal) near 0; argmax = most frequent token | Not supported: model NLL beats the marginal predictor on 8/8 codebooks in every run and band (`FINDINGS.md`). But in the high band the margin is small (marginal 7.2-7.6 vs model 6.4-7.0) and predicted entropy is 0.89-1.07x the marginal's; in the low band entropy is about 0.5x. |
| H2 | Later RVQ codebooks are worse (independent heads cannot use residual structure) | NLL/accuracy decline with codebook index | NLL slope per codebook > 0 in 9/9 high and 8/9 low run-band pairs; top-1 on codebook 8 is 0.000 in most high-band runs. **Consistent but not proof**: residual codebooks are intrinsically harder; the discriminating test (condition codebook k on codebooks < k, "H6") was not run in Phase 4 (Phase 6's `depth_ar` implements it). |
| H3 | The model does not use context near the gap | uniform cross-attention | Rejected: attention at distance 1 frame is 5x to 860x the weight at >= 257 frames in 30 of 32 (run, layer) pairs of the first 8 runs; layer 0 of two 17-song runs is about 1x. Only 2 examples per band. |
| H4 | Held-out loss has not converged | tail slope, last vs best | No run meets the plateau rule (|slope| < 0.01 nats/100k and last within 1% of min). Last minus best: songs01 +1.03, songs04 +0.29, phase2 +0.12, 17-song runs +0.01 to +0.07. Overfitting is present on top of underfitting. |
| fp16 check | Is validation precision a confounder? | H2's pooled fp32 numbers vs the logged fp16 numbers | Max abs diff 0.329 nats (`phase1_songs17`), 0.094 (`phase3_songs17_seed44`), <= 0.031 elsewhere. An in-process fp16 pass reproduces the log to < 1e-6, so it is a precision effect, not a pooling bug. |
| exp02 | Which loss term produces the large gradients? | gradient norms of each term, back-propagated separately | STFT term norm = 1.5%-6.5% of token term norm; waveform-L1 < 0.08; token-norm grows during training (about 3-9 early, 15-380 late), so the ">100" "spike" counter is a drift past a threshold. |
| exp03 | Does the STFT/decoded loss help held-out quality? | paired ON vs OFF twins, difference in units of seed std | No detectable effect in this setting (all metrics < 0.7 seed-std at step 300k; no point > 2 std). |

## File inventory (`results/`)

| File | Content |
|---|---|
| `FINDINGS.md` | Generated cross-run tables (H1-H4, fp16, exp02, exp03). Covers **9** runs (the 8 clean ones plus the STFT-off twin); `README.md`'s first section still says 8. |
| `run_metadata.json` | Per run: checkpoint, step, architecture, `eps_bug_clean` (true for all), `best_val_loss`, torch version. |
| `h1_mode_averaging.json` | 144 records (9 runs x 2 bands x 8 codebooks; `codebook` is **0-based** here and in figures, 1-based in `FINDINGS.md`). Keys: `n_cells` (examples x gap positions; 80 in the first record, check each row), `pred_entropy_mean/std/p50` (nats), `train_marginal_entropy`, `holdout_target_entropy`, `uniform_entropy` (6.93), `entropy_ratio_to_marginal`, `kl_to_train_marginal_mean/p50` (nats), `nll_mean`, `marginal_predictor_nll`, `beats_marginal_predictor`, `mode_token`, `mode_token_train_freq`, `p_argmax_eq_mode`, `p_target_eq_mode`, `mode_fallback_lift`, `distinct_pred_tokens`, `distinct_target_tokens`, `top1_prob_mean`, `acc_top1`. |
| `h2_codebook_position.json`, `h2_cells.csv` | `aggregation` (`per_mask_len` 1,440 rows or `merged_over_mask_len` 576 rows), `band`, `mask_len`, `codebook` (0-7), `gap_pos` (frame within the gap), `n` (8 for per-mask-len cells), `acc_top1` with Wilson 95% bounds, `acc_top5`, `nll_mean`, `ppl`. |
| `h2_consistency_check.json` | Per run: each tag's `h2_reduction` (fp32, batch 1) vs `in_process_reference` (fp16, batch 8) vs `historical_metrics_csv`; deltas and maxima. |
| `h3_cross_attention.json`, `h3_attention_by_distance.csv` | Examples (2 per band, 4-frame gap, window start, left/right context length) and mean attention weight per context column by signed distance bin `[1,2) ... [257,1e6)`, per run, band, layer 0-3. |
| `h4_convergence.json`, `h4_curve.csv` | Per run: first/last/min value and step, `tail_slope_nats_per_100k_steps` (OLS over the last 20 validation points), `tail_fit_r2`, `plateaued`, `still_decreasing`, `train_val_gap_at_checkpoint` (only where a `train/avg_nll` point exists, i.e. `phase1_songs17` = 0.372 nats). Curves for `val/combined_loss` and `train/avg_nll`. |
| `summary.csv` | 8,928 tidy rows (experiment, run, checkpoint, step, band, mask_len, method, metric, value, codebook, gap_pos, layer, head, distance_bin, n): H1 metrics, H2 `acc_top1/acc_top5/nll_mean`, H3 `mean_weight_by_distance` and `attn_entropy_ratio`. |
| `exp02/<run>.json`, `exp02/summary.csv`, `exp02/gradient_attribution.{png,pdf}` | Per batch rows (`token_loss`, `dec_stft_loss`, `token_norm`, `dec_l1_norm`, `total_norm`, finite flags) and per checkpoint summary: `n_batches`=100, `weight_decay`, `param_l2` (weight L2 norm), `token_median/p90`, `dec_stft_median/p90/max`, `dec_l1_median`, `total_median`, `stft_over_token_median`, `frac_total_gt_100`, `nonfinite_frac`. 26 checkpoints, 10 runs (includes Phase 0 `baseline`, `bigmodel`). |
| `exp03_stft_ablation/comparison.csv`, `summary.json`, `stft_ablation.{png,pdf}` | Paired ON vs OFF per validation step (every 25k) for `val/*` metrics and the 300 `train/avg_nll` points: `metric,step,on,off,off_minus_on`. `summary.json` gives for `final` and `best_at_cutoff`: `on`, `off`, `off_minus_on`, `seed_std`, `z` = |diff|/std, `verdict` (< 1 std "no detectable effect", 1-2 "inconclusive", > 2 "effect"; rule fixed before seeing the result). |
| `audio_samples/phase4_stft_off_seed43/{best_val,latest}/gap*.wav` | OFF-twin fills (both at step 300,000). Compare with `phase_3/results/audio_samples/phase3_regularization_seed43/`. |
| `figures/` | H1-H4 figures, see below. |

## Reading the figures

**Important: which run is drawn.** `figures/h1_*`, `h2_*`, `h3_*` are overwritten for every run in a call, so they show the **last run processed**, which is `phase4_stft_off_seed43` at step 300,000 (verify in each figure's title).
`README.md`'s "Results" section describes `phase1_songs17`; for that run or any other, use the JSON/CSV tables or `FINDINGS.md`. `h4_convergence` and exp02/exp03 figures cover all runs.

- `h1_mode_averaging_{high,low}_activity`: 2x2. (a) bars = mean predicted entropy per codebook, orange diamonds = train-marginal entropy, green circles = entropy of the holdout targets, dashed = uniform (6.93). Bars at the diamonds mean "as uncertain as no context"; bars at the circles mean "as certain as the data allows". (b) KL(model||train marginal), log scale; the grey band at the bottom marks "indistinguishable from the marginal"; bars far above = not collapsed. (c) fraction of cells where argmax / target equals the globally most frequent token. (d) model NLL (solid) vs context-free marginal-predictor NLL (dashed); the vertical gap is what context buys. Lower is better.
- `h2_heatmap_by_mask_len_{acc_top1,nll_mean}`: grid of rows = codebook 0-7, columns = gap position, one panel per (band, gap length), annotated with cell values; colour scale shared. Blues for accuracy (darker = better); oranges for NLL (darker = worse). Look for a vertical gradient (codebook effect) not for single cells (n = 8). `h2_heatmap_merged_*` merges gap lengths (count-weighted).
- `h3_attention_vs_distance`: mean attention weight vs signed distance from the gap edge (log-symmetric x), one row per decoder layer, columns bands, one line per example; dashed horizontal = uniform (1/context_len). Lines above the dashed line near 0 = local focus. The abrupt drop at the window edges is the end of context (the left/right context lengths differ per example).
- `h3_attention_concentration`: (a) cumulative attention mass vs |distance| vs a uniform reference, (b) attention entropy / ln(context length) (1 = uniform), (c) attention centre of mass in frames.
- `h3_attention_map_{band}_ex{0,1}`: heads x layers heat maps (rows = gap positions, columns = context frames), red dotted line = gap position in the context.
- `h4_convergence`: (a) full run, `val/combined_loss` solid and `train/avg_nll` dashed, log step axis; (b) last 25% with OLS fit; horizontal dashed = each run's minimum. A curve above its dashed minimum at the end = ticking up (overfitting). The train/val gap at the checkpoint exists only for `phase1_songs17` (0.372 nats), because the exported metrics contain a `train/avg_nll` point at the checkpoint step only there.
- `exp02/gradient_attribution`: three panels vs checkpoint step: median token-loss gradient norm (log; dotted line = clip 1.0, so everything above it is clipped), STFT/token norm ratio (log; always 0.015-0.065), model weight L2 norm (regularized runs about 310-620 vs 1,200-1,850 for others). Gradient norms use `train()` mode and fp16 autocast; weight decay differs by run.
- `exp03_stft_ablation/stft_ablation`: ON (blue) vs OFF per metric vs step, with a shaded band of +/- 1 seed std around ON. Lines inside the band = no detectable effect.

## What to trust

- Strong: pipeline determinism (songs17 rows reproduced with max diff 0.0), H1's rejection of strict collapse, H3's rejection of "ignores context", exp02's ratio (agrees with logged training norms: 53.7 vs 52.9 and 220.5 vs 221.9).
- Moderate: H2 (consistent trend, confounded with intrinsic RVQ difficulty), H4 (validation noise: slopes over 20 points).
- Weak: any single H2/H3 cell (n = 8; 2 attention examples per band; ratios dividing by very small far-context weights, e.g. 860x, are unstable). exp03 is one seed per arm at one loss weight (0.1): "no detectable effect" is not "no effect"; the noise reference mixes a 600k-step seed (42) with 300k-step ones.
- Precision: H1/H2 use fp32 batch 1; logged validation uses fp16 batch 8. Compare like with like. The baseline `latest.pt` has NaN weights; its exp02 late point is `best.pt` (step 78k).
- OFF twin has no exported `metrics.csv` (so no H4, no logged-value fp16 check).
