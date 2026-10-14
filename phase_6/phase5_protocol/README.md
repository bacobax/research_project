# Phase 5 protocol applied to the Phase 6 model

Question: Phase 6 (`phase6_final_songs17_seed43`) looked slightly worse than the songs17 baseline in `../results/eval/FINDINGS.md`. Do the three Phase 5 experiments (context contamination, gap position, consistency checks) say the same?
The Phase 5 scripts (`phase_5/exp0{1,2,3}_*.py`) run unchanged; `p6_protocol.py` swaps checkpoints, the example set and the `depth_ar` scoring in-process (only change to Phase 5: two figure titles now count examples/checkpoints instead of saying 64 / "4 single-song").
Shared definitions: `phase_5/INTERPRETING_RESULTS.md` (same CSV columns, conditions A/B/C/E/D/Z, figures).

    uv run python phase_6/phase5_protocol/run_exp01.py --device cuda:1   # ~15 min
    uv run python phase_6/phase5_protocol/run_exp02.py --device cuda:0   # ~6 min
    uv run python phase_6/phase5_protocol/run_exp03.py --device cuda:0   # ~12 min
Results: `results/exp01_context_contamination`, `exp02_gap_position`, `exp03_consistency_checks` (CSV/JSON + `figures/*.png|pdf`), run logs `results/exp0N.log`.

## What differs from Phase 5
- Checkpoints (4, same layout as Phase 5): Phase 6 `final` (300k) and `best_val` (30k, the Phase 5 convention), songs17 seed 43 and seed 44 `final` (same data, seed family, budget). Phase 5's single-song checkpoints are not the right comparator for a 17-song model.
- Held-out set: 128 gaps (16 per band x length 1-4, the Phase 6 validation construction), identical for all runs (holdout ranges asserted equal); Phase 5 had 64. Real gaps: the same 4 (n = 4, illustrative).
- `depth_ar`: tokens from the greedy pass, token NLL teacher-forced (exact); identical to Phase 5's definition for independent heads. Exception: exp03 fp16/fp32 NLL uses the greedy-conditioned logits (inference path).
- exp02 uses clean-context windows (as in Phase 5), exp03 training draws pool all 17 songs, and the TensorBoard curves in exp03's selection check come from different validation protocols (Phase 6: zeroed context, fp32, 128 examples; baselines: clean, fp16, 64), so compare Spearman and step choices, not absolute levels.

## Findings (95% CI, paired over 128 held-out gaps)

**Confirmed: Phase 6 is not better, slightly worse.** Condition A (clean context, same examples):

| checkpoint | token NLL | top-1 | crop SI-SDR (dB) |
|---|---|---|---|
| Phase 6 final | 4.961 | 0.087 | 11.19 |
| Phase 6 best_val (30k) | 4.952 | 0.063 | 10.07 |
| songs17 s43 final | 4.846 | 0.115 | 12.18 |
| songs17 s44 final | 4.867 | 0.115 | 12.14 |

These reproduce `eval_final.py` (4.964 / 0.085 for Phase 6 final). Phase 6 is behind by ~0.1 nats, ~3 points of top-1 and ~1 dB, beyond the s43-vs-s44 spread (0.02 nats, 0.00 top-1, 0.04 dB).

**exp01, context contamination.**
- Contamination profile is model independent and unchanged (agreement 0.9938, far mismatch 0.018).
- C minus A, NLL: Phase 6 +0.003 +/- 0.017 (final), +0.000 +/- 0.010 (best_val); baselines +0.024 +/- 0.021 and +0.022 +/- 0.017. Smaller for Phase 6, but intervals overlap, so training on zeroed context has at most a small effect on predictions (it was already near zero).
- C minus A, crop SI-SDR: Phase 6 -2.41 +/- 0.27 (final), -2.31 +/- 0.34 (best_val); baselines -2.66 +/- 0.42, -2.76 +/- 0.35. **Same 2-3 dB audio loss from contaminated context**; zeroed-context training did not remove it.
- E minus A is ~0 for all (-0.13 +/- 0.18 for Phase 6 final): as in Phase 5, the loss is the contaminated context left in the output, which splice removes. D (wider mask) is still worse for all (-3.3 to -3.7 dB).
- Z minus A (unfilled gap): -1.44 dB (Phase 6 final), -0.32 dB (best_val) vs about -2.4 dB for the baselines. This is a weaker fill, not a better floor: Phase 6 `best_val` barely beats leaving the silence in this metric.
- Real gaps (n = 4): same pattern, intervals +/-1-3 dB; nothing separates the runs.

**exp02, gap position.** Still null. edge_8 minus centre NLL: +0.023 (s43), +0.023 (s44), +0.000 (Phase 6 final), -0.005 (best_val), each with +/-0.011-0.015; inference window minus centre +0.000 to +0.004, all intervals around 0. Pooled: edge_8 +0.010 +/- 0.007, inference window +0.002 +/- 0.003. The training sampler is unchanged (12.8% of draws within 32 frames of the centre). Position does not explain Phase 6's deficit.

**exp03, consistency.**
- Iterations: 1 vs 10 passes give identical tokens at all 4 real gaps for all 4 checkpoints (also with depth_ar).
- fp16 vs fp32 argmax agreement, held-out: Phase 6 final 28%, s43 49%, s44 53%, Phase 6 best_val 99.7% (real gaps 54%, 70%, 70%, 100%); mean NLL fp16 minus fp32 +0.131 (Phase 6 final), +0.055, +0.044. 300k checkpoints have flat predictive distributions, so argmax flips with tiny numeric changes, as for `phase3_regularization_seed43` in Phase 5 (82%); Phase 6 is the most affected. Phase 6 validates in fp32, so its logged numbers are unaffected; deploying it in fp16 would not be safe.
- Selection (30 validation points each): Spearman(NLL, gap SI-SDR) -0.62 (p < 0.001) Phase 6, -0.34 (p = 0.07) s43, -0.58 (p < 0.001) s44. Negative = lower NLL goes with better audio, unlike the single-song runs in Phase 5; but the min-NLL point is still early and costly: SI-SDR lost by selecting on NLL 3.9 dB (Phase 6, picks step 30k, best audio at 280k), 4.9 dB (s43, 20k vs 240k), 1.6 dB (s44, 280k vs 260k). Upper bounds (max of noisy points). Real gaps' activity percentiles in training: 9, 22, 25, 65 (Phase 6) vs 9, 22, 25, 65 / 10, 23, 25, 66.

## Reading
Everything agrees with the common-protocol evaluation: the combined Phase 6 recipe gives a slightly weaker model, and none of the Phase 5 mechanisms is the cause or the fix. Contaminated-context loss persists (2-3 dB, absent in E), gap position and iteration count are irrelevant, and NLL-based selection is still unreliable. Not testable here: which of the four bundled changes hurts (no ablations), and whether depth_ar exposure bias (greedy loses ~25% top-1 vs teacher forcing, `../results/eval/FINDINGS.md` P3) explains it.
Caveats: one Phase 6 seed, two baseline seeds (weak noise floor), all held-out gaps from one song, real-gap rows n = 4.
