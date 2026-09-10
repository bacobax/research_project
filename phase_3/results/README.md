# Phase 3 results

## What Phase 3 is, and isn't

An earlier version of the repo handout sketched two much bigger follow-ups: "Phase 3 —
characterization" (a full gap-length sweep) and "Phase 4 — statistical rigor" (3 seeds on every
headline config, including a 1,000,000-step run). Both were cut as overkill for this project's
scale — re-running every headline config 3x, including `phase1_songs17`'s full 1,000,000-step
budget twice more, would cost more compute than the rest of the project combined. This Phase 3
replaces both with the minimal version of each that still answers a real question:

1. **Seed robustness** (`seed_check/`): does the project's strongest claim so far — that more
   data (Phase 1) closes the train/val gap more than regularization alone (Phase 2) — survive a
   seed check, at a reduced (300,000-step) budget chosen because the qualitative contrast between
   the two configs was already clearly visible by that point in their original runs?
2. **Gap-length x activity-band sweep** (`exp01_gap_length_activity_sweep.json` /
   `FINDINGS.md`): a pure-evaluation degradation curve — standard in the inpainting literature —
   over the two headline checkpoints, at zero additional training cost.

## Sub-experiment A: seed robustness

**Question**: is the songs17-vs-regularization contrast a real effect, or could it be single-seed
noise? **Method**: `configs/train/phase3_songs17_seed{43,44}.yaml` and
`configs/train/phase3_regularization_seed{43,44}.yaml` are copies of their parent configs
(`phase1_songs17.yaml`, `phase2_regularization.yaml`) with only `seed` and `total_steps: 300000`
changed — each is its own fully-scheduled 300k-step run (its own cosine LR decay to zero at
300k), not the parent run truncated early. The comparison is fair because every one of the 3
seeds per family (including the two *original* seed-42 runs) is scored by its best
`val/combined_loss` at or before step 300,000 — not each run's own eventual best.

**Result** (all 4 new runs completed cleanly, no NaN guard events; full numbers in
`seed_check/README.md` and `seed_check/family_summary.csv`): the contrast is **more nuanced than
the single-seed headline comparison suggested**, and depends on which metric you trust —
echoing Phase 1's own caution about token loss vs. decoded-audio-quality metrics not always
agreeing.

- **Token-loss (`val/combined_loss`) contrast is not clearly seed-robust.** At the matched
  300k-step budget: `songs17_data_scaling` mean 4.957 ± 0.057 (3 seeds) vs.
  `regularization_ablation` mean 5.000 ± 0.017 (3 seeds). The gap between the two family means
  (0.043) is *smaller* than `songs17`'s own seed-to-seed standard deviation (0.057) — i.e. a
  single `songs17` seed could easily land on either side of `regularization`'s mean by chance.
  The much larger gap in the original single-seed headline numbers (4.87 @ step 950k vs.
  4.9936 @ step 25k) came from comparing `songs17` near the very end of its 1,000,000-step
  budget against `regularization`'s own early peak — an apples-to-oranges comparison across very
  different points in each run's own trajectory, not a fair like-for-like check.
- **Decoded-audio quality (gap-only SI-SDR) contrast IS seed-robust**, and clearly favors more
  data: `songs17_data_scaling` mean -4.49 ± 2.96 dB vs. `regularization_ablation` mean
  -8.41 ± 1.24 dB (3 seeds each). The ~3.9dB gap between family means exceeds
  `regularization`'s own seed-to-seed std by a wide margin, and is comparable to `songs17`'s own
  (much noisier) std — a real, if noisy, separation.
- **A visible echo of Phase 1's "late partial recovery" finding**: both new `regularization`
  seeds (43, 44) show the same pattern Phase 1 found in `songs01`/`songs04` — a mid-training
  peak (worst point) around step 150k-275k, followed by a partial recovery back toward the
  step-300k cutoff (see `seed_check/figures/seed_check_combined_loss.pdf`). This wasn't visible
  in the original `phase2_regularization` seed-42 run within its own first 300k steps (it kept
  climbing through that window) — another seed-dependent trajectory shape, not a fixed property
  of the regularized configuration.
- **Honest read**: "more data beats regularization" is well-supported as an audio-quality
  claim (seed-robust, consistent with Phase 1's own audio-metric findings) but should *not* be
  presented as a token-loss claim at this budget — the token-loss gap is within seed noise.

## Sub-experiment B: gap-length x activity-band sweep

**IMPORTANT — out-of-distribution caveat, read this before the numbers**: both `phase1_songs17`
and `phase2_regularization` were trained with `mask_len_max=4` and `curriculum=false` — neither
model ever saw a gap longer than 4 frames (~53ms) during training. `mask_len` 8 and 16 (frames)
in this sweep are genuinely out-of-distribution probes, not a same-distribution grid extension.
Every table and figure below marks these rows explicitly; never describe degradation (or lack of
it) at those lengths as "the model generalizes."

**Question**: how does infill quality degrade as gap length grows, within the trained range
(1-4 frames) and beyond it (8, 16 frames — OOD), split by activity band? **Method**: pure
evaluation (no new training) of both headline checkpoints' `best_val.pt`, reusing
`phase_0/common.py`'s Trainer-reconstruction and validation infrastructure with
`mask_lengths=(1,2,4,8,16)`. See `FINDINGS.md` for the full per-run tables and
`figures/gap_length_degradation.pdf` for the plotted curves.

**Result** (both runs, gap-only SI-SDR, the metric most directly sensitive to fill quality):
degradation with gap length is visible in both the trained range and the OOD range for both
checkpoints and both activity bands (e.g. `phase1_songs17` high-activity: +0.38dB at 13ms →
-3.49dB at 53ms [end of trained range] → -26.92dB at 213ms [OOD]). `phase1_songs17` sits above
`phase2_regularization` on gap-only SI-SDR at nearly every gap length and band, consistent with
the seed-check's headline finding that more data produces better decoded-audio quality than
regularization alone — but this comparison is incidental to the sweep's main purpose
(characterizing degradation shape), not a substitute for the seed-check's own comparison.

## Audio samples

`audio_samples/<run_name>/{latest,best_val}/gap{0,1,2,3}_step_<N>.wav` — one small (~6.6s
context-window crop, ~310KB) real-gap inpainting example per gap, per run, at two checkpoints:
`latest` (the run's own final trained checkpoint) and `best_val` (its lowest-`val/combined_loss`
checkpoint). All 4 real carved gaps (15/20/30/50ms) are included, generated via
`Trainer.inpaint_all_gaps` — the same routine used for in-training periodic snapshots — against
the exact checkpoint via `phase_0/common.build_trainer`, so these are real model output, not
cached training-time artifacts (which are rarely saved at exactly the best-val step). See
`scripts/export_gap_audio_samples.py` at the repo root; the same script populates every phase's
`results/audio_samples/` this way.

## Regeneration

```bash
uv run python phase_3/export_seed_check.py
uv run python phase_3/exp01_gap_length_activity_sweep.py
uv run python phase_3/make_summary.py
uv run python scripts/export_gap_audio_samples.py --runs phase3_songs17_seed43 phase3_songs17_seed44 phase3_regularization_seed43 phase3_regularization_seed44
```
