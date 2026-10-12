# Phase 4 — causal underfitting diagnostics (H1-H4)

The data-scaling result (Phase 1) is correlational: it shows held-out loss moves with song
count, not *why* the model still underfits at 17 songs (training NLL rises 1.32 -> 4.51 nats
from 1 -> 17 songs, not what a purely data-starved model looks like). `RERUN_PLAN.md` Section 10
lists four mechanism-level hypotheses to check before deciding whether the fix is more data, more
capacity, or a different architecture/objective. This phase runs those four diagnostics (H1-H4,
"Phase A" of that plan) against the `phase1_songs17` checkpoint.

**Only run this against eps-bug-clean checkpoints** (`RERUN_PLAN.md` Section 3) — a checkpoint
trained with a meaningful fraction of steps hijacked by the STFT-loss epsilon bug can't be
trusted for these mechanism-level diagnostics. The script prints a loud warning (not a hard
refusal) if asked to anyway.

## Files

- `exp01_underfitting_diagnostics.py` — pure-evaluation script (no new training), reusing
  `phase_0/common.py`'s Trainer-reconstruction infrastructure, same pattern as
  `phase_3/exp01_gap_length_activity_sweep.py`. Builds one trainer + one held-out example set per
  requested run, then runs all four diagnostics off that shared state:
  - **H1** (mode-averaging): per-codebook predicted-distribution entropy vs. the empirical
    training-token marginal's entropy, KL(model ‖ marginal), how often the model falls back to
    the single most-frequent training token, and whether the model ever does *worse* than a
    context-free marginal predictor.
  - **H2** (RVQ-hierarchy structure): top-1/top-5 accuracy and NLL broken down by
    `(codebook, gap position)` instead of the pooled numbers training normally reports, plus a
    three-way consistency check against the in-process `run_validation_safe` pass and the
    historical `phase_1/results/metrics.csv` value at the checkpoint's step.
  - **H3** (context usage): cross-attention weights (via a kwargs-mutating pre-hook, since
    `nn.TransformerDecoderLayer` hardcodes `need_weights=False`) mapped to signed frame-distance
    from the nearest gap edge, aggregated into distance bins.
  - **H4** (convergence): held-out loss (`val/combined_loss`) and train loss (`train/avg_nll`)
    re-plotted directly from the already-extracted `phase_1/results/metrics.csv` — **never** the
    raw TensorBoard file (confirmed in-session: parsing `phase1_songs17`'s 1.4GB event file timed
    out at 570s with zero output; the scalar history already lives in the CSV with sha256
    provenance in `phase_1/results/provenance.json`).
- `exp02_gradient_spike_sources.py` — pure-evaluation script (no new training): back-propagates
  the token loss, the decoded-domain STFT term and the decoded-domain waveform-L1 term
  *separately* on 100 training batches per checkpoint and records each gradient norm, plus the
  model's weight norm. The STFT-loss evidence. See "Exp02" below and `METHODS.md`.
- `exp03_stft_ablation.py` — paired comparison of the two STFT-ablation twins (post-processing of
  TensorBoard scalars, no GPU); `--self-check` validates it against the exported seed-check numbers.
- `make_summary.py` — regenerates `results/FINDINGS.md` (cross-run tables for H1-H4, the fp16/fp32
  check and exp02) from the result files. No GPU needed.
- `METHODS.md` — how each analysis was done, step by step, with caveats.
- `__init__.py`

## Running

```bash
uv run python phase_4/exp01_underfitting_diagnostics.py                                  # H1-H4 on phase1_songs17
uv run python phase_4/exp01_underfitting_diagnostics.py --diagnostics h4 --runs phase1_songs01 phase1_songs17
uv run python phase_4/exp01_underfitting_diagnostics.py --diagnostics h1 h2               # skip H3's attention hooks
```

H1-H3 build a full `Trainer` (re-encodes all 17 songs through EnCodec, no cache — this dominates
runtime, a few minutes) and need a CUDA device with room for inference (the run's own
`configs/train/phase1_songs17.yaml` pins `cuda:1`). H4 is pure CSV parsing, no GPU, runs in
seconds, and intentionally runs first in `main()` so a missing/renamed metrics file fails fast
before any GPU work starts.

```bash
# all 8 clean runs (the script overwrites results/ on each call, so always pass the full set):
uv run python phase_4/exp01_underfitting_diagnostics.py --device cuda:1 --runs \
  phase1_songs01 phase1_songs17 phase1_songs04 phase2_regularization \
  phase3_regularization_seed43 phase3_regularization_seed44 phase3_songs17_seed43 phase3_songs17_seed44
uv run python phase_4/exp02_gradient_spike_sources.py --device cuda:1   # all runs in CHECKPOINT_PLAN
uv run python phase_4/exp03_stft_ablation.py                            # STFT ablation comparison
uv run python phase_4/make_summary.py                                   # -> results/FINDINGS.md
```

## Key correctness notes (see script docstrings/comments for the full reasoning)

- `Trainer.run_validation()` ends with `self.model.train()` — the script re-asserts `.eval()`
  immediately after the H2 reference pass, before any H1/H2/H3 forward call.
- `EncoderDecoderCodebookInfiller.forward` returns logits that are exactly zero outside the gap
  span; every diagnostic forward call slices to the gap before any analysis touches the logits
  (a forgotten slice would look identical to the "mode-averaging" signal H1 tests for — softmax
  of an all-zero row is exactly uniform).
- H1/H2's forward passes are fp32, no autocast, batch-of-1 (mirrors
  `Trainer._predict_validation_window`); the historical/reference numbers are fp16-autocast,
  batch-of-8. **This gap turned out to be large and systematic, not 1e-2-scale noise as
  initially expected** — see Results below.
- H3's attention-hook forward passes are kept strictly separate from H1/H2's, so H1/H2's reported
  numbers are never affected by the (harmless but non-bitwise-identical) numerics of the
  instrumented, fused-attention-disabled forward path.

## Results (phase1_songs17, best_val.pt, step 950,000)

Full data/figures in `results/` (`h1_mode_averaging.json`, `h2_codebook_position.json` +
`h2_consistency_check.json`, `h3_cross_attention.json`, `h4_convergence.json`, `results/figures/`).
Headline findings from the first run:

1. **H1 — the crude "mode-averaging" hypothesis is not supported.** The model beats a
   context-free marginal-token predictor by 1.5-2.5 nats on every codebook (figure panel (d)),
   and KL(model ‖ train marginal) is far from zero (figure panel (b)) — it is not collapsing onto
   the unconditional token distribution. There is a milder, real effect worth noting: predicted
   entropy per codebook tracks the *shape* of the marginal's entropy curve across codebook index
   (both rise together from codebook 0 to 7) while sitting well above the true holdout-target
   entropy — i.e. the model is more uncertain than it should be, without being purely
   context-free.
2. **H2 — failure is concentrated in later codebooks, consistent with the RVQ-hierarchy
   hypothesis**, most visibly in the high-activity band (accuracy trends down from codebook 0 to
   7); no comparably strong trend across gap position. Every cell has n=8 — treat individual
   cells as noisy, the codebook-index trend across many cells as the real signal.
3. **H3 — the model does use local context, more so in later decoder layers.** All four layers'
   attention-vs-distance curves sit above the uniform reference; attention concentrates further
   toward the gap edge by layer 3 than layer 0, with low-activity examples concentrating faster
   across layers than high-activity ones.
4. **H4 — held-out loss has not cleanly plateaued but is noisy with a weak negative tail slope**;
   critically, `val/combined_loss` actually *ticks back up* after its step-950,000 minimum
   (4.869 -> 4.938 by step 1,000,000) while `train/avg_nll` keeps falling — the 0.372-nat
   train/val gap at the checkpoint is itself direct evidence of genuine (if mild) overfitting
   onset, independent of the visual loss-curve read.
5. **Unplanned finding, worth reporting on its own**: the H2 consistency check (comparing fp32
   batch-of-1 evaluation against the fp16-autocast batch-of-8 evaluation training actually used)
   found deltas up to **0.33 nats** — e.g. `val/high_len_2_loss` is 6.37 nats in fp32 vs. 6.70 in
   fp16. An in-process fp16 reference pass reproduces the historical fp16 log to <1e-6 nats,
   ruling out a batch-size or dataset-construction bug — this is a genuine fp16-vs-fp32
   evaluation-precision effect, on a scale **comparable to the report's headline data-scaling
   effect sizes**. Every number in the report derived from `val/*` metrics (including the
   checkpoint-selection criterion itself) was computed under fp16 autocast; this is worth an
   explicit caveat, and possibly its own follow-up ablation, rather than silently assuming
   evaluation precision doesn't matter.

## Caveats

- n=8 per H2 cell (the run's own `validation_examples_per_length_band`) — individual cells are
  noisy; only trends aggregated across many cells should be treated as signal. Not addressed in
  this pass (see `RERUN_PLAN.md` Section 10 for the "extended example set" follow-up idea).
- H1's `'train'` marginal is computed over all 17 training songs; the held-out examples come only
  from the pivot song's holdout regions — the two are not the same distribution, which the script
  reports (three separate marginals: `train`, `holdout_all`, `holdout_targets`) rather than
  hiding.
- H3 uses 2 examples per band at `mask_len=4` — a qualitative/illustrative sample, not a
  high-n statistical claim.

## Update 2026-10-10: all clean runs + the Phase B reruns

The H1-H4 diagnostics were repeated on 8 checkpoints: the original clean runs (`phase1_songs01`,
`phase1_songs17`, `phase3_songs17_seed43/44`) and the four Phase B reruns (`phase1_songs04`,
`phase2_regularization`, `phase3_regularization_seed43/44`). The `phase1_songs17` numbers
reproduce the earlier (committed) results exactly (max difference 0.0 over 1,424 rows), so the
pipeline is deterministic. All cross-run tables are in `results/FINDINGS.md`; headline points:

- **H1** — in every run and both activity bands the model beats a context-free marginal predictor
  on all 8 codebooks, so the strong "collapse to the marginal" reading is not supported. In the
  high-activity band, though, the margin is small (0.5-1.0 nats) and predicted entropy is about
  the same as the training marginal's (0.89-1.07x); in the low-activity band the margin is
  1.3-2.3 nats and entropy is about half the marginal's. The argmax equals the single most
  frequent training token in at most 1.4% (high) / 11% (low) of cells.
- **H2** — later codebooks are worse in 8/8 high-activity and 7/8 low-activity run/band pairs
  (NLL on codebook 8 minus codebook 1: +0.5 to +1.8 nats in the high band; top-1 on codebook 8 is
  0.000 in 7/8 runs there). Consistent with the independent-heads/RVQ-hierarchy hypothesis, but a
  decline is also expected from RVQ residuals getting harder, so it is not proof (H6 is the test).
- **H3** — cross-attention is concentrated near the gap (5x to 860x more weight on the nearest
  context column than on columns >= 257 frames away) in 30 of 32 (run, layer) pairs; layer 0 is
  close to uniform (1x) in two 17-song runs. Context use is not the failure mode.
- **H4** — no run satisfies the plateau criterion. Held-out loss ends above its minimum in 7 of 8
  runs: `songs01` +1.03 nats (strong overfitting), `songs04` +0.29, `phase2_regularization`
  +0.12, the 17-song runs +0.01 to +0.07.
- **fp16 vs fp32 evaluation** — the large gap seen earlier is not general: 0.33 nats for
  `phase1_songs17`, 0.094 for `phase3_songs17_seed44`, 0.021 or less for the other six.

## Exp02 — gradient attribution (STFT-loss evidence)

Motivation: the Phase B' audit of the reruns' aggregated TensorBoard logs reported 10-21% of steps
with `grad_norm > 100` or non-finite, which looked like the STFT-loss epsilon bug persisting.
Exp02 measures directly which loss term produces the gradient (method: `METHODS.md`; data:
`results/exp02/`; figure: `results/exp02/gradient_attribution.png`; table: end of `results/FINDINGS.md`).
26 checkpoints from 10 runs, including the two Phase 0 single-song runs (`baseline`, `bigmodel`).

**Observations**

1. The STFT term's gradient norm is **1.5%-6.5%** of the token-loss gradient norm at every
   checkpoint measured (median over 100 batches); its largest absolute value is 8.1
   (`phase2_regularization`, step 600k, where the token term is 377). The waveform-L1 term is
   below 0.08 everywhere. The token loss accounts for essentially the whole gradient.
2. The ratio is somewhat higher for the three runs that use the old `decoded_loss_log_eps = 1e-7`
   default (`baseline` 3.8-3.9%, `bigmodel` 4.3-6.5%, `songs01` 2.1-6.0%) than for the eps = 1e-3
   runs (1.5-3.4%), but the effect is small and never makes the STFT term dominant.
3. The token-loss gradient norm grows steadily during training in every run: ~3-9 at the first
   checkpoint, 15-380 at the last (the 17-song seed-42 run, measured only at 950k/1M, is flat at
   ~65). The ">100" counter therefore fires on every late step of a run once the norm has
   crossed 100: the reruns' "spike rate" is a drift past an arbitrary threshold, not a set of
   outlier steps. Non-finite steps (~0.045% of steps, ordinary fp16 loss-scaler skips) are the
   same in clean and rerun logs.
4. The measurement agrees with the logged training-time norms (53.7 measured vs 52.9 logged for
   `phase3_songs17_seed43` @300k; 220.5 vs 221.9 for `phase3_regularization_seed43` @300k).
5. The weight L2 norm falls with training in every run that has an early and a late checkpoint up to
   300k (e.g. 621 -> 314 for `phase2_regularization`; 1485 -> 1273 for `phase3_songs17_seed43`)
   while the gradient norm rises, and the regularized runs (weight decay 0.1) have both the
   smaller weights and the larger final gradients. This is consistent with weight decay shrinking
   scale-invariant weights and inflating gradients, but it is a **hypothesis, not tested**; the
   17-song seed-42 run (950k-1M steps, weights ~1686) does not fit a simple version of it.
6. `phase_0` `baseline`'s `latest.pt` (step 162k) contains non-finite weights (148 tensors): that
   run diverged after its best checkpoint at step 78k, so its late point here is `best.pt`
   (step 78k). `bigmodel`'s `latest.pt` is finite.

**What this does and does not show.** It shows that, at the current weight (0.1), the STFT term
contributes a few percent of the gradient norm and that the earlier "spike" reading was a
threshold artifact. It does **not** show whether the STFT loss helps, hurts or does nothing for
held-out quality: a term that is small in gradient norm can still matter in directions the token
loss ignores, and with clipping always active (norm >= 3 vs. clip 1.0) Adam sees direction only.
Answering that needs the matched training ablation (`decoded_loss_enabled: false`, same seed and
data as `phase3_songs17_seed43`), which has **not** been run. Separately, validation NLL of the
four reruns is unchanged from the original runs (e.g. `phase2_regularization` 4.995 vs 4.9936),
so the eps fix did not move held-out loss.

## Exp03 — STFT-loss training ablation (answers supervisor Q2 for this setting)

**Design.** Matched twins: regularized single song, seed 43, 300,000 steps. The ON run is
`phase3_regularization_seed43` (decoded STFT/waveform loss, weight 0.1); the OFF run is
`phase4_stft_off_seed43` (`configs/train/phase4_stft_off_seed43.yaml`), identical except
`decoded_loss_enabled: false`. Method and the decision rule: `METHODS.md` Section 7.

**Twin check.** Training loss is bit-identical to the ON run before step 7000, when the decoded loss
would start (max difference 5e-5), and the token loss stays within about 1e-4 afterwards, so the
comparison is paired batch-for-batch.

**Result (step 300,000, the best-validation step of both runs).** Differences are OFF minus ON,
in units of the seed-to-seed std of the regularization family (`phase_3/results/seed_check`):

| metric | ON | OFF | OFF - ON | / seed std |
|---|---|---|---|---|
| validation NLL | 5.0247 | 5.0111 | -0.014 | 0.69 |
| high-activity top-1 | 0.0195 | 0.0238 | +0.004 | 0.55 |
| low-activity top-1 | 0.1217 | 0.1172 | -0.005 | 0.31 |
| gap-only SI-SDR (dB) | -7.59 | -7.01 | +0.58 | 0.64 |

Over all 12 validation points (every 25k steps) no metric differs by more than 2 seed-std
(largest: gap-only SI-SDR, 1.40 std at step 75k); NLL is slightly lower with the loss off at 10 of
12 points (mean -0.4 std), gap-only SI-SDR has no consistent sign. Figure:
`results/exp03_stft_ablation/stft_ablation.png`; tables: `results/FINDINGS.md`.

**Conclusion.** **No detectable effect** of the STFT loss in this setting, and no sign that it improves the
decoded-audio metrics. This agrees with exp02 (the term is 1.5-6.5% of the gradient norm). Limits: one
seed per arm and one loss weight (0.1), so effects below the seed spread cannot be excluded ("no
detectable effect" is not "no effect"); the 17-song models were not tested. The natural extension is a
seed-44 twin (~10 h) or a larger weight.

**H1-H3 on the two twins' final checkpoints** (`results/FINDINGS.md`). The structural diagnostics also
agree: predicted entropy / train-marginal entropy 0.90 (OFF) vs 0.89 (ON) in the high band and 0.59 vs
0.59 in the low band; NLL on codebook 1 / 8 in the high band 5.57 / 7.33 vs 5.61 / 7.39; model NLL beats
the marginal predictor on all 8 codebooks in both. Cross-attention concentration (ratio of the weight at
distance 1 to the weight at >=257 frames) matches in layers 0, 1 and 3 (14x/25x/14x vs 13x/26x/15x);
layer 2 differs (50x vs 581x), but that ratio divides by a very small far-context weight and is
noisy with 2 examples per band. Real-gap audio from the OFF twin is in
`results/audio_samples/phase4_stft_off_seed43/`, next to the ON twin's in
`phase_3/results/audio_samples/phase3_regularization_seed43/`. `eps_bug_clean` is true for the OFF twin
(same epsilon/clip as its ON twin); this flag was set in the saved outputs rather than by a rerun, since
the numbers are deterministic. The OFF twin has no exported `metrics.csv`, so H4 and the logged-value
column of the fp16 check are not available for it.

