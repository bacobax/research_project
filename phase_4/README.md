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
