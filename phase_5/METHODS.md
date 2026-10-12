# Phase 5 — how each analysis was done

Results and the Q3 table are in `README.md`. Statements here are about procedure; anything marked
**Interpretation** was reasoned, not tested.

## 0. Common setup

- **Checkpoints:** `best_val.pt` of the four single-song runs: `baseline` (Phase 0, step 20k),
  `phase1_songs01` (10k), `phase2_regularization` (25k), `phase3_regularization_seed43` (300k).
  Rebuilt with `phase_0/common.build_trainer` (scratch output folder, nothing under `outputs/runs`
  is written), model in `eval()`.
- **Inference call:** fp32, no autocast, batch of 1, gap tokens replaced by the mask token,
  boundary tensors from the mask — the same call `Trainer.inpaint_all_gaps` makes.
- **Inference window:** context budget `max_len − gap_len − 16` split evenly around the gap, as in
  `inpaint_all_gaps` (493–496 frames for 1–4-frame gaps).
- **Held-out gaps:** the run's own validation set (`build_real_validation_examples`, seed
  `cfg.seed + 1009`): 8 examples per (activity band, gap length 1–4) = 64 per checkpoint, all inside
  held-out regions the model never trained on.
- **Real gaps:** the 4 carved gaps of `nuvole_bianche_multigap_4x_15ms_20ms_30ms_50ms` (annotation
  JSON: exact zeroed sample spans). The original ungapped audio (`data/nuvole_bianche.mp3`) is
  loaded with `load_source_wav_matching_annotation` so sample indices match the annotation.
- **Frame geometry:** 24 kHz audio, 75 frames/s, 320 samples per frame.

## 1. Experiment 1 — context contamination (`exp01_context_contamination.py`)

**Encodings.** The clean source is encoded once with the run's `AudioEncoder` (whole song, as in
training). For each gap a copy of the same source with the gap's samples set to zero is encoded,
again as a whole song; encodings are cached across checkpoints. Clean and gapped encodings
therefore differ only because of the zeroed samples. Sanity check: the training pipeline's own
tokens of the gapped file agree with this re-encoding on 99.38% of tokens (the rest comes from
normalising before vs after resampling).

**Zeroing for held-out gaps.** Two variants per gap with rounded mask `[g0, g1)`:
- *aligned*: samples `[g0·320, g1·320)` — exactly the masked frames, so only the encoder's
  context effect remains;
- *real-like*: start and end each moved by a random offset within ±half a frame (deterministic per
  gap), chosen so that the code's rounding (`_resolve_gap_frames`) maps the span back to the same
  `[g0, g1)` — the same geometry the real gaps have.

**Conditions** (same window, same gap; references and targets always come from the clean audio):
- A — clean tokens, rounded mask (training-like);
- B — aligned-zeroing tokens, rounded mask;
- C — real-like (or real) zeroing tokens, rounded mask (what inference does today);
- E — C's predictions written into the clean tokens (separates "worse predictions" from
  "corrupted context left in the output");
- D — as C, but the mask covers every frame the zeroing touches (`floor(start/320)` to
  `ceil(end/320)`), up to 2 frames longer than the rounded mask;
- Z — as C without inpainting (the gapped tokens decoded as they are).

**Metrics.**
- token NLL and top-1 accuracy of the clean target tokens on the rounded-mask frames (A, B, C, D);
- audio: the filled tokens and the clean tokens are decoded over the gap ± 16 frames (the same crop
  the training decoded loss uses); SI-SDR and log-mel L1 on that crop, spectral convergence on the
  crop, SI-SDR and SNR on the frames touched by the gap. Reference = decoded clean tokens, so codec
  error is not counted as inpainting error. The short gap-only region (1–6 frames) makes gap-only
  SI-SDR noisy; the crop metrics are the ones reported.
- statistics: per checkpoint, the paired difference to A (same example) with a 95% normal
  interval (1.96 · sd / √n), n = 64 held-out gaps or 4 real gaps.

**Contamination profile.** For every gap, the tokens of the zeroed encoding are compared with the
clean encoding at every frame within ±64 frames of the rounded mask (mask frames excluded),
per codebook; reported as the share that differ, by signed distance (−1 = frame just before the
mask, +1 = just after). Model-independent, computed once. Also the share of differing tokens more
than 64 frames from any real gap.

**Mask coverage.** For each real gap: silent samples outside the rounded mask, and masked samples
that were not silent.

**Interpretation.** The asymmetric profile (no change before an aligned gap, long decay after) is
what a causal encoder with a recurrent layer produces; EnCodec's 24 kHz model is the causal
streaming variant. This was inferred from the data, not checked in the EnCodec code.

## 2. Experiment 2 — gap position (`exp02_gap_position.py`)

- **Training distribution:** 20,000 draws from the run's own training dataset (`trainer.dataset`,
  re-seeded with `cfg.seed`); the local start of the masked span is recorded.
- **Position sweep:** in each held-out region (one per activity band), 12 gap starts per gap length
  1–4 are spaced evenly over the part of the region where every shifted window fits inside the
  region. Each gap is evaluated with its local start at 8, 32, 64, 128, 192, 248, 320, 384, 448 and
  496 in a 512-frame window, and once in the inference-style window. Token NLL and top-1 accuracy
  only (no decoding). 96 gaps × 11 windows per checkpoint.
- **Statistics:** paired differences to the centred (248) 512-frame window, pooled over
  checkpoints, 95% normal interval.

## 3. Experiment 3 — remaining checks (`exp03_consistency_checks.py`)

- **Iterations:** at each real gap, the `inpaint_all_gaps` loop is replicated (predictions written
  back into the gap between passes) for 10 passes; tokens after pass 1 and pass 10 are compared.
- **Precision:** logits with and without fp16 autocast on the 64 held-out examples and the 4 real
  gaps; share of identical argmax tokens and the NLL difference.
- **Selection:** from each single-song run's TensorBoard log (via `phase_2.export_results.load_scalar_points`),
  all validation points with both `val/combined_loss` and `val/si_sdr_db_gap_only`. Spearman
  correlation, the step chosen by minimum NLL vs the step with the best gap SI-SDR, and the SI-SDR
  difference between them. The Phase 0 `baseline` log predates the audio metrics and is absent.
  The "SI-SDR lost" is an upper bound: the best of several noisy validation points is optimistic.
- **Gap content:** 20,000 training draws record `mask_mean_activity` (the sampler's own activity
  feature); the 4 real gaps' activity is the mean of `activity_per_frame` over their frames;
  reported as percentiles of the training distribution.

## 4. Limits

- Four checkpoints, all single-song; the 17-song models were not tested here.
- Only 4 real gaps: their numbers are illustrative; the held-out gaps (64 per checkpoint) carry
  the statistics.
- Held-out zeroing is synthetic; the real-like variant matches the real gaps' rounding geometry
  but not their exact durations.
- The splice fix suggested in `README.md` is supported by condition E but was not implemented
  or measured on raw waveforms.

## 5. Reproduction

See `README.md` → Running. `--figures-only` (exp01, exp03) redraws the figures from saved outputs.
