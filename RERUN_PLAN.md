# Rerun plan: STFT-loss `eps` confound

Status as of this writing: logging fix implemented and tested, **configs not yet edited, no reruns launched**.

## 1. Background

While investigating the supervisor's questions ("what's the impact of the STFT loss?", "argmax vs weighted average?", "is the loss computed on gap-only or gap+context?"), a code audit of `src/audio_infill/train.py` found that the Sec 5 numerical-stability fix (`decoded_loss_log_eps`: `1e-7` -> `1e-3`, `grad_clip`: `1.5` -> `1.0`) was applied to only **one** of the two arms being compared in the report, not both. This is an uncontrolled confound in the data-scaling and regularization comparisons, not just a hypothetical concern — it was confirmed empirically from the actual TensorBoard `train/grad_norm` histories of the completed runs (see Section 3).

Q3 ("gap-only vs gap+context") and the argmax/weighted-average question were both audited and found **not** to be bugs — see Section 6.

## 2. The bug, in one paragraph

`MultiResolutionSTFTLoss` (`train.py`, class around line ~1795) computes `log(magnitude)` on the decoded audio spectrogram, with the magnitude floored at `eps` to avoid `log(0)`. The gradient of `log(x)` at the floor is `1/eps`. With `eps = 1e-7`, that's a slope of `10,000,000`. Solo piano has frequent near-silent spectral bins (sustain, decay, quiet passages), so this isn't a rare edge case — it fires constantly, and the resulting huge gradient routinely propagates back through the whole network, periodically producing `inf`/`nan` gradients or merely-huge-but-finite ones that dominate and misdirect that training step's update (full mechanism recap: `log(x)` formula, straight-through estimator, AMP `GradScaler` skip-on-inf behavior — all previously discussed in conversation, not repeated here).

Fix already known and validated in part of the project: `eps = 1e-3` (slope `1,000`, ~10,000x gentler), paired with `grad_clip = 1.0`.

## 3. Which runs have the bug, confirmed empirically

Checked every config actually used for a completed run (`configs/train/*.yaml`) against its training TensorBoard log (`outputs/runs/.../tb/events.out.tfevents.*`, `train/grad_norm` scalar, pre-clip norm as returned by `clip_grad_norm_`).

| Run | Report section | `decoded_loss_log_eps` | `grad_clip` | steps | % steps with grad_norm > 100 |
|---|---|---|---|---|---|
| `phase1_songs01` (1 song) | Sec 3 — data scaling | `1e-7` (buggy) | 1.5 | 100k | 0.045% — negligible |
| `phase1_songs04` (4 songs) | Sec 3 — data scaling | `1e-7` (buggy) | 1.5 | 300k | **6.07%** |
| `phase1_songs17` (17 songs) | Sec 3 — data scaling | `1e-3` (fixed) | 1.0 | 1M | 0.047% — negligible |
| `phase2_regularization` | Sec 4 — regularization ablation | `1e-7` (buggy) | 1.5 | 600k | **22.8%** |
| `phase3_regularization_seed43` | Sec 6 — seed check | `1e-7` (buggy) | 1.5 | 300k | **18.6%** |
| `phase3_regularization_seed44` | Sec 6 — seed check | `1e-7` (buggy) | 1.5 | 300k | **18.7%** |
| `phase3_songs17_seed43` | Sec 6 — seed check | `1e-3` (fixed) | 1.0 | 300k | 0.045% — negligible |
| `phase3_songs17_seed44` | Sec 6 — seed check | `1e-3` (fixed) | 1.0 | 300k | 0.045% — negligible |

Not yet measured (lower priority, pre-date the multi-song work): `nuvole_bianche_short_gaps_encoder_decoder`, `_bigmodel` (Sec 1 single-song overfitting demo — the claim there is about train-vs-holdout token accuracy, not audio metrics, so likely insensitive to this bug either way).

**Reading the table:** the eps bug alone isn't sufficient to explain the spike rate — `phase1_songs01` has the same buggy `eps` as the regularization family but a negligible spike rate (same order as the "fixed" runs). The regularization config (`d_model` 256->128, dropout 0.10->0.30, weight decay 0.01->0.10) appears to interact with the bug: a smaller, more regularized model produces near-silent predicted spectral content far more often, which hits the `eps` floor far more often. `phase1_songs04` sits in between, plausibly because more songs exposes the model to more varied silence patterns.

## 4. Runs to rerun, and why

| Rerun | Reason |
|---|---|
| `phase1_songs04` | 6.07% of steps had a misdirected/wasted gradient update — moderate confound against `phase1_songs01`/`phase1_songs17`, which are both clean |
| `phase2_regularization` | 22.8% — severe; this run's checkpoints (step 25k, step 550k) are the basis of the Sec 4 "regularization as alternative to data" headline numbers |
| `phase3_regularization_seed43` | 18.6% — same issue, feeds the Sec 6 seed-robustness claim |
| `phase3_regularization_seed44` | 18.7% — same issue, feeds the Sec 6 seed-robustness claim |

**Not rerunning:** `phase1_songs01`, `phase1_songs17`, `phase3_songs17_seed43`, `phase3_songs17_seed44` — all already at the negligible ~0.045% baseline rate, so the fix would not be expected to move their numbers meaningfully. (Could still rerun `phase1_songs01` purely for config-identicalness/rigor if time allows — low expected impact, not prioritized.)

## 5. Exact config changes for each rerun

In each of the 4 YAML files below, change (or add) exactly these two fields — nothing else:

```yaml
decoded_loss_log_eps: 0.001   # was unset -> defaulted to 1e-7
grad_clip: 1.0                # was 1.5
```

Files to edit (not yet done):
- `configs/train/phase1_songs04.yaml`
- `configs/train/phase2_regularization.yaml`
- `configs/train/phase3_regularization_seed43.yaml`
- `configs/train/phase3_regularization_seed44.yaml`

Everything else (seed, architecture, `total_steps`, dataset, optimizer, STFT weights, etc.) stays identical to the original run, so the only intentional difference vs. the original (buggy) version of each run is the numerical-stability fix itself. This mirrors exactly what `phase1_songs17.yaml` / `phase3_songs17_seed43.yaml` / `phase3_songs17_seed44.yaml` already have.

**Do not** also change `decoded_loss_weight` or anything related to the STFT loss's contribution while doing this rerun — that's a separate, deliberate ablation (see Section 7), not this bug fix. Conflating the two would make it impossible to tell which change caused what.

## 6. Audited and confirmed *not* bugs (no action needed)

- **Loss/metric scope (supervisor's Q3):** `_compute_loss` and `_compute_masked_metrics` (train.py ~2486, ~2632) both mask strictly to gap positions via `loss_mask`, which is itself built as `zeros; loss_mask[mask_start:mask_start+mask_len] = True` in every dataset variant. No context leakage into loss or reported accuracy/perplexity.
- **Argmax vs. weighted average (supervisor's Q2):** `_compute_decoded_domain_loss` (train.py ~2502) already uses a straight-through estimator — forward pass uses hard/argmax-decoded audio (realistic), backward pass flows through `logits_to_embeddings`'s softmax-weighted codebook average (differentiable). This is the right design, not a bug.
- **Minor, low-priority, not blocking:** the log message at train.py:2408 still says "hard argmax path," which is stale given the straight-through implementation — cosmetic only. Also, `decoded_loss_temperature` is parsed from config but never actually passed into `logits_to_embeddings` (train.py:2573), so it's silently a no-op — harmless since no run has varied it, but dead config surface.

## 7. Separate, not-yet-started: STFT-loss impact ablation

Distinct from the bug fix above — a deliberate experiment to answer "what does the STFT/decoded loss term actually contribute," to be run **after** the harmonized reruns in Section 5, not combined with them: `decoded_loss_weight = 0` vs. current, same seed/data/everything else, compare token NLL and decoded-audio metrics (SI-SDR, spectral convergence).

## 8. Logging change already implemented (code, not config)

`src/audio_infill/train.py`, `Trainer.train()` — per-step TensorBoard scalar writes (previously ~18 `add_scalar` calls unconditionally every step, the dominant contributor to on-disk log size at these step counts — e.g. 1.4GB for the 1M-step `phase1_songs17` run) are now aggregated and flushed every `cfg.log_every` steps (default 100) instead of every step:

- `grad_norm` -> `train/grad_norm_avg`, `train/grad_norm_max`, `train/grad_norm_spike_count` (count of steps with `grad_norm > GRAD_NORM_SPIKE_THRESHOLD = 100.0` or non-finite), `train/grad_norm_nonfinite` (flag). Kept as max/count rather than a plain average specifically so rare single-step spikes (the subject of this whole investigation) remain visible after aggregation, rather than being averaged away.
- `decoded_loss*` (6 tags) -> `avg_decoded_loss*` at the same cadence.
- Activity-sampling (`sample_*`) metrics moved into the same periodic block.
- `train/loss`, `train/token_loss`, `train/acc_top1`, `train/acc_top5`, `train/ppl`, `train/nll` per-step tags removed outright (redundant with the pre-existing `train/avg_*` tags at the same cadence).

Expected effect: roughly 100x smaller TensorBoard logs at default `log_every`, no loss of diagnostic resolution for the thing that matters here.

Tested: `pytest tests/test_early_stopping.py tests/test_validation_integration.py tests/test_decoded_loss.py` — 19 passed, 1 pre-existing failure unrelated to this change (missing test fixture `data/processed/multigap/wav_test_multigap_4x_1p0s_2p0s_5p0s_10p0s.wav`, confirmed present on `master` before this session's edits too).

**This logging change applies automatically to the 4 reruns** (it's in `train.py`, not per-config) — no additional action needed beyond the config edits in Section 5.

## 9. Open items / not yet decided

- Whether to archive the old (buggy-eps) TensorBoard logs/results for an explicit "before/after the fix" footnote in the report, vs. just overwriting. Leaning toward archive-and-footnote for transparency, not yet actioned.
- Whether to also rerun `phase1_songs01` for perfect setting-identicalness (low expected impact, optional).
- The `decoded_loss_temperature` dead-config issue (Section 6) — fix or remove, not yet decided, not blocking.
- Sec 1 baseline runs (`nuvole_bianche_short_gaps_encoder_decoder*`) — grad_norm spike rate not yet measured.

## 10. Causal diagnostics — why it underfits, not just "does more data help"

The data-scaling result (Sec 3) is correlational: it shows the loss moves with song count, not *why* the model still underfits at 17 songs (training NLL rising 1.32 -> 4.51 as data increases is not what a purely data-starved model looks like — a data-starved model should still memorize its own training set). The following hypotheses each target a distinct mechanism, with a concrete diagnostic and plot for each.

| ID | Hypothesis | Diagnostic | Plot | Cost |
|---|---|---|---|---|
| H1 | Non-autoregressive, per-codebook-independent heads mode-average instead of committing to one plausible fill (ties to the argmax/weighted-average question) | Compare predicted-distribution entropy per position/codebook vs. the empirical marginal's entropy; check how often the model just predicts the globally most-frequent token | Bar/violin chart of predicted vs. marginal entropy, per codebook | Free — existing checkpoint |
| H2 | RVQ hierarchy breaks independent per-codebook heads (codebook *k* needs the residual from `1..k-1`) | Break down top-1 accuracy / NLL by (codebook index, position-in-gap) instead of pooled | 8x4 heatmap (codebook x gap position), one per checkpoint | Free — existing checkpoint |
| H3 | Model isn't using context effectively | Pull decoder cross-attention weights for held-out examples | Attention-weight-vs-context-distance plot, high- vs low-activity examples | Free — existing checkpoint |
| H4 | Training hasn't converged — budget, not data, is the bottleneck | Re-plot held-out loss vs. step, zoomed into the tail of the run | Loss-vs-step, log-x, full run | Free — existing TB log |
| H5 | Capacity bottleneck | Train 2-3 model sizes on the same 17-song set, matched compute | "Capacity scaling" curve (loss vs. param count), mirrors the Sec 3 data-scaling plot | New training |
| H6 | Direct test of H1 — within-frame codebook autoregression (condition codebook *k*'s head on ground-truth tokens of codebooks `1..k-1` in the same frame) | One retrain vs. baseline, same data/seed | Before/after bar chart, token NLL + gap-only SI-SDR, same style as Sec 4 | New training |

### Why the eps-bug reruns (Sections 3-5) matter for this list

H1-H4 are read off a trained checkpoint's weights. A checkpoint from a run where ~1 in 5 steps had its gradient direction hijacked by the `eps` bug (`phase2_regularization`, `phase3_regularization_seed43/44`, and to a lesser degree `phase1_songs04`) can't be trusted for these diagnostics — an observed pattern (diffuse attention, mode-averaged predictions, a particular codebook heatmap shape) might be a real architectural/objective property, or might just be an artifact of that noisy trajectory. There's no way to tell them apart without rerunning.

The already-clean checkpoints (`phase1_songs01`, `phase1_songs17`, `phase3_songs17_seed43/44`, ~0.045% spike rate) are **not** affected and can be analyzed immediately, no dependency on Section 5's reruns.

H5/H6 are new training runs — just launch them with the fixed settings (`eps=1e-3`, `grad_clip=1.0`) from the start; no conflict with the rerun plan, just don't forget.

### Combined execution plan

| Phase | What | Depends on | Can start | Status |
|---|---|---|---|---|
| A | Run H1-H4 on `phase1_songs17` and `phase1_songs01` (already clean) | Nothing | Now | **`phase1_songs17` done** — see `phase_4/` (script, README, results). `phase1_songs01` not yet run (one flag away: `--runs phase1_songs01`). |
| B | Launch the 4 reruns from Section 5 (`phase1_songs04`, `phase2_regularization`, `phase3_regularization_seed43/44`) with the fixed `eps`/`grad_clip` | Nothing | Now, in parallel with A | Not started — configs not yet edited. |
| B' | Re-run the grad_norm spike-rate audit (Section 3's method) on the new logs from B, confirm spike rate is back down to ~0.045% | B finished | After B | Not started. |
| C | Re-run H1-H4 on the new `phase2_regularization` / `phase1_songs04` checkpoints, now safely comparable to Phase A's results | B' confirmed clean | After B' | Not started. `phase_4/exp01_underfitting_diagnostics.py --runs phase2_regularization phase1_songs04` once B' confirms those checkpoints are clean. |
| D | Launch H5 (capacity sweep) and H6 (codebook-factorization ablation), configured with the fixed eps/clip from the start | Nothing | Now, or parallel with B, GPU budget permitting | Not started. |

A, B, and D have no dependencies on each other and can run concurrently. Only C is gated, waiting on B and B'.

### Phase A results (`phase1_songs17`, `best_val.pt`, step 950,000) — summary

Full writeup in `phase_4/README.md`; data/figures in `phase_4/results/`. Headline findings:

- **H1**: the crude "pure mode-averaging" hypothesis is not supported — the model beats a context-free marginal-token predictor by 1.5-2.5 nats on every codebook. A milder effect is real: predicted entropy per codebook tracks the *shape* of the training marginal's entropy curve while sitting above the true holdout-target entropy.
- **H2**: accuracy/NLL degrade with codebook index (most visible in the high-activity band), consistent with the RVQ-hierarchy hypothesis; no comparable trend across gap position. n=8 per cell — only the aggregate trend, not individual cells, should be read as signal.
- **H3**: the model does use local context, increasingly so in later decoder layers; all layers' attention-vs-distance curves sit above the uniform reference.
- **H4**: `val/combined_loss` ticks back up after its step-950,000 minimum (4.869 -> 4.938 by step 1,000,000) while `train/avg_nll` keeps falling — the 0.372-nat train/val gap at the checkpoint is direct evidence of mild overfitting onset, on top of the already-known data-limited underfitting story.
- **Unplanned finding**: the H2 consistency check (fp32 batch-of-1 vs. the fp16-autocast batch-of-8 evaluation training actually used) found deltas up to **0.33 nats** on some groups (e.g. `val/high_len_2_loss`: 6.37 nats fp32 vs. 6.70 fp16) — confirmed genuine (an in-process fp16 reference reproduces the historical fp16 log to <1e-6 nats, ruling out a bug in the fp32 reduction), not previously known, and on a scale **comparable to the report's headline data-scaling effect sizes**. Every `val/*`-derived number in the report, including the checkpoint-selection criterion itself, was computed under fp16 autocast. Worth an explicit caveat in the report and possibly its own follow-up (a controlled fp16-vs-fp32 evaluation ablation) — not yet added to this plan as a formal hypothesis, flagging it here for now.

## 12. Repo state

`src/audio_infill/train.py` has uncommitted changes implementing Section 8 (logging). `phase_4/` (script, README, results) is new and uncommitted. Nothing has been committed this session. No config files have been edited yet. No training runs have been (re)launched yet (Phase B/D not started). Phase A done for `phase1_songs17` only.
