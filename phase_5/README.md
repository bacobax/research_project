# Phase 5 — is training consistent with how the model is used at inference? (supervisor Q3)

Phase 5 turns Q3 into a list of concrete aspects and measures each one on the four single-song
checkpoints (`baseline` = Phase 0 original run, `phase1_songs01`, `phase2_regularization`,
`phase3_regularization_seed43`; all `best_val.pt`). No training happens here. Procedures are in
`METHODS.md`; every number below is in `results/` and every claim has a figure.

## Answer to Q3 in one paragraph

Training and inference agree on the loss scope, the gap lengths, the gap position, the decoding
(argmax per codebook, a single pass) and, for three of the four checkpoints, the numeric precision.
They disagree in **how the context is produced at a real gap**. In training, the gap is a mask laid
over tokens of clean audio. At a real gap, the gap's samples are zeroed and the gapped file is
encoded. EnCodec's 24 kHz encoder is causal, so the silence changes the tokens **after** the gap
for a long time (86% of tokens at the next frame, about 30% still 0.85 s later). On top of that,
the gap-to-frame rounding leaves 11–33% of each real gap's silent samples outside the mask. The
model's **predictions are not affected** (token NLL and accuracy unchanged), but the corrupted
tokens after the gap are decoded unchanged and cost **2–4 dB of SI-SDR around the gap**. That loss
disappears when the same predictions are written into clean context. So the inconsistency sits in
the inference pipeline (what is re-encoded and output), not in the model. Two smaller mismatches:
checkpoints are selected by token NLL, which tracks audio quality poorly, and one checkpoint's
argmax tokens are sensitive to fp16.

## Q3 aspect table

| # | Aspect | Training | Inference | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | What is scored | gap frames only | gap frames only | consistent | code audit (`docs/RERUN_PLAN.md` §6) |
| 2 | Gap length | 1–4 frames, uniform | real gaps round to 3, 1, 2, 1 frames | consistent (in range) | exp01 coverage table |
| 3 | **Context tokens next to the gap** | clean audio, gap masked | gapped audio encoded, gap masked | **inconsistent** | exp01: 86% of tokens changed at +1 frame, 46% at +8, ~30% at +64 (0.85 s); 0% before an aligned gap (causal encoder); 1.8% of tokens differ even >64 frames from any real gap |
| 3a | effect on predictions | | | **none detectable** | token NLL C − A between −0.027 and +0.022 nats, accuracy ±0.005, every CI includes 0, all 4 checkpoints |
| 3b | effect on output audio | | | **−1.7 to −2.4 dB** SI-SDR (held-out), **−2.8 to −4.2 dB** (real gaps), log-mel L1 +0.15 to +0.36 | entirely from contaminated context tokens left in the output: writing the same predictions into clean context (E) gives −0.01 to +0.34 dB (held-out) and −0.12 to +0.35 dB (real gaps) |
| 4 | **Mask coverage of real gaps** | gaps are frame-aligned | start/end rounded to frames | **inconsistent** | 20%, 33%, 11% and 27% of the silent samples of gaps 0–3 lie outside the mask |
| 4a | does full coverage help? | | | **no, worse** | a mask covering every silent frame (D) costs −2.8 to −3.4 dB (held-out) and −4.2 to −6.0 dB (real gaps): the model's predictions are worse than the partly silent tokens they replace |
| 5 | Gap position in the window | uniform over the 512-frame window (12.7% within ±32 frames of the centre) | centred in a 493–496-frame window | consistent | inference window vs centred 512 window: NLL −0.007 ± 0.009; left edge +0.024 ± 0.017 (the only measurable position effect) |
| 6 | Inference iterations | one pass | `inpaint_iters` passes | consistent (no-op) | identical tokens after 1 and 10 iterations at every real gap, every checkpoint |
| 7 | Numeric precision | fp16 autocast | fp32 | consistent for 3/4 checkpoints | argmax agreement 99.4–99.9% (held-out) and 100% (real gaps); `phase3_regularization_seed43` (step 300k): 82% / 95%, NLL change only +0.003 (near-tied flat distributions) |
| 8 | Checkpoint selection | min validation token NLL | audio is the deliverable | **weakly aligned** | Spearman(NLL, gap SI-SDR) +0.78 (`songs01`, p = 0.008, i.e. audio improves while NLL worsens), +0.22, −0.53, −0.20, −0.08; selecting on NLL gives 0.7–2.6 dB less than the best validation point (an upper bound: the max of noisy points is optimistic) |
| 9 | Gap content | sampler favours active regions (median activity 0.57) | 4 real gaps | consistent | real gaps at the 12th, 26th, 29th, 65th percentile of training gaps |
| 10 | Decoding | argmax per codebook (validation) | argmax per codebook | consistent | code (Q1 is about the training-time decoded loss) |

## What follows for the report

- **Q3 can be closed:** state the table, then the two real mismatches. The context-encoding
  mismatch is a pipeline issue with a clear fix that needs no retraining: decode the predicted gap
  and splice only the gap's samples into the original waveform, so that no re-encoded context
  reaches the output. Condition E shows this recovers the 2–4 dB.
- The model itself is robust to the corrupted context (3a), consistent with Phase 4's finding
  that its predictions are close to the marginal on high-activity material: it does not exploit
  fine context detail well enough for the corruption to matter.
- **Rounding the gap to frames is not the bottleneck** (4a): covering the whole silence makes
  things worse, because the model's fills are worse than partly correct tokens.
- **Model selection on token NLL is a real but secondary mismatch** (8): worth one sentence and a
  future-work item (select on a decoded-audio metric).

## Figures

| Figure | Shows |
|---|---|
| `results/exp01_context_contamination/figures/contamination_profile` | share of context tokens changed by the zeroing vs distance from the gap, and per codebook |
| `results/exp01_context_contamination/figures/real_gap_mask_coverage` | the 4 real gaps on the frame grid: silent samples vs masked frames vs full coverage |
| `results/exp01_context_contamination/figures/effect_heldout_gaps` | change vs clean context for conditions B, C, E, D, Z on 64 held-out gaps per checkpoint (95% CI) |
| `results/exp01_context_contamination/figures/effect_real_gaps_vs_clean` | the same at the 4 real gaps |
| `results/exp01_context_contamination/figures/real_gaps_quality` | absolute SI-SDR / log-mel per real gap and condition |
| `results/exp02_gap_position/figures/train_vs_inference_gap_position` | distribution of training gap positions vs the inference position |
| `results/exp02_gap_position/figures/quality_vs_gap_position` | NLL / accuracy of the same gaps at 10 positions plus the inference window |
| `results/exp03_consistency_checks/figures/selection_nll_vs_audio` | validation NLL vs gap SI-SDR, with selected vs best-audio points |
| `results/exp03_consistency_checks/figures/gap_activity_train_vs_real` | activity of training gaps vs the 4 real gaps |

Each figure exists as PNG and PDF.

## Files

- `exp01_context_contamination.py` — conditions A/B/C/E/D/Z at real and held-out gaps; contamination profile; mask coverage. `--figures-only` redraws from the CSVs.
- `exp02_gap_position.py` — training gap-position distribution; quality vs position.
- `exp03_consistency_checks.py` — iterations, precision, selection, gap content. `--figures-only` redraws.
- `plot_style.py` — shared figure style (validated palette, markers as a second cue).
- `METHODS.md` — procedures and caveats.

## Running

```bash
uv run python phase_5/exp01_context_contamination.py --device cuda:1   # ~25 min (two full-song encodes per held-out gap, cached)
uv run python phase_5/exp02_gap_position.py --device cuda:1            # ~10 min
uv run python phase_5/exp03_consistency_checks.py --device cuda:1      # ~10 min
```
