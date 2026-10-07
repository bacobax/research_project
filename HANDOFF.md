# Session handoff — start here

Read this first in a fresh session. It orients you, then points to the two detailed reference
docs (`RERUN_PLAN.md`, `phase_4/README.md`) for everything else — don't duplicate reading those,
just use them.

## 1. What this project is

Transformer that inpaints short gaps (13-53ms, 1-4 EnCodec frames) in solo-piano music tokens.
University research project; report dated 11 Sept 2026. Central finding so far: the model
underfits even as training data scales up (1 -> 17 songs) — train NLL *rises* from 1.32 to 4.51
nats, which isn't what a purely data-starved model looks like. The open question driving all
current work: **why**, mechanistically — data limitation, architecture, optimization, or
objective mismatch?

## 2. What happened in the last session (chronological)

1. Supervisor asked three pointed questions about the report draft (impact of the STFT loss?
   why argmax vs. weighted-average token decoding? is loss computed gap-only or gap+context?).
   Audited the code directly (`src/audio_infill/train.py`) to answer precisely. Q2 and Q3 came
   back **confirmed correct, not bugs** (RERUN_PLAN.md Section 6).
2. That audit surfaced a real bug while checking Q1: the STFT-loss numerical-stability fix
   (`decoded_loss_log_eps`: 1e-7 -> 1e-3, paired with `grad_clip`: 1.5 -> 1.0) was applied to only
   the 17-song run family, not to the 1-song/4-song/regularization families — an uncontrolled
   confound in the report's main comparisons. Confirmed **empirically** (not just theoretically)
   by pulling `train/grad_norm` histories from the actual TensorBoard logs: the regularization
   family had ~19-23% of training steps with a gradient-norm spike large enough to hijack that
   step's update direction, vs. ~0.045% for the clean (17-song, 1-song) runs. Full writeup:
   `RERUN_PLAN.md` Sections 1-6.
3. Fixed a disk-space problem (host was at 100%, ~15GB free) caused by the training loop writing
   ~18 TensorBoard scalars unconditionally every single step. Reworked `Trainer.train()` in
   `src/audio_infill/train.py` to aggregate and flush at `cfg.log_every` cadence instead
   (`RERUN_PLAN.md` Section 8) — grad_norm specifically keeps max/spike-count aggregates, not
   just an average, so the exact diagnostic signal from step 2 survives at ~100x less log volume.
   Tested (`pytest tests/test_early_stopping.py tests/test_validation_integration.py
   tests/test_decoded_loss.py` — 19 passed, 1 pre-existing unrelated failure).
4. Wrote `RERUN_PLAN.md` as the single source of truth for the bug, which runs need rerunning,
   and a combined execution plan (reproduced in Section 4 below).
5. To get *causal*, not just correlational, evidence for why the model underfits, designed and
   ran four diagnostics (H1-H4) against the `phase1_songs17` checkpoint — this is "Phase A" of
   the plan. New directory `phase_4/` (script `exp01_underfitting_diagnostics.py`, `README.md`,
   `results/`). Full results and how to read every chart: `phase_4/README.md`. One-line summary
   of findings:
   - **H1** (mode-averaging): not supported in the crude form — model beats a context-free
     marginal-token predictor by 1.5-2.5 nats on every codebook.
   - **H2** (RVQ-hierarchy / independent-codebook-heads): accuracy degrades with codebook index
     in the high-activity band (consistent with the hypothesis, not proof of it — see the
     conversation's long discussion of why a decline is also expected "for free" from RVQ
     residuals getting harder regardless of architecture). Much weaker/absent in low-activity.
   - **H3** (context usage): ruled out as a failure mode — decoder attention is measurably more
     concentrated near the gap than uniform, increasingly so in deeper layers.
   - **H4** (convergence): held-out loss ticks back up after its step-950,000 minimum while
     train loss keeps falling — a 0.372-nat train/val gap at the checkpoint, direct evidence of
     mild overfitting onset layered on top of the known underfitting.
   - **Unplanned finding**: a genuine, large (~0.33 nat) fp16-autocast-vs-fp32 evaluation
     precision gap, confirmed real (not a bug) via cross-check. Comparable in scale to the
     report's headline data-scaling effect sizes. Not yet turned into a formal hypothesis or
     fixed — flagged in `RERUN_PLAN.md` Section 10 for follow-up.

## 3. Documents to read, in this order

1. `RERUN_PLAN.md` — the eps/grad_clip bug, which runs are confirmed affected, exact config
   fix, and the combined phase plan (Section 10). This is the main tracking doc.
2. `phase_4/README.md` — what the H1-H4 diagnostics are, how to run them, full Phase A results
   and how to interpret every figure in `phase_4/results/figures/`.
3. (Only if you need the mechanism-level "why" behind H1-H3, e.g. RVQ residual structure, the
   MusicGen delay-pattern analogy for H2/H6, or precisely what each chart axis means — that
   level of detail lives in this session's conversation history, not in a file. If a fresh
   session needs it reconstructed, the short version: RVQ encodes each frame through 8
   codebooks sequentially, codebook k quantizing the residual left by codebooks 1..k-1; this
   model's decoder predicts all 8 independently per frame with no cross-codebook conditioning,
   which is the same structural issue MusicGen's paper calls the "parallel pattern" and found to
   underperform their "delay pattern" fix. H6 below is the proposed fix for this architecture
   specifically.)

## 4. The combined execution plan (current status)

| Phase | What | Depends on | Status |
|---|---|---|---|
| **A** | H1-H4 on `phase1_songs17` and `phase1_songs01` | Nothing | **`phase1_songs17` done.** `phase1_songs01` not run yet — one command, see Section 5 below. |
| **B** | Launch 4 reruns (`phase1_songs04`, `phase2_regularization`, `phase3_regularization_seed43/44`) with fixed `eps`/`grad_clip` | Nothing | **Not started.** Configs not yet edited (verified: all 4 still have `grad_clip: 1.5`, no `decoded_loss_log_eps` override). |
| **B'** | Re-audit grad_norm spike rate on B's new logs, confirm ~0.045% | B finished | Not started. |
| **C** | Re-run H1-H4 on the new `phase2_regularization`/`phase1_songs04` checkpoints | B' confirmed clean | Not started, blocked on B/B'. |
| **D** | H5 (capacity sweep) + H6 (codebook-factorization ablation), new training, configured with fixed eps/clip from the start | Nothing | Not started. Not yet designed in detail — H6 especially is the most direct causal test of the H2 finding above, worth prioritizing. |

A, B, and D have no dependencies on each other — all three can start immediately, in parallel. Only C is gated on B/B' finishing.

## 5. Concrete next steps, ready to execute

**Finish Phase A** (cheap, no new training, just re-run the existing diagnostic script on the other clean checkpoint):
```bash
uv run python phase_4/exp01_underfitting_diagnostics.py --runs phase1_songs01
```

**Start Phase B** (edit exactly 2 fields in each of 4 YAML files, then launch training):
```yaml
decoded_loss_log_eps: 0.001   # add this line
grad_clip: 1.0                 # change from 1.5
```
in `configs/train/phase1_songs04.yaml`, `configs/train/phase2_regularization.yaml`,
`configs/train/phase3_regularization_seed43.yaml`, `configs/train/phase3_regularization_seed44.yaml`.
Change nothing else — seed, steps, architecture all stay identical, this is purely the
numerical-stability fix (RERUN_PLAN.md Section 5). Archive or at least don't overwrite the old
(buggy-eps) TensorBoard logs/checkpoints before relaunching — `RERUN_PLAN.md` Section 9 flags
this as an open decision (keep old results for an explicit before/after footnote vs. just
replacing); not yet resolved, decide before launching.

**After B finishes**, redo the grad_norm spike-rate audit (same method as `RERUN_PLAN.md`
Section 3 — pull `train/grad_norm_spike_count`/`train/grad_norm_max` from the new, now-aggregated
TensorBoard logs per the Section 8 logging change) to confirm the fix actually worked before
trusting anything downstream.

**Phase D / H6** is probably the highest-value next design task: a retrain where codebook k's
head is conditioned on the (teacher-forced) tokens of codebooks 1..k-1 for the same gap position,
vs. the current independent-heads baseline, same data/seed otherwise. This is the actual causal
test of the H2 finding — not yet scoped into a concrete implementation plan.

## 6. Constraints to not forget

- **Disk**: host was at 100% capacity, ~15GB free, at the time of this session. Check
  `df -h /` before any new training run or large artifact write. `phase_4/`'s entire footprint
  is ~4MB by design (JSON/CSV/PNG/PDF only, no checkpoints/logs) — keep any new diagnostic work
  to that same budget.
- **GPU**: `phase1_songs17.yaml` (and likely the other configs) pin `device: cuda:1`. At last
  check, `cuda:1` was nearly empty (good) while `cuda:0` was nearly full with an unrelated other
  user's job — re-check `nvidia-smi` before launching anything, don't assume this still holds.
- **eps-bug-clean set**: only `phase1_songs01`, `phase1_songs17`, `phase3_songs17_seed43`,
  `phase3_songs17_seed44` are currently safe to treat as mechanism-level evidence (H1-H3 style
  diagnostics). Don't run causal diagnostics against `phase1_songs04`/`phase2_regularization`/
  `phase3_regularization_seed43/44` and treat the results as trustworthy until Phase B + B' are
  done — `phase_4/exp01_underfitting_diagnostics.py` already prints a loud warning if asked to
  anyway, don't suppress or ignore it.

## 7. Repo state as of end of last session

Uncommitted, nothing pushed:
- `src/audio_infill/train.py` — modified (Section 8 logging change).
- `RERUN_PLAN.md` — new.
- `phase_4/` — new (script, README, results — all small, see disk note above).
- `HANDOFF.md` (this file) — new.

No config files edited. No training runs (re)launched. Nothing committed this session — if you
want these changes preserved across a `git clean`/branch switch/etc., commit or stash first.
