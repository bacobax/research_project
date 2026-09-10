# Project handout — audio inpainting research (context for starting fresh chats)

This file exists so a new chat session can pick up any phase of this project without needing
the full history of how we got here. Paste/point a new session at this file and tell it which
phase you want to work on. **Replaces the previous version** — Phase 1 (data scaling) and Phase 2
(regularization due-diligence) are both done; Phase 3 (seed robustness + a gap-length/activity-band
sweep) is running right now. Read §8 first if your first task is just "check on the training."

## 1. The project, in one paragraph

We're training a transformer to fill short gaps (13-50ms) in EnCodec-tokenized audio. The
original two runs — "baseline" (d_model=256, 4 layers) and "bigmodel" (d_model=384, 6 layers) —
trained on a single 400-second song (Nuvole Bianche) and both severely overfit: train token
accuracy climbed to 86-94% while held-out validation accuracy stayed near 1-10%, validation loss
climbing monotonically. **Phase 0 (done)** verified this was a real generalization gap, not a
measurement bug, and found the audio-domain picture is more nuanced than token accuracy alone
suggests. **Phase 1 (done)** tested whether more training data changes that picture — it does,
partially, and the story turned out more interesting than a clean "more data helps": token loss
and decoded-audio-quality metrics (SI-SDR, spectral convergence) don't always agree, and the
most-data regime (17 songs) shows real, sustained improvement in audio quality even as token
loss plateaus. **Phase 2 (done)** asked the complementary question on the original single song:
does regularization alone — smaller model, more dropout, more weight decay, no data change —
also narrow the gap? It does not: the regularized run overfits almost immediately (best
`val/combined_loss` at step 25,000 of a 600,000-step budget) and gets worse for the rest of
training, with only a small late recovery in the final ~15%. Together, Phase 1 and Phase 2
support this project's strongest claim so far — **more data beats regularization alone** — but
that claim rested on one seed per config. **Phase 3 (running now)** checks whether it holds
across seeds, plus a cheap gap-length/activity-band characterization sweep on the two headline
checkpoints. This is for a university assignment — the goal is a *defensible* result (negative,
positive, or nuanced), not a working model. Rigor (independent pipeline validation, honest
baselines, multiple complementary metrics, root-caused failures) is itself part of the
contribution.

## 2. Key facts about the setup (don't need to rediscover these)

- **Codebase**: `src/audio_infill/train.py` (~3900+ lines now) is the whole
  training/validation/model system — `Trainer` class, `TrainConfig` dataclass, EnCodec wrapper
  (`AudioEncoder`), dataset classes, validation-example builders, loss functions, plotting
  helpers. Entry point: `uv run audio-infill-train --config <yaml>`. `src/audio_infill/config.py`
  has a **duplicate** `TrainConfig` (used by tests/`graph.py`) that must be kept in sync
  field-for-field with `train.py`'s copy — `tests/test_train_config_schema.py` asserts this.
  Package uses `uv`, src-layout.
- **Data**: `data/nuvole_bianche.mp3` is the original (ungapped) source — this is the "pivot"
  song used by every run, single-song or multi-song. The processed/gapped dataset lives in
  `data/processed/nuvole_bianche_short_gaps/` (a `.wav` + a `.json` annotation with exact gap
  positions). 4 gaps were carved out: 15/20/30/50ms. `dataset/` (17 Einaudi tracks, 24kHz mono,
  ~75 min total, gitignored) is the extra-songs pool for multi-song training —
  `dataset/008_*` (a different recording of the *same* Nuvole Bianche composition) is always
  excluded, since training on it would leak into the pivot's held-out validation regions.
- **Multi-song training** (added in Phase 1): `resolve_extra_wav_paths` + `MultiSongMaskedSpanDataset`
  let a run sample training windows from N songs while validation stays fixed on the pivot's
  held-out regions (deterministic, seed-based, unaffected by song count). Config fields:
  `extra_wavs_dir`, `extra_wavs_exclude`, `extra_wavs_limit`, `song_sampling`.
- **Automated training-loop safety/stopping** (added in Phase 1): two independent, orthogonal
  guards, both opt-in via config:
  - `nan_guard_enabled` (default `True`) — checks `torch.isfinite(total_loss)` every step,
    before the optimizer update; stops immediately and saves `latest.pt` on the first non-finite
    loss, so a diverging run never trains on garbage for tens of thousands of steps (which is
    exactly what happened, undetected, to the original baseline run).
  - `early_stopping_enabled` (default `False`) — patience-based: `early_stopping_patience`
    consecutive non-improving validation checks stop the run; `early_stopping_min_delta` gives a
    tolerance band so noise-sized fluctuations don't consume patience. **All Phase 1/2 runs have
    this `False`** — deliberately, to see the full training trajectory rather than whatever point
    an automated patience window happens to stop at (see §5 for why this mattered).
- **Live decoded-audio-quality validation metrics** (added in Phase 1,
  `validation_audio_metrics_enabled`): every validation call, in addition to token-level loss,
  computes SI-SDR and spectral convergence on the model's actual decoded predictions for each
  held-out example, in **two variants** — margin-included (`val/si_sdr_db`,
  `val/spectral_convergence`, comparable to Phase 0's exp03/exp04 crop) and gap-only
  (`val/si_sdr_db_gap_only`, `val/spectral_convergence_gap_only`, `margin_frames=0` — isolates
  actual fill quality from the fixed context around it, since with 1-4 frame gaps and a
  16-frame margin, 80-94% of the margin-included crop is unmodified context). **This is a
  reporting-only signal** — it does not drive checkpoint selection or stopping, only
  `val/combined_loss` (token CE) does that (`self.best_val_loss`/`best_val.pt`).
- **Periodic in-training inpaint snapshots** (`test_fill_every`) now save just each real gap's
  context-window crop (`samples/gap{i}_step_{N}.wav`, a few seconds) instead of decoding and
  saving the entire ~400s reconstructed song every time — the old behavior was both slow (a real
  full-track EnCodec decode) and a real disk cost over a long run. The **one** full-song
  reconstruction (`{sample}_infilled.wav` in the dataset dir, and a matching wav under the run's
  `samples/`) still happens exactly once, at the very end of training (`main()`'s own call,
  unaffected).
- **A real, root-caused gradient-explosion incident** (Phase 1, `phase1_songs17`'s first
  no-patience attempt): diverged to NaN at step 131,549 after ~9,400 steps of grad_norm spiking
  into the 10⁴-10⁶ range while `grad_clip` masked the effect on the loss. Root cause: the
  decoded-domain loss's log-magnitude STFT term computes `log(clamp_min(|STFT|, eps))` with
  `eps=1e-7` hardcoded — `d/dx log(x) = 1/x`, so near-silent audio (common in low-activity
  windows) clamped to that floor produces gradients up to ~1e7. Fix: `decoded_loss_log_eps`
  config field (now threaded through `MultiResolutionSTFTLoss`, previously unconfigurable),
  raised to `0.001` for the re-run, plus `grad_clip` tightened 1.5→1.0 as defense-in-depth. The
  re-run (with the fix) ran the *entire* remaining 870k steps clean, including straight through
  the exact step range where the original diverged — confirmed via TB grad_norm history.
- **Useful shell tool**: `gpu-status` (on PATH) — shows GPU memory/util and *your own* processes,
  collapsing shared command-line prefixes into a diff. Always check before launching.
- **`phase_0/` folder**: reusable infrastructure for loading any checkpoint into a live `Trainer`
  *without ever touching the real run directories* (`phase_0/common.build_trainer(run_name,
  checkpoint_tag)`). Verified safe (file mtimes checked byte-identical before/after). Reuse this
  for any future phase that needs to load a checkpoint. Its `RUNS` dict now also has
  `phase1_songs17`/`phase2_regularization` entries (added for Phase 3's gap-length sweep) besides
  the original `baseline`/`bigmodel` ones — Phase 1/2/3 runs live one directory level shallower
  (`outputs/runs/<group>/<run_name>/checkpoints/`) than Phase 0's; `build_trainer` itself is
  layout-agnostic, so add further entries there the same way for any future phase.
- **Disk is tight**: system volume is at ~16GB free (99-100% used) as of this writing, shared
  with whatever else is on this machine. Every run's `save_every` is set well below `total_steps`
  now (periodic `step_N.pt` snapshots for resumability), which is the main long-run disk cost —
  check `df -h` before launching another long run, and consider whether old superseded run
  directories (clearly named `*_diverged_*`, `*_aborted_*`) can be pruned first.

## 3. Narrative: how we got from "it's overfitting" to where we are now

1. Baseline + bigmodel (single song) both overfit; baseline silently diverged to NaN at step
   ~96,600 and trained on garbage for ~92k more steps before anyone noticed. Both manually killed
   once the overfitting pattern was confirmed.
2. **Phase 0**: validated the measurement itself (pipeline correct, model ≈ trivial baselines on
   token accuracy and synthetic-mask audio metrics, but beats every baseline on the 4 real carved
   gaps in audio-domain terms — suggestive, n=4, not conclusive). Full results in §4.
3. **Phase 1 built out**: multi-song training support, the two automated guards (NaN + patience),
   live audio-quality validation metrics, and crop-only snapshots (all described in §2). First
   swept 1→4→17 songs *with* patience-based early stopping (fixed 250k-step ceiling). All three
   stopped via patience; useful, but the user pushed back: a low-patience stop can't distinguish
   "genuinely done getting worse" from "caught a temporary dip that would have recovered."
4. **Re-ran Phase 1 without patience** (`early_stopping_enabled: false`), longer budgets
   (1 song→100k, 4 songs→300k, 17 songs→1,000,000 steps), NaN guard still on. Mid-sweep,
   `songs17` diverged (see §2's root-cause bullet); killed, fixed, restarted clean, ran the full
   remaining budget without incident. `songs01`/`songs04`/`songs17` all completed their full,
   un-truncated budgets. Full results in §5.
5. A real methodological detour worth remembering: partway through, real doubt was raised about
   whether the "overfitting" trend across every run (including the very first baseline/bigmodel)
   might partly be validation-frequency noise rather than a real trend. This was checked with
   actual statistics (Mann-Kendall trend test, reversal-vs-continuation rates, cross-referencing
   the much-denser `train/avg_loss` signal) on all five completed single/small runs — see §5's
   "trend-significance check" for the real, mixed answer (some runs: statistically significant
   real trend, no noise; others, especially the no-patience `songs01`/`songs04`: a real, if
   partial, late-training recovery phase that a tight patience window would have missed).
6. **Phase 2 ran to completion**: single-song regularization due-diligence — smaller model,
   higher dropout/weight-decay, 600k steps, no patience. It overfits almost immediately (best
   `val/combined_loss` at step 25,000) and gets worse for the rest of the budget, with a small
   late recovery in the final ~15%. Full results in §6.
7. **Phase 3 launched** (running now, see §7/§8): with Phase 1 (more data helps) and Phase 2
   (regularization alone doesn't) both done, the project's strongest claim rests on one seed
   each. Rather than the much more expensive originally-planned "Phase 3 characterization sweep"
   and "Phase 4 statistical rigor" (3 seeds on every headline config, including a
   1,000,000-step run), Phase 3 does the minimal version of each: 2 extra seeds per config at a
   reduced 300,000-step budget (chosen because the qualitative contrast was already visible by
   then in the original runs), plus a zero-retraining gap-length/activity-band sweep over the two
   existing headline checkpoints.

## 4. Phase 0 — done. (Unchanged since the original handout; historical/closed.)

**Question it answered**: is the eval pipeline itself trustworthy, and does the model actually
beat trivial non-learned alternatives? **Location**: `phase_0/` at repo root,
`phase_0/README.md` explains the scripts, `phase_0/results/FINDINGS.md` has the raw numbers.

Four experiments, all already run:

1. **exp01 — train-region control (decisive).** Control-region accuracy tracked the training
   loop's own reported train accuracy closely while holdout accuracy stayed near-random —
   **the pipeline is measuring a real generalization gap, not a bug.**
2. **exp02 — token-space baselines.** Model ≈ `repeat_left` (5.1-6.1% vs 5.47% token top-1
   accuracy); beat it in only 1 of 4 checkpoints.
3. **exp03 — decoded-audio metrics on synthetic masks.** Model roughly ties simple baselines
   (SI-SDR ~9.9-10.9dB vs ~9.9-10.5dB); spectral convergence slightly *worse* than baselines.
4. **exp04 — the 4 real carved-out gaps, end-to-end.** Model beats every non-ceiling baseline in
   15/16 (gap × checkpoint) comparisons on SI-SDR and spectral convergence, despite near-zero
   token accuracy on these same gaps.

**Synthesis**: token-exact-match says the model learned little beyond "copy a neighboring
frame"; audio-domain metrics on *synthetic* masks mostly agree. But on the actual *real* target
gaps, decoded audio is consistently closer to ground truth than any heuristic — suggestive
(n=4), not conclusive, which became Phase 1's motivation.

## 5. Phase 1 — data scaling. Done. Full results below.

**Question it answered**: does more training data change the single-song overfitting picture,
and does it change differently depending on which metric you trust?

**Configs**: `configs/train/phase1_songs{01,04,17}.yaml` (1/4/17 songs respectively; `songs17`
excludes `dataset/008_*`, see §2). **Runs**: `outputs/runs/phase1_data_scaling/`.

**Final results (no-patience re-run, full budgets, all three completed):**

| regime | steps | best `val/combined_loss` | best `val/si_sdr_db` | best `val/si_sdr_db_gap_only` |
|---|---|---|---|---|
| 1 song (`songs01`) | 100,000 (complete) | 5.14 (step 10k) | — | — |
| 4 songs (`songs04`) | 300,000 (complete) | 5.07 (step 10k) | — | — |
| 17 songs (`songs17`) | 1,000,000 (complete) | 4.87 (step 950k) | **14.03dB** (step 720k) | **+0.58dB** (step 460k) |

(SI-SDR wasn't tracked in the older, less-complete pulls for `songs01`/`songs04` in this
document — re-derive from their TB event files if needed, same query pattern as §8.)
`songs17`'s gap-only SI-SDR — the metric specifically isolating actual fill quality from margin
context — went **positive** repeatedly in the second half of training (best +0.58dB), a real
qualitative milestone: the model's own predictions became indistinguishable from or better than
the reconstruction floor on some held-out windows, something no earlier run/regime came close to.

**Key findings, in priority order:**

1. **Best token-loss improves monotonically with data size** (5.14→5.07→4.87), and the
   degradation curve gets visibly gentler as data grows — the core signal this sweep was
   designed to surface.
2. **Token loss and decoded-audio-quality metrics don't always agree.** Most visible in
   `songs17`: `val/high_acc_top1` trended up across the whole run while `val/combined_loss` was
   flat-to-worsening over the same stretch. This echoes Phase 0's own caution about token
   accuracy being a weak proxy — now showing up *inside* the training dynamics, not just as a
   post-hoc comparison.
3. **Trend-significance check** (real statistics run on all 5 completed single/small-scale
   runs — original baseline/bigmodel + no-patience `songs01`/`songs04`/`songs17` — using
   Mann-Kendall trend tests, up/down reversal rates, and cross-referencing the much-denser
   `train/avg_loss` signal):
   - Original baseline/bigmodel: clean, close-to-zero-reversal, monotonic val-loss climbs while
     train loss falls smoothly — bigmodel's trend is Mann-Kendall **significant (p=0.024)**. No
     support for "just noise" in these two.
   - No-patience `songs01`/`songs04`: **both show a real, sustained (not single-blip) partial
     recovery phase in their final ~30-40%** of training — `songs01` peaks at step 60k then
     improves for 4 straight checkpoints to 100k; `songs04` peaks around step 200k then improves
     for the final 10 checkpoints to 290k. A tight patience window (like the original
     sweep's `patience=6`, no tolerance) would have stopped both right at the peak, missing this.
     **But** full-trajectory Mann-Kendall is still significant for net worsening in both
     (`songs01` p=0.032, `songs04` p=0.013), and neither recovery gets back to the early-training
     best — so the honest read is "sharp-degrade-then-partial-recover," not "no overfitting."
     A plausible, more mundane explanation for the recovery timing: both start well into their
     cosine LR schedule's decay tail (60%/67% through), where shrinking step size commonly
     dampens oscillation — **not independently verified, flagged as the next thing to check**
     before it goes in the writeup as "genuine late recovery."
   - `songs17` (most data): qualitatively *different*, not just gentler — Mann-Kendall gives a
     *negative* S (net improving direction), not quite significant (p=0.073), and 60% of its
     up-moves reverse by the next checkpoint (vs. 20-33% for the other two) — noisy oscillation
     around a mild improving trend, not momentum-driven drift either way.
   - In **every** run tested, `train/avg_loss` was smooth and monotonically improving with zero
     bumps the whole way through, including exactly where validation peaked — ruling out
     "training itself hit a rough patch" as an explanation for validation's shape; the val
     trajectory's noise/shape is a property of the small (64-example) fixed holdout set's
     estimate, not shared optimization noise.
4. **A gradient-explosion incident was hit and fixed mid-sweep** — see §2/§3. Worth a
   methods/limitations footnote: evidence of a debugged, rigorous pipeline, not swept under the
   rug.

## 6. Phase 2 — regularization due-diligence. Done. Full results below.

**Question it answered**: on the *original single song* (no additional data), does cheap
regularization alone — smaller model, more dropout, more weight decay — narrow the train/val
gap? This isolates regularization from the data-scaling effect already measured in Phase 1.

**Config**: `configs/train/phase2_regularization.yaml`. **Run**:
`outputs/runs/phase2_regularization/phase2_regularization/`. Changes vs. the Phase 1 single-song
baseline (`phase1_songs01.yaml`), everything else (LR, warmup, batch size, decoded-loss weights,
activity-sampling settings) held identical for clean attribution:

| field | phase1_songs01 | phase2_regularization |
|---|---|---|
| `d_model` / `n_heads` / `n_layers` | 256 / 4 / 4 | **128** / 4 / 4 |
| `dropout` | 0.1 | **0.3** |
| `weight_decay` | 0.01 | **0.1** |
| `total_steps` | 100,000 | **600,000** |
| `validation_every` | 10,000 | **25,000** (less frequent, per the trend-noise discussion in §5) |
| `save_every` | 100,000 | 150,000 |
| `early_stopping_enabled` | `false` | `false` (same — no patience) |
| `nan_guard_enabled` | `true` | `true` (same) |
| `validation_audio_metrics_enabled` | `true` | `true` (same) |

**Regularization ideas considered but deliberately *not* included** (to keep the three changed
levers cleanly attributable) — worth a follow-up ablation if this run doesn't close the gap:
curriculum learning (`curriculum: true`, already fully built, never tried in any run so far),
a lower learning rate, data augmentation (would need new code — no augmentation hook exists
anywhere in the pipeline currently), increasing `decoded_loss_weight`.

**Result**: best `val/combined_loss` = 4.9936 at step 25,000 of the 600,000-step budget — the
run overfits almost immediately, then gets *worse* for the rest of training (climbing to ~5.4-5.5
by step 350k-600k), with a small late recovery in the final ~15% (3 consecutive improving
checkpoints just before the budget ends, never getting back to the step-25k best). Ran the full
600k steps clean, no NaN guard event. **Regularization alone does not close the train/val gap on
this single song** — a real contrast against Phase 1's data-scaling result, though (like every
result so far) from a single seed; see §7. Full numbers/figures: `phase_2/results/`
(`run_summary.csv`, `best_by_group.csv`, `figures/`).

## 7. Phase 3 — seed robustness + gap-length/activity-band sweep. Done.

**Question it's answering**: Phase 1 (more data helps) and Phase 2 (regularization alone
doesn't) together support this project's strongest claim — more data beats regularization at
closing the train/val gap — but that claim rests on one seed per config. Separately, is there a
standard-in-the-literature degradation curve (metrics vs. gap length, split by activity band) to
report for the two headline checkpoints? The much more expensive originally-planned versions of
both ideas (a full characterization sweep; 3 seeds on every headline config including a
1,000,000-step run) were replaced with right-sized versions of each — see
`phase_3/results/README.md` for the full reasoning.

**Sub-experiment A — seed robustness** (`phase_3/export_seed_check.py`). Configs:
`configs/train/phase3_songs17_seed{43,44}.yaml`, `configs/train/phase3_regularization_seed{43,44}.yaml`
— copies of `phase1_songs17.yaml`/`phase2_regularization.yaml` with only `seed` and
`total_steps: 300000` changed (each its own fully-scheduled 300k-step run, not a truncation).
Runs: `outputs/runs/phase3_seed_check/`. Every one of the 3 seeds per family (including the two
*original* seed-42 runs) is compared by its best `val/combined_loss` at or before step 300,000 —
not each run's own eventual best — so the comparison is fair despite the original runs having
much longer budgets. All 4 new runs completed cleanly (no NaN guard events).

**Result — more nuanced than the single-seed headline comparison suggested**, and depends on
which metric you trust (echoing Phase 1's own token-loss-vs-audio-metric caution):
- **Token-loss contrast is not clearly seed-robust**: at the matched 300k-step budget,
  `songs17_data_scaling` mean `val/combined_loss` = 4.957 ± 0.057 (3 seeds) vs.
  `regularization_ablation` mean = 5.000 ± 0.017 (3 seeds) — the 0.043 gap between family means
  is *smaller* than `songs17`'s own seed-to-seed std. The much larger single-seed headline gap
  (4.87 @ step 950k vs. 4.9936 @ step 25k) came from comparing each run near its own trajectory's
  best point at very different budgets, not a fair like-for-like check.
- **Decoded-audio-quality contrast (gap-only SI-SDR) IS seed-robust**: `songs17_data_scaling`
  mean -4.49 ± 2.96 dB vs. `regularization_ablation` mean -8.41 ± 1.24 dB — a ~3.9dB gap that
  clearly exceeds `regularization`'s own seed variance.
- **Bonus finding**: both new `regularization` seeds independently reproduce Phase 1's
  "late partial recovery" pattern (mid-training peak, then partial recovery toward the budget's
  end) — the original seed-42 run didn't show this within its own first 300k steps, so it's a
  seed-dependent trajectory shape, not a fixed property of the regularized config.
- **Honest takeaway**: "more data beats regularization" holds up as an audio-quality claim, not
  as a token-loss claim, at this budget. Full numbers: `phase_3/results/seed_check/README.md` /
  `family_summary.csv`; trajectory figure: `phase_3/results/seed_check/figures/seed_check_combined_loss.pdf`.

**Sub-experiment B — gap-length x activity-band sweep** (`phase_3/exp01_gap_length_activity_sweep.py`,
done — no training involved). Pure evaluation of `phase1_songs17`/`phase2_regularization`'s
`best_val.pt` checkpoints at `mask_lengths=(1,2,4,8,16)` frames x activity band, reusing
`phase_0/common.py`'s Trainer-reconstruction infrastructure (now extended with `RUNS` entries for
these two runs, an optional `mask_lengths` override on `build_real_validation_examples`, and a
new `build_validation_audio_targets` helper — see §2). **Out-of-distribution caveat**: both
checkpoints were trained with `mask_len_max=4` and `curriculum=false`, so `mask_len` 8/16 is
genuinely out-of-distribution, not a same-distribution grid extension — every output marks this
explicitly. **Result**: gap-only SI-SDR degrades with gap length in both the trained (1-4 frame)
and OOD (8/16 frame) range, for both checkpoints and both activity bands (e.g. `phase1_songs17`
high-activity: +0.38dB at 13ms → -3.49dB at 53ms [end of trained range] → -26.92dB at 213ms
[OOD]); `phase1_songs17` sits above `phase2_regularization` on gap-only SI-SDR at nearly every
gap length and band. Full tables: `phase_3/results/FINDINGS.md`; figure:
`phase_3/results/figures/gap_length_degradation.pdf`.

## 8. How to check on / manage the currently-running training

**Nothing about a training run is tied to any particular Claude Code session.** Every run in
this project is launched via `nohup ... & disown`, fully detached from the shell that launched
it — it keeps running (or finishes, or diverges and self-stops via the NaN guard) regardless of
whether any AI session is watching. What *does* end when a session closes: any `/loop`-based
hourly check-in that session set up (session-local `CronCreate` jobs — check `is there currently
a recurring check-in?` isn't answerable from outside the session that created it; just set up a
new one if you want one, see below).

**To check current status of any run:**
```bash
gpu-status                                    # confirms which GPU(s) are active, whose process
ps aux | grep "audio-infill-train --config"   # which config is running, and its PID
tail -c 2000 outputs/runs/<group>/<run_name>/launch.log | tr '\r' '\n' | tail -5
grep -E "non-finite|Training stopped early|Training complete" outputs/runs/<group>/<run_name>/launch.log
```
The last `grep` is the single most important check: `non-finite` = the NaN guard fired,
`Training stopped early` = either that or (only if `early_stopping_enabled: true`) patience
fired, `Training complete` = ran its full `total_steps` normally.

**To pull TB scalars** (val/combined_loss, val/si_sdr_db, val/si_sdr_db_gap_only,
val/spectral_convergence, etc.) — use `tensorboard.backend.event_processing.event_accumulator`.
**Long runs' event files get large and the reader can exceed a ~100s sync timeout** — for any
run past a few hundred thousand steps, launch the read as a background process
(`nohup uv run python - > /tmp/.../out.log 2>&1 <<'EOF' & ... EOF`) and poll/read the output file
rather than trying to read it synchronously.

**To launch a new run**: `gpu-status` first (never launch onto a GPU something else is using),
then the same `nohup uv run audio-infill-train --config <yaml> > outputs/runs/<group>/<run_name>/launch.log 2>&1 & disown`
pattern every run in this project has used. If queuing two runs sequentially on one GPU, wrap
both in a single backgrounded `bash -c 'cmd1 ; cmd2'` rather than polling for the first to finish.

**To set up an hourly check-in for yourself**: `/loop 1h check on <run(s)>, pull TB scalars
including val/si_sdr_db and val/spectral_convergence (margin-included and gap-only), summarize
the trend, flag if the NaN guard fired, note anything worth a manual kill/extend call` — this
project has used exactly this pattern (a local, session-scoped `CronCreate` job, not a cloud
schedule, since the check needs local filesystem/GPU access) throughout Phase 1/2. Update or
re-create it whenever the set of active runs changes (e.g. a run finishes and frees a GPU) so it
doesn't keep referencing stale run names.

**Don't do this**: don't `kill` a healthy run without a real reason — the whole point of removing
patience for Phase 1/2 is to see the *entire* trajectory, not stop early on a hunch. Do intervene
(and document why) on a confirmed NaN/diverged run, or on explicit instruction.

## 9. Planned next phases (not started)

- **(Superseded)** An earlier version of this handout planned a "Phase 3 — characterization"
  (full gap-length sweep) and "Phase 4 — statistical rigor" (3 seeds on every headline config).
  Both were replaced by the actual, cheaper Phase 3 in §7 — a gap-length sweep restricted to the
  two headline checkpoints (no new training), and a 2-extra-seed check at a reduced 300k-step
  budget rather than 3 full seeds on every config.
- **The LR-anneal-vs-genuine-recovery question flagged in §5** — pull `train/lr` alongside
  `val/combined_loss` for `songs01`/`songs04` and check whether the late-training recovery phase
  timing lines up with the cosine schedule's decay tail. Cheap (no new training), would sharpen
  how the "partial recovery" finding gets framed in the writeup.
- **Curriculum learning / lower-LR / data-augmentation ablations** flagged in §6 as deliberately
  excluded from Phase 2, worth their own isolated runs if Phase 2's combined changes don't close
  the gap and finer attribution is wanted.
- **Also worth citing, not re-doing**: `notebooks/same_song_retrieval_baselines.ipynb` has
  independent waveform-domain retrieval baselines with its own `evaluate_fill_metrics()` — cite
  separately in the writeup rather than re-deriving.

## 10. Practical notes for whoever runs the next phase

- Two RTX 3090s (24GB each); check `gpu-status` before launching anything.
- `uv run audio-infill-train --config <yaml>` to launch; `uv run pytest -q` for the test suite
  (expect the same 8 pre-existing failures, unrelated to this project's work — confirmed
  identical on a clean checkout since before Phase 0). Two more test files
  (`tests/test_phase1_export_results.py`, `tests/test_phase_scaffold.py`) may show collection
  errors (`ModuleNotFoundError`) if run without `--ignore` — these belong to a separate,
  in-progress `phase_N/results/` export-scaffolding effort (see `scripts/scaffold_phase.py`,
  `README.md`), unrelated to training; exclude them explicitly when running the suite.
- No patience for long due-diligence runs (Phase 1/2/3 all deliberately disable it) — but always
  leave `nan_guard_enabled: true` unless you have a specific reason not to.
- When loading any checkpoint for evaluation/analysis, reuse `phase_0/common.build_trainer(...)`
  — verified safe, never touches real run directories. `phase_0/common.py`'s `RUNS` dict already
  has entries for `baseline`/`bigmodel`/`phase1_songs17`/`phase2_regularization` (see §2); add
  further entries there the same way for any future phase's checkpoints.
- Git attribution for commits: end messages with
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (see `git log` for the exact
  convention in use).
- **Disk is tight** (§2) — check `df -h` before any new long run.
