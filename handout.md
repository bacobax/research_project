# Project handout — audio inpainting research (context for starting fresh chats)

This file exists so a new chat session can pick up any phase of this project without needing
the full history of how we got here. Paste/point a new session at this file and tell it which
phase you want to work on.

## 1. The project, in one paragraph

We're training a transformer to fill short gaps (13-50ms) in EnCodec-tokenized audio, on a
single 400-second song (Nuvole Bianche). Two training runs — a "baseline" (d_model=256, 4
layers) and a "bigmodel" (d_model=384, 6 layers) variant with a lower LR — both severely
overfit: train token accuracy climbed to 86-94% while held-out validation accuracy stayed near
1-10% from the very first evaluation onward, with validation loss climbing monotonically the
whole time. **Phase 0 (done) verified this conclusion is trustworthy** rather than a measurement
bug, and additionally found that the *audio-domain* picture is more nuanced than the token
accuracy alone suggests. This is for a university assignment — the goal is a *defensible negative
result*, not a working model. A negative result argued with enough rigor (multiple experiments,
honest baselines, real metrics) is a fine outcome.

## 2. Key facts about the setup (don't need to rediscover these)

- **Codebase**: `src/audio_infill/train.py` (~3400 lines) is the whole training/validation/model
  system — `Trainer` class, `TrainConfig` dataclass, EnCodec wrapper (`AudioEncoder`), dataset
  classes, validation-example builders, loss functions, plotting helpers. Entry point:
  `uv run audio-infill-train --config <yaml>`. Package uses `uv`, src-layout, `src/audio_infill/`.
- **Data**: `data/nuvole_bianche.mp3` is the original (ungapped) source. The processed/gapped
  dataset lives in `data/processed/nuvole_bianche_short_gaps/` (a `.wav` + a `.json` annotation
  with exact gap positions in samples/frames). 4 gaps were carved out: 15/20/30/50ms (frame
  counts [4,2,2,1] at 75 EnCodec frames/sec, 24kHz, K=8 codebooks, 1024-token vocab).
  `configs/data/nuvole_bianche_short_gaps.yaml` is the dataset-gen config.
- **Training configs**: `configs/train/nuvole_bianche_short_gaps_encoder_decoder.yaml`
  (baseline) and `..._bigmodel.yaml` (bigger model, lower LR 2e-4 vs 3e-4, more warmup). Both use
  the new encoder-decoder architecture with boundary conditioning and a decoded-domain STFT loss
  (recent features — see commits `5f5b88d`, `d542786`, `b1a1b12`).
- **Runs and checkpoints** (`outputs/runs/nuvole_bianche_short_gaps/<run_name>/checkpoints/`):
  - baseline: `best_val.pt` (step 20,000, early-stopping point) and `best.pt` (step 78,000) are
    clean. `latest.pt`/`step_108000.pt`/`step_162000.pt` are **all-NaN** — the baseline run
    diverged to NaN at step ~96,600 (an fp16/gradient-explosion incident, investigated but not
    fully root-caused) and kept training on garbage for ~92k more steps before we caught it.
  - bigmodel: `best_val.pt` (step 20,000) and `best.pt` (step 108,000) are both clean; this run
    never diverged.
  - Both runs were manually killed once the overfitting pattern was confirmed (see §3).
- **Useful shell tool**: `gpu-status` (installed at `~/.local/bin/gpu-status`, already on PATH)
  — shows GPU memory/util and *your own* processes with paths relative to the launch dir, and
  collapses shared command-line prefixes across processes into a diff. Use it to check GPU
  availability before launching new runs.
- **`phase_0/` folder** (see §4) has reusable infrastructure for loading any checkpoint into a
  live `Trainer` object *without ever touching the real run directories* — this is the safe
  pattern to reuse for any future phase that needs to load a checkpoint (see §4's "safe
  Trainer reconstruction" note; don't reinvent this).

## 3. Narrative: how we got to "it's overfitting, prove it rigorously"

1. Launched the bigmodel run in parallel with the already-running baseline run (different GPUs).
2. Asked "how are the runs going" → discovered via TensorBoard event-file inspection that (a)
   the baseline run had silently gone all-NaN since step ~96,600, and (b) *both* runs showed
   classic overfitting: train accuracy climbing to 86-94%, validation accuracy pinned at 1-10%
   with monotonically rising validation loss, from the very first eval (step 20,000) onward, with
   zero sign of improvement across 6 consecutive eval checkpoints.
3. Killed the baseline run cleanly (SIGTERM, exited on its own).
4. Confirmed the overfitting trend was consistent, worsening, and present in *both* model sizes
   (so not a capacity artifact) → killed the bigmodel run too, per explicit instruction.
5. Discussed next directions for turning this into a defensible negative result for the
   assignment. Key insight: **token top-1 accuracy is weak evidence that "inpainting doesn't
   work"** — many different EnCodec token sequences decode to near-identical audio, so low token
   accuracy doesn't necessarily mean bad audio. Before trusting the negative result, the
   measurement itself needed validating. That became Phase 0.

## 4. Phase 0 — done. Full results below.

**Question it answered**: is the eval pipeline itself trustworthy, and does the model actually
beat trivial non-learned alternatives?

**Location**: `phase_0/` at repo root. `phase_0/README.md` explains the scripts in detail.
`phase_0/common.py` has all the reusable infrastructure — most importantly:

- **Safe Trainer reconstruction pattern** (`common.build_trainer(run_name, checkpoint_tag)`):
  rebuilds a `Trainer` from the run's *real* training YAML config (via `parse_args`, exactly
  like the training entry point does — never from `ckpt["config"]`, which can contain stale
  paths), forces `cfg.output_dir` to `phase_0/results/scratch/` *before* `Trainer.__init__`
  creates any directories, then calls `trainer.load_checkpoint(...)` to load only the weights.
  **Verified safe**: file mtimes under `outputs/runs/**/{checkpoints,tb,samples}` were checked
  byte-identical before and after every Phase 0 script ran. Reuse this pattern for any future
  phase that needs to load a checkpoint — don't touch `outputs/runs/` directly.
- `RecordingWriter` — drop-in replacement for the real `SummaryWriter` so scalar metrics
  (acc_top1, etc.) from `Trainer.run_validation()` can be read back programmatically instead of
  only written to TensorBoard files.
- `build_train_region_control_examples` — the train-region-control builder (exp01, see below).
- `fill_baseline(name, example, rng)` — non-learned token-space fillers: `repeat_left`,
  `repeat_right`, `nearest`, `random_tokens`, `codec_ceiling` (ground truth, unmasked — the
  ceiling, not a filling method).
- `audio_metrics(...)` — SI-SDR, SNR, waveform L1/MSE, log-mel L1, multi-res STFT
  spectral-convergence/log-magnitude, computed on a decoded gap+margin crop.
- `decode_gap_crop_audio(...)` — **use this, not a full-window decode.** Decoding only the
  gap+margin crop (matching `cfg.decoded_loss_margin_frames`, same as the training-time decoded
  loss) is ~10-15x cheaper than decoding the full 512-frame window and slicing samples
  afterward. (We hit this the slow way first — decoding full windows for 64 examples ×7
  methods×4 checkpoints was on track to take 2+ hours; the crop-first version took ~10 minutes.)

**Four experiments, all already run** (`phase_0/results/exp0N_*.json` + `summary.csv` +
`FINDINGS.md` have the full raw numbers):

1. **exp01 — train-region control (the decisive one).** Built validation-style examples from
   regions the model *trained* on (not held out) and ran the identical
   `Trainer.run_validation()` code path.
   **Result: pipeline is correct.** At every checkpoint, control-region accuracy tracked the
   training loop's own reported train accuracy closely (e.g. bigmodel/best: 94.2% reported vs
   97.4%/91.4% high/low on control regions), while holdout accuracy stayed near-random (0.6-1.4%)
   regardless of how high train accuracy climbed. This rules out "validation is just buggy" —
   the generalization gap is real.

2. **exp02 — token-space baselines.** Scored `repeat_left`/`repeat_right`/`nearest`/
   `random_tokens`/`codec_ceiling` against the model on the *exact* examples the real training
   run validated on (rebuilt deterministically from the run's own seed).
   **Result: model ≈ `repeat_left`.** Model: 5.1-6.1% token top-1 accuracy across the 4
   checkpoints; `repeat_left` (just copying the previous frame): 5.47%, flat across checkpoints
   (it doesn't depend on the model). Model beats `repeat_left` in only 1 of 4 checkpoints.
   Sanity checks passed: `random_tokens` ≈ 0%, `codec_ceiling` = 100%.

3. **exp03 — decoded-audio metrics on the same synthetic validation masks.** Same
   examples/methods as exp02, but decode to audio and score SI-SDR / spectral convergence /
   log-mel etc. on the gap+margin crop.
   **Result: model roughly ties the simple baselines here.** SI-SDR ~9.9-10.9dB for the model vs
   ~9.9-10.5dB for `repeat_left`/`repeat_right`/`nearest`; spectral convergence slightly *worse*
   for the model (0.19-0.26 vs ~0.18-0.19 for the baselines). `random_tokens` is clearly worst
   (negative SI-SDR); `codec_ceiling` is ~59.5dB (the achievable ceiling in this token space).

4. **exp04 — the 4 real carved-out gaps, end-to-end.** Ground truth reloaded from
   `data/nuvole_bianche.mp3` (resampled/normalized identically to how the gapped dataset was
   built, so sample indices align with the annotation), encoded fresh, scored against
   encode-then-decode of the *true* audio (not the raw waveform — so codec-reconstruction error
   isn't conflated with infill error).
   **Result: model beats every non-ceiling baseline in 15 of 16 (gap × checkpoint)
   comparisons** on SI-SDR and spectral convergence — despite scoring near-zero token accuracy
   on these same gaps. E.g. baseline/best on the 40ms gap: model SI-SDR=12.74dB vs
   `repeat_left`=6.05dB, `nearest`=3.81dB.

**The honest synthesis** (already written into `FINDINGS.md`, worth restating for the writeup):
token-exact-match says the model learned little beyond "copy a neighboring frame." Audio-domain
metrics on *synthetic* masks mostly agree with that. But on the actual *real* target gaps, the
model's decoded audio is consistently closer to ground truth than any hand-coded heuristic. Since
exp04 is only n=4 gaps, this is suggestive, not conclusive — flag it as a limitation, and note
that Phase 1's data-scaling experiment (more songs) is exactly the kind of follow-up that could
tell us whether this is a real effect or noise.

**Verification already done and passing**: `uv run pytest -q` shows 8 pre-existing failures
(confirmed identical on a clean checkout with a git stash — unrelated to any of this work,
already broken before Phase 0 started, likely from in-progress schema-change work reflected in
the still-uncommitted `src/audio_infill/make_gapped_dataset.py` / `tests/test_config_parsing.py`
/ `tests/test_data_config_parsing.py`). All 143 real checkpoint/TensorBoard files have
byte-identical mtimes before and after every Phase 0 script ran.

## 5. Planned next phases (not started — pick one per fresh chat)

From the "what next" discussion, in priority order:

- **Phase 1 — data scaling.** Same architecture, fixed step budget (60-100k steps, *not* 600k —
  the overfitting signature was fully visible by step 40k in both runs, so there's no reason to
  train nearly this long again), three regimes: 1 song (already have it) → ~10 songs → ~50-100
  songs. Plot val loss / spectral distance against dataset size. This is the single most
  valuable next experiment: turns "it doesn't work" into "it fails at this data scale, with this
  trend as data grows" — a much stronger, more publishable claim either way it lands.
- **Phase 2 — regularization due-diligence ablations (cheap).** Dropout 0.1→0.3, weight decay
  0.01→0.1, a genuinely *smaller* model (try d_model=128, 2 layers — both existing runs were
  arguably too big for a single 400s song), and early stopping (we already know from
  `best_val.pt` that early-stopping-at-step-20000 was the checkpoint selector's own choice in
  both runs). If nothing closes the train/val gap on one song, that's real evidence, not just an
  assumption.
- **Phase 3 — characterization.** Metrics vs. gap length (systematically sweep 1/2/4/8/16-frame
  gaps ≈ 13-213ms, not just the 4 we happened to carve), split by activity band (high vs low).
  Degradation curves like this are standard in the inpainting literature and show *where* it
  breaks, not just *that* it breaks.
- **Phase 4 — statistical rigor.** 3 seeds on whatever the headline configs end up being, report
  mean ± std, so the negative (or positive) result isn't an artifact of seed 42.
- **Also worth doing at some point, lower priority**: actually root-cause the baseline run's NaN
  divergence at step ~96,600 (a GradScaler-related fp16 gradient explosion, never fully
  diagnosed — see `_compute_loss`'s `F.cross_entropy` on an empty mask selection as one
  candidate cause, though this was never conclusively confirmed) so it doesn't recur in longer
  future runs. Not urgent since we're now planning short (60-100k step) runs anyway.
- **Also worth citing, not re-doing**: `notebooks/same_song_retrieval_baselines.ipynb` already
  has waveform-domain retrieval baselines (copy-paste, boundary-aligned crossfade) with its own
  `evaluate_fill_metrics()`. Phase 0 deliberately didn't re-port these (token-space baselines
  were scored on the model's own exact evaluation examples instead, for strict apples-to-apples
  comparison) — cite the notebook's numbers separately in the writeup rather than re-deriving
  them.

## 6. Practical notes for whoever (whichever fresh session) runs the next phase

- Two RTX 3090s (24GB each) are available; check `gpu-status` before launching anything.
- `uv run audio-infill-train --config <yaml>` to launch training;
  `uv run pytest -q` for the test suite (expect the same 8 pre-existing failures, unrelated to
  this work, unless you're specifically fixing that WIP config-schema change).
- Don't train for 600k steps again — the overfitting signature was visible by step ~40k in both
  prior runs. Budget short runs (60-100k steps) so more experiments fit in the same wall-clock.
- When loading any checkpoint for evaluation/analysis, reuse `phase_0/common.build_trainer(...)`
  rather than writing a new loading path — it's already verified safe (never touches the real
  run directories) and handles the config-reconstruction subtleties correctly.
- Git attribution for any commits from here on: end commit messages with
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (see repo's recent commit history for
  the exact convention already in use, e.g. `git log`).
