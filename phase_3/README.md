# Phase 3 — seed robustness + gap-length/activity-band sweep

Two right-sized follow-ups to Phase 1 (data scaling) and Phase 2 (regularization ablation),
replacing the original, much more expensive "characterization sweep" / "3 seeds on everything"
ideas sketched in an earlier version of the repo handout (see `results/README.md` for why).

## Files

- `export_seed_check.py` — reads TensorBoard logs for 6 training runs (the two original
  headline runs, `phase1_songs17` and `phase2_regularization`, plus 2 new seeds of each at a
  reduced 300,000-step budget) and reports whether the "more data beats regularization alone"
  contrast holds when compared fairly (same step cutoff for every run). No new training or
  inference happens here — pure TensorBoard-log parsing, same pattern as
  `phase_1/export_results.py` / `phase_2/export_results.py`.
- `exp01_gap_length_activity_sweep.py` — pure-evaluation script (no new training) that runs the
  two headline checkpoints' validation pipeline at synthetic gap lengths (1,2,4,8,16 frames) x
  activity band (high,low), reusing `phase_0/common.py`'s Trainer-reconstruction infrastructure.
  **Reports gap lengths beyond 4 frames as out-of-distribution — see `results/README.md`.**
- `make_summary.py` — reads `results/exp01_gap_length_activity_sweep.json` and writes
  `results/FINDINGS.md`, following `phase_0/make_summary.py`'s JSON-to-Markdown pattern (there's
  no training-run TensorBoard log to export from for this experiment, unlike the seed check).
- `__init__.py`

## Running

```bash
uv run python phase_3/export_seed_check.py
uv run python phase_3/exp01_gap_length_activity_sweep.py && uv run python phase_3/make_summary.py
```

`export_seed_check.py` requires all 4 new training runs
(`configs/train/phase3_songs17_seed{43,44}.yaml`, `configs/train/phase3_regularization_seed{43,44}.yaml`)
to have finished (`checkpoints/final.pt` present) — it fails fast otherwise.
`exp01_gap_length_activity_sweep.py` needs no new training; it only reads the already-completed
`phase1_songs17`/`phase2_regularization` `best_val.pt` checkpoints.

## Results

See `results/README.md` for the full writeup, `results/seed_check/README.md` for the seed-check
methodology and numbers, and `results/FINDINGS.md` for the gap-length sweep.
