# Phase 3 seed-robustness check

**All values in this table are the best `val/combined_loss` reached at or before step
300,000** -- for the two original seed-42 runs this is *not* their eventual best
(reported elsewhere as 4.87 at step 950,000 for `phase1_songs17`, 4.9936 at step 25,000 for
`phase2_regularization`, the latter already ≤300,000 so unaffected). This table exists
specifically to make a fair, budget-matched 3-seed comparison of the "more data beats
regularization alone" contrast, not to restate the headline single-seed results.

## Family summary (mean ± sample std across 3 seeds)

| family | n seeds | seeds | combined NLL | high-activity acc | SI-SDR gap-only (dB) |
|---|---|---|---|---|---|
| songs17_data_scaling | 3 | 42,43,44 | 4.9569 ± 0.0568 | 0.0269 ± 0.0205 | -4.49 ± 2.96 |
| regularization_ablation | 3 | 42,43,44 | 4.9996 ± 0.0167 | 0.0161 ± 0.0057 | -8.41 ± 1.24 |

## Contents

- `per_run_at_cutoff.csv`: one row per run (6 total) with its own best-at-cutoff step and metrics.
- `family_summary.csv` / `tables/family_summary.tex`: the table above.
- `metrics.csv`: all loaded scalar points for all 6 runs, filtered to step ≤ 300,000.
- `figures/seed_check_combined_loss.{png,pdf}`: val/combined_loss trajectories to the cutoff,
  colored by family, one linestyle per seed, star at each run's best-at-cutoff point.
- `provenance.json`: source hashes and config paths for every run.

## Regeneration

```bash
uv run python phase_3/export_seed_check.py
```
