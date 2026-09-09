# Phase 1 data-scaling results

This directory is a compact, Git-trackable export of the one-, four-, and seventeen-song
training runs. Raw checkpoints, TensorBoard logs, validation bundles, and WAV files remain
under `outputs/runs/phase1_data_scaling/` and are intentionally not copied here.

## Main results

| Songs | Best validation step | Combined NLL | High-activity top-1 | Low-activity top-1 | Stop |
|---:|---:|---:|---:|---:|---:|
| 1 | 10,000 | 5.1398 | 0.0075 | 0.0843 | 100,000 (completed) |
| 4 | 10,000 | 5.0666 | 0.0072 | 0.0918 | 300,000 (completed) |
| 17 | 950,000 | 4.8695 | 0.0612 | 0.1657 | 1,000,000 (completed) |

The selected checkpoint is the step with the minimum recorded `val/combined_loss`; it is not
the final checkpoint. These are descriptive results from one seed per condition and should not
be presented as uncertainty estimates.

## Contents

- `metrics.csv`: exact validation points, 1,000-step training aggregates, and run metadata.
- `run_summary.csv` and `tables/run_summary.tex`: one row per scaling condition.
- `best_by_group.csv` and `tables/best_by_group.tex`: best-step metrics by activity and gap length.
- `figures/`: PNG previews and vector PDFs for the paper. Qualitative figures: `figures/qualitative_high_activity_len_4.pdf`, `figures/qualitative_low_activity_len_4.pdf`.
- `configs/` and `song_manifests/`: exact per-run input snapshots.
- `provenance.json`: source hashes plus hashes and locations of every retained raw artifact.
- `source_state.patch`: tracked runtime-code changes relative to the recorded Git commit.

## Regeneration

Run from the repository root:

```bash
uv run python phase_1/export_results.py \
  --runs phase1_songs01 phase1_songs04 phase1_songs17 \
  --input-root outputs/runs/phase1_data_scaling \
  --output-root phase_1/results
```

The exporter never modifies the raw runs. Large binary artifacts remain excluded by the
repository's existing `outputs/` and `data/` ignore rules.
