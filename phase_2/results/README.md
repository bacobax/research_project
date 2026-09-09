# Phase 2 regularization-ablation results

This directory is a compact, Git-trackable export of the single-song regularization ablation
(smaller model, higher dropout, higher weight decay vs. the Phase 1 `phase1_songs01` baseline —
see the repo handout for the exact config diff). Raw checkpoints, TensorBoard logs, validation
bundles, and WAV files remain under `outputs/runs/phase2_regularization/` and are intentionally
not copied here.

## Main results

| $d_{model}$ | Dropout | Weight decay | Best validation step | Combined NLL | SI-SDR gap-only (dB) | High-activity top-1 | Low-activity top-1 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.3 | 0.1 | 25,000 | 4.9936 | -8.98 | 0.0101 | 0.1104 | 600,000 (completed) |

The selected checkpoint is the step with the minimum recorded `val/combined_loss`; it is not
the final checkpoint. `figures/validation_loss.png` shows the full trajectory, including a late
recovery phase in the run's final ~15% of training (val/combined_loss improved for 3 consecutive
checkpoints just before the run's 600k-step budget ended) — worth noting in the writeup even
though the best checkpoint remains an early one. This is a descriptive result from one seed and
should not be presented as an uncertainty estimate.

## Contents

- `metrics.csv`: exact validation points, 1,000-step training aggregates, and run metadata.
- `run_summary.csv` and `tables/run_summary.tex`: one row per regularization condition.
- `best_by_group.csv` and `tables/best_by_group.tex`: best-step metrics by activity and gap length.
- `figures/`: PNG previews and vector PDFs for the paper, including `validation_audio_metrics.*`
  (SI-SDR and spectral convergence, margin-included and gap-only). Qualitative figures:
  `figures/qualitative_high_activity_len_4.pdf`, `figures/qualitative_low_activity_len_4.pdf`.
- `configs/` and `song_manifests/`: exact per-run input snapshots.
- `provenance.json`: source hashes plus hashes and locations of every retained raw artifact.
- `source_state.patch`: tracked runtime-code changes relative to the recorded Git commit.

## Regeneration

Run from the repository root:

```bash
uv run python phase_2/export_results.py \
  --runs phase2_regularization \
  --input-root outputs/runs/phase2_regularization \
  --output-root phase_2/results
```

The exporter never modifies the raw runs. Large binary artifacts remain excluded by the
repository's existing `outputs/` and `data/` ignore rules.
