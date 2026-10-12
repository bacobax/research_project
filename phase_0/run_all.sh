#!/usr/bin/env bash
# Run every Phase 0 experiment (exp01-exp04) in order, then build the summary.
# Usage: phase_0/run_all.sh   (from anywhere; cds to the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python phase_0/exp01_train_region_control.py
uv run python phase_0/exp02_baselines.py
uv run python phase_0/exp03_decoded_metrics.py
uv run python phase_0/exp04_real_gaps.py
uv run python phase_0/make_summary.py
