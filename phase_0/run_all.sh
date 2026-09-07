#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python phase_0/exp01_train_region_control.py
uv run python phase_0/exp02_baselines.py
uv run python phase_0/exp03_decoded_metrics.py
uv run python phase_0/exp04_real_gaps.py
uv run python phase_0/make_summary.py
