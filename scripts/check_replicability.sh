#!/usr/bin/env bash
# Post-setup replicability check (tests/test_replicability.py): does every script run on this checkout?
# Usage: scripts/check_replicability.sh [setup|full] [pytest args...]
#   setup (default): checksums, bit-exact dataset rebuild, every script imports/--help, standalone
#                    scripts run, offline report modes reproduce the tracked results (~5 min).
#   full:            setup + few-step training through each launch wrapper and the Phase 6
#                    scripts on the resulting checkpoints (~15 min on a GPU; ~5 GB temp space).
# Device: AUDIO_INFILL_TEST_DEVICE=cuda:0|cpu (default: GPU with most free memory).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LEVEL="${1:-setup}"
shift || true

AUDIO_INFILL_REPRO="$LEVEL" uv run --project "$REPO_ROOT" pytest -v -rs tests/test_replicability.py "$@"
