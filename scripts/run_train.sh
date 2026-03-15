#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-configs/train/longrun.yaml}"

python -m audio_infill.train --config "$CONFIG_PATH" "$@"
