#!/usr/bin/env bash
# Resume the final training run from its latest checkpoint.
# Override the config with CONFIG_PATH=... and the checkpoint with RESUME_CKPT=...;
# extra args are forwarded to audio-infill-train (a later --resume wins).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/train/phase6_final_songs17_seed43.yaml}"
RESUME_CKPT="${RESUME_CKPT:-outputs/runs/phase6_final/phase6_final_songs17_seed43/checkpoints/latest.pt}"

if [[ ! -f "$RESUME_CKPT" && " $* " != *" --resume "* ]]; then
    echo "resume.sh: checkpoint not found: $RESUME_CKPT (start a run with scripts/run_train.sh, or set RESUME_CKPT=...)" >&2
    exit 1
fi

uv run --project "$REPO_ROOT" audio-infill-train --config "$CONFIG_PATH" --resume "$RESUME_CKPT" "$@"
