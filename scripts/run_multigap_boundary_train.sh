#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-configs/train/multigap_encoder_decoder_decoded_loss_val_v1_boundary_lighter.yaml}"

python -m audio_infill.train --config "$CONFIG_PATH" "$@"
