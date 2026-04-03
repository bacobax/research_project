#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-/projets/Fbassignana/research_project/configs/train/multigap_encoder_decoder_decoded_loss_val_v4_boundary_retrieval_topk1.yaml}"

python -m audio_infill.train --config "$CONFIG_PATH" "$@"
