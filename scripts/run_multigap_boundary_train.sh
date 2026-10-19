#!/usr/bin/env bash
# Legacy (pre-Phase 0): train the encoder-decoder multi-gap model with boundary conditioning (lighter variant).
# Needs data/processed/multigap/, which the SETUP.md steps cannot regenerate (see SETUP.md).
# Override the config with CONFIG_PATH=...; extra args are forwarded to audio-infill-train.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/train/multigap_encoder_decoder_decoded_loss_val_v1_boundary_lighter.yaml}"

uv run --project "$REPO_ROOT" audio-infill-train --config "$CONFIG_PATH" "$@"
