#!/usr/bin/env bash
# Start the default long training run (configs/train/longrun.yaml).
# Override the config with CONFIG_PATH=...; extra args are forwarded to audio-infill-train.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/train/longrun.yaml}"

uv run --project "$REPO_ROOT" audio-infill-train --config "$CONFIG_PATH" "$@"
