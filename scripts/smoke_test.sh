#!/usr/bin/env bash
# Fast end-to-end pipeline check using the smallest training config (configs/train/smoke.yaml).
# Override the config with CONFIG_PATH=...; extra args are forwarded to audio-infill-train.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/train/smoke.yaml}"

uv run --project "$REPO_ROOT" audio-infill-train --config "$CONFIG_PATH" "$@"
