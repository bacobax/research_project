#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DS_DIR="${DS_DIR:-ds}"
SAMPLE="${SAMPLE:-wav_test_gap_5p000s}"
AUTO_HPARAM="${AUTO_HPARAM:-true}"

D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
N_LAYERS="${N_LAYERS:-8}"
DROPOUT="${DROPOUT:-0.1}"

BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.5}"
WARMUP_STEPS="${WARMUP_STEPS:-7000}"
TOTAL_STEPS="${TOTAL_STEPS:-486000}"
LOG_EVERY="${LOG_EVERY:-6000}"
SAVE_EVERY="${SAVE_EVERY:-54000}"
NUM_WORKERS="${NUM_WORKERS:-2}"

OUTPUT_DIR="${OUTPUT_DIR:-runs/longestrun}"
RUN_NAME="${RUN_NAME:-longrun}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

SEQ_LEN="${SEQ_LEN:-1024}"
MAX_LEN="${MAX_LEN:-1024}"
CTX_LEFT="${CTX_LEFT:-325}"
CTX_RIGHT="${CTX_RIGHT:-324}"
MASK_MIN="${MASK_MIN:-256}"
MASK_MAX="${MASK_MAX:-375}"

INPAINT_ITERS="${INPAINT_ITERS:-10}"
INPAINT_OUTPUT="${INPAINT_OUTPUT:-}"
RESUME="${RESUME:-}"
INPAINT_ONLY="${INPAINT_ONLY:-false}"
TEST_FILL_EVERY="${TEST_FILL_EVERY:-54000}"


EXTRA_ARGS=()

if [[ "$AUTO_HPARAM" == "true" ]]; then
    EXTRA_ARGS+=(--auto-hparam)
fi

if [[ -n "$RESUME" ]]; then
    EXTRA_ARGS+=(--resume "$RESUME")
fi

if [[ "$INPAINT_ONLY" == "true" ]]; then
    EXTRA_ARGS+=(--inpaint-only)
fi

if [[ -n "$INPAINT_OUTPUT" ]]; then
    EXTRA_ARGS+=(--inpaint-output "$INPAINT_OUTPUT")
fi

if [[ -n "$RUN_NAME" ]]; then
    EXTRA_ARGS+=(--run-name "$RUN_NAME")
fi

echo "=== Audio Infiller Training ==="
echo "Sample:       $SAMPLE (from $DS_DIR)"
echo "Auto hparam:  $AUTO_HPARAM"
echo "Model:        d=$D_MODEL h=$N_HEADS L=$N_LAYERS"
echo "Training:     steps=$TOTAL_STEPS bs=$BATCH_SIZE lr=$LR"
echo "Output:       $OUTPUT_DIR"
echo "Device:       $DEVICE"
echo "==============================="

python train.py \
    --ds-dir "$DS_DIR" \
    --sample "$SAMPLE" \
    --seq-len "$SEQ_LEN" \
    --max-len "$MAX_LEN" \
    --ctx-left "$CTX_LEFT" \
    --ctx-right "$CTX_RIGHT" \
    --mask-len-min "$MASK_MIN" \
    --mask-len-max "$MASK_MAX" \
    --d-model "$D_MODEL" \
    --n-heads "$N_HEADS" \
    --n-layers "$N_LAYERS" \
    --dropout "$DROPOUT" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --grad-clip "$GRAD_CLIP" \
    --warmup-steps "$WARMUP_STEPS" \
    --total-steps "$TOTAL_STEPS" \
    --log-every "$LOG_EVERY" \
    --save-every "$SAVE_EVERY" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --inpaint-iters "$INPAINT_ITERS" \
    --test-fill-every "$TEST_FILL_EVERY" \


    
    "${EXTRA_ARGS[@]}"
