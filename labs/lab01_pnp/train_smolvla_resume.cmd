#!/bin/bash
# Lab 01 — resume SmolVLA fine-tune from a saved checkpoint (single GPU).
#
# Prereq: full checkpoint under
#   outputs/train/lab01_pnp_smolvla/checkpoints/<FROM>/
#   (pretrained_model/ + training_state/ — both required for --resume).
#
#   source .venv-rocm/bin/activate
#   LAB01_TRAIN_RESUME_FROM=007500 LAB01_TRAIN_RESUME_STEPS=20000 \
#     ./labs/lab01_pnp/train_smolvla_resume.cmd
#   LAB01_TRAIN_RESUME_FROM=050000 LAB01_TRAIN_RESUME_STEPS=100000 \
#     LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_SAVE_FREQ=10000 \
#     ./labs/lab01_pnp/train_smolvla_resume.cmd
#   ./labs/lab01_pnp/train_smolvla_resume.cmd --wait
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

WAIT=false
for arg in "$@"; do
    if [[ "$arg" == "--wait" ]]; then
        WAIT=true
    fi
done

OUTPUT="$LAB01_TRAIN_OUTPUT"
FROM="$LAB01_TRAIN_RESUME_FROM"
CKPT="$OUTPUT/checkpoints/$FROM"
PRE="$CKPT/pretrained_model"
TS="$CKPT/training_state"
CONFIG="$PRE/train_config.json"
STEPS="$LAB01_TRAIN_RESUME_STEPS"
BATCH="$LAB01_TRAIN_BATCH_SIZE"
SAVE_FREQ="$LAB01_TRAIN_SAVE_FREQ"
LOG="${LAB01_TRAIN_RESUME_LOG:-$REPO_ROOT/train_smolvla_resume_${FROM}_to_${STEPS}.log}"

LEROBOT_TRAIN="$(dirname "$PYTHON")/lerobot-train"
if [[ ! -x "$LEROBOT_TRAIN" ]]; then
    echo "lerobot-train not found. Run: make rocm-sync" >&2
    exit 1
fi

checkpoint_ready() {
    [[ -f "$CONFIG" ]] || return 1
    [[ -f "$TS/training_step.json" ]] || return 1
    [[ -f "$TS/optimizer_state.safetensors" ]] || return 1
    [[ -f "$PRE/model.safetensors" ]] || return 1
    local model_bytes opt_bytes
    model_bytes=$(stat -c%s "$PRE/model.safetensors")
    opt_bytes=$(stat -c%s "$TS/optimizer_state.safetensors")
    [[ "$model_bytes" -gt 800000000 ]] || return 1
    [[ "$opt_bytes" -gt 350000000 ]] || return 1
}

if [[ "$WAIT" == true ]]; then
    echo "Waiting for checkpoint copy to finish under $CKPT ..."
    for i in $(seq 1 360); do
        if checkpoint_ready; then
            echo "Checkpoint ready."
            break
        fi
        model_mb=$(( $(stat -c%s "$PRE/model.safetensors" 2>/dev/null || echo 0) / 1024 / 1024 ))
        opt_mb=$(( $(stat -c%s "$TS/optimizer_state.safetensors" 2>/dev/null || echo 0) / 1024 / 1024 ))
        echo "  [$i] model=${model_mb}MB opt=${opt_mb}MB train_config=$([ -f "$CONFIG" ] && echo yes || echo no)"
        sleep 30
    done
fi

if ! checkpoint_ready; then
    echo "Checkpoint not ready: $CKPT" >&2
    echo "Need pretrained_model/ (865MB model + train_config.json) and training_state/." >&2
    exit 1
fi

echo "=== Lab 01: SmolVLA resume $FROM → $STEPS ==="
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT"
echo "Batch:    $BATCH   save_freq: $SAVE_FREQ"
echo "Log:      $LOG"
echo ""

mkdir -p "$OUTPUT"
set +e
"$LEROBOT_TRAIN" \
    --config_path="$CONFIG" \
    --resume=true \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --save_freq="$SAVE_FREQ" \
    --output_dir="$OUTPUT" \
    --policy.push_to_hub=false \
    2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
