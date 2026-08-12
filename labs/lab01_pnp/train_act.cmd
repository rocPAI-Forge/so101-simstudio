#!/bin/bash
# Lab 01 — ACT imitation learning on so101-simstudio-pnp.
#
# Prereq: dataset at ./datasets/so101-simstudio-pnp (validate first).
#         .venv-rocm with lerobot (make rocm-sync).
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/train_act.cmd
#
# Options (pass through to lerobot-train):
#   ./labs/lab01_pnp/train_act.cmd --steps 100000 --batch_size 16
set -euo pipefail
source "$(dirname "$0")/../../scripts/quicktest/_common.sh"

LEROBOT_TRAIN="$(dirname "$PYTHON")/lerobot-train"
if [[ ! -x "$LEROBOT_TRAIN" ]]; then
    echo "lerobot-train not found. Run: make rocm-sync" >&2
    exit 1
fi

OUTPUT_DIR="${LAB01_ACT_OUTPUT:-./outputs/train/lab01_pnp_act}"
STEPS="${LAB01_ACT_STEPS:-10000}"
BATCH_SIZE="${LAB01_ACT_BATCH_SIZE:-8}"
SAVE_FREQ="${LAB01_ACT_SAVE_FREQ:-10000}"

echo "=== Lab 01: ACT training ==="
echo "Dataset:  ./datasets/so101-simstudio-pnp"
echo "Output:   $OUTPUT_DIR"
echo "Steps:    $STEPS  batch_size: $BATCH_SIZE"
echo "Log:      $REPO_ROOT/train_act.log"
echo ""

set +e
"$LEROBOT_TRAIN" \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id=alexhegit/so101-simstudio-pnp \
    --dataset.root=./datasets/so101-simstudio-pnp \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --job_name=lab01_pnp_act \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    "$@" \
    2>&1 | tee train_act.log
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Training exited with error ($status). Last lines of train_act.log:"
    tail -30 train_act.log
    exit "$status"
fi

echo ""
echo "Training complete. Latest checkpoint:"
ls -td "$OUTPUT_DIR"/checkpoints/*/pretrained_model 2>/dev/null | head -1
