#!/bin/bash
# Lab 01 — ACT imitation learning.
#
# Defaults: labs/lab01_pnp/_env.sh (override via LAB01_* env vars).
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/train_act.cmd
#
# Options (pass through to lerobot-train):
#   ./labs/lab01_pnp/train_act.cmd --steps 100000 --batch_size 16
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

echo "=== Lab 01: ACT training ==="
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Output:   $LAB01_ACT_OUTPUT"
echo "Steps:    $LAB01_ACT_STEPS  batch_size: $LAB01_ACT_BATCH_SIZE"
if [[ -n "${LAB01_ACT_STATE_DIM:-}" ]]; then
    echo "State:    first $LAB01_ACT_STATE_DIM dims of observation.state (on-disk dataset unchanged)"
    export LAB01_ACT_STATE_DIM
else
    echo "State:    full observation.state (15-D pos+vel+ee)"
fi
echo "Log:      $REPO_ROOT/$LAB01_ACT_LOG"
echo ""

set +e
"$PYTHON" -m simstudio.scripts.train_act \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
    --dataset.root="$LAB01_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --output_dir="$LAB01_ACT_OUTPUT" \
    --job_name="$LAB01_ACT_JOB_NAME" \
    --batch_size="$LAB01_ACT_BATCH_SIZE" \
    --steps="$LAB01_ACT_STEPS" \
    --save_checkpoint=true \
    --save_freq="$LAB01_ACT_SAVE_FREQ" \
    "$@" \
    2>&1 | tee "$LAB01_ACT_LOG"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Training exited with error ($status). Last lines of $LAB01_ACT_LOG:"
    tail -30 "$LAB01_ACT_LOG"
    exit "$status"
fi

echo ""
echo "Training complete. Latest checkpoint:"
ls -td "$LAB01_ACT_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1
