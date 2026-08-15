#!/bin/bash
# Lab 01 — SmolVLA fine-tune from lerobot/smolvla_base.
#
# Defaults: labs/lab01_pnp/_env.sh (Shared + Train sections). Set batch/steps for your GPU.
#
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/train.cmd
#   LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_STEPS=50000 LAB01_TRAIN_SAVE_FREQ=10000 \
#     LAB01_TRAIN_NUM_WORKERS=8 ./labs/lab01_pnp/train.cmd
#   ./labs/lab01_pnp/train.cmd --steps 20000 --batch_size 1
#   ./labs/lab01_pnp/train.cmd --resume true
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

LEROBOT_TRAIN="$(dirname "$PYTHON")/lerobot-train"
if [[ ! -x "$LEROBOT_TRAIN" ]]; then
    echo "lerobot-train not found. Run: make rocm-sync" >&2
    exit 1
fi

echo "=== Lab 01: SmolVLA fine-tune ==="
echo "Base:     lerobot/smolvla_base"
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Output:   $LAB01_TRAIN_OUTPUT"
echo "Steps:    $LAB01_TRAIN_STEPS  batch_size: $LAB01_TRAIN_BATCH_SIZE  warmup: $LAB01_TRAIN_WARMUP"
echo "Workers:  $LAB01_TRAIN_NUM_WORKERS  save_freq: $LAB01_TRAIN_SAVE_FREQ"
echo "Log:      $REPO_ROOT/train.log"
echo ""

set +e
"$LEROBOT_TRAIN" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --policy.scheduler_warmup_steps="$LAB01_TRAIN_WARMUP" \
    --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
    --dataset.root="$LAB01_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --output_dir="$LAB01_TRAIN_OUTPUT" \
    --job_name=lab01_pnp_smolvla \
    --rename_map="$LAB01_RENAME_MAP" \
    --batch_size="$LAB01_TRAIN_BATCH_SIZE" \
    --steps="$LAB01_TRAIN_STEPS" \
    --num_workers="$LAB01_TRAIN_NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$LAB01_TRAIN_SAVE_FREQ" \
    "$@" \
    2>&1 | tee train.log
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Training exited with error ($status). Last lines of train.log:"
    tail -30 train.log
    echo ""
    echo "Fix the error above, then re-run: ./labs/lab01_pnp/train.cmd"
    read -r -p "Press Enter to close this window..."
fi
exit "$status"
