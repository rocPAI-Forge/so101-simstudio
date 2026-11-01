#!/bin/bash
# Finetune SmolVLA on the RAX drawer place dataset (MI300X / DORobot).
set -euo pipefail
_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$_DIR/_env.sh"
PYTHON="${PYTHON:-$REPO_ROOT/.venv-rocm/bin/python}"
mkdir -p "$(dirname "$RAX_SMOL_LOG")"

echo "=== RAX drawer SmolVLA ==="
echo "Dataset: $RAX_DATASET_ROOT"
echo "Output:  $RAX_SMOL_OUTPUT  steps=$RAX_SMOL_STEPS batch=$RAX_SMOL_BATCH_SIZE"

set +e
"$PYTHON" -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=0 \
  --dataset.repo_id="$RAX_DATASET_REPO_ID" \
  --dataset.root="$RAX_DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --output_dir="$RAX_SMOL_OUTPUT" \
  --job_name="$RAX_SMOL_JOB_NAME" \
  --batch_size="$RAX_SMOL_BATCH_SIZE" \
  --steps="$RAX_SMOL_STEPS" \
  --save_checkpoint=true \
  --save_freq="$RAX_SMOL_SAVE_FREQ" \
  --rename_map="$RAX_SMOL_RENAME_MAP" \
  "$@" \
  2>&1 | tee "$RAX_SMOL_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
