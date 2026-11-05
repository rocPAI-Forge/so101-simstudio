#!/bin/bash
# Train ACT 6-D on the RAX drawer place dataset (intended for MI300X / DORobot).
set -euo pipefail
_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$_DIR/_env.sh"
PYTHON="${PYTHON:-$REPO_ROOT/.venv-rocm/bin/python}"
mkdir -p "$(dirname "$RAX_ACT_LOG")"

echo "=== RAX drawer ACT 6-D ==="
echo "Dataset: $RAX_DATASET_ROOT ($RAX_DATASET_REPO_ID)"
echo "Output:  $RAX_ACT_OUTPUT  steps=$RAX_ACT_STEPS batch=$RAX_ACT_BATCH_SIZE state_dim=$RAX_ACT_STATE_DIM"

export LAB01_ACT_STATE_DIM="$RAX_ACT_STATE_DIM"
set +e
"$PYTHON" -m simstudio.scripts.train_act \
  --policy.type=act \
  --policy.push_to_hub=false \
  --dataset.repo_id="$RAX_DATASET_REPO_ID" \
  --dataset.root="$RAX_DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --output_dir="$RAX_ACT_OUTPUT" \
  --job_name="$RAX_ACT_JOB_NAME" \
  --batch_size="$RAX_ACT_BATCH_SIZE" \
  --steps="$RAX_ACT_STEPS" \
  --save_checkpoint=true \
  --save_freq="$RAX_ACT_SAVE_FREQ" \
  "$@" \
  2>&1 | tee "$RAX_ACT_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
