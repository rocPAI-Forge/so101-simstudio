#!/bin/bash
# Lab 01 — SmolVLA fine-tune from lerobot/smolvla_base on so101-simstudio-pnp.
#
# Prereq: dataset at ./datasets/so101-simstudio-pnp (validate first).
#         .venv-rocm with lerobot[smolvla] (make rocm-sync).
#         Network for first-run HF weight download.
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/train.cmd
#
# Options (pass through to lerobot-train):
#   ./labs/lab01_pnp/train.cmd --steps 20000 --batch_size 1
#   ./labs/lab01_pnp/train.cmd --resume true
set -euo pipefail
source "$(dirname "$0")/../../scripts/quicktest/_common.sh"

LEROBOT_TRAIN="$(dirname "$PYTHON")/lerobot-train"
if [[ ! -x "$LEROBOT_TRAIN" ]]; then
    echo "lerobot-train not found. Run: make rocm-sync" >&2
    exit 1
fi

OUTPUT_DIR="${LAB01_TRAIN_OUTPUT:-./outputs/train/lab01_pnp_smolvla}"
STEPS="${LAB01_TRAIN_STEPS:-7500}"
BATCH_SIZE="${LAB01_TRAIN_BATCH_SIZE:-4}"
WARMUP="${LAB01_TRAIN_WARMUP:-500}"
SAVE_FREQ="${LAB01_TRAIN_SAVE_FREQ:-2000}"

# Dataset keys: camera_top/front/wrist → smolvla_base: camera1/2/3 (+ empty_cameras=1)
RENAME_MAP='{"observation.images.camera_top":"observation.images.camera1","observation.images.camera_front":"observation.images.camera2","observation.images.camera_wrist":"observation.images.camera3"}'

echo "=== Lab 01: SmolVLA fine-tune ==="
echo "Base:     lerobot/smolvla_base"
echo "Dataset:  ./datasets/so101-simstudio-pnp"
echo "Output:   $OUTPUT_DIR"
echo "Steps:    $STEPS  batch_size: $BATCH_SIZE  warmup: $WARMUP"
echo "Log:      $REPO_ROOT/train.log"
echo ""

set +e
"$LEROBOT_TRAIN" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --policy.scheduler_warmup_steps="$WARMUP" \
    --dataset.repo_id=alexhegit/so101-simstudio-pnp \
    --dataset.root=./datasets/so101-simstudio-pnp \
    --dataset.video_backend=pyav \
    --output_dir="$OUTPUT_DIR" \
    --job_name=lab01_pnp_smolvla \
    --rename_map="$RENAME_MAP" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
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
