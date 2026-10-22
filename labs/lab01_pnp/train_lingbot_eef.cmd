#!/bin/bash
# Fair LingBot-VA path: 7-D LIBERO EEF deltas + gripper, not stuffed joints.
#
# The derived dataset is local scratch. Build it automatically when absent,
# then delegate model training to the shared LingBot entrypoint.
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_LAB01_DIR/../.." && pwd)"
_USER_OUTPUT="${LAB01_LINGBOT_OUTPUT:-}"
_USER_STEPS="${LAB01_LINGBOT_STEPS:-}"
_USER_BATCH_SIZE="${LAB01_LINGBOT_BATCH_SIZE:-}"
_USER_SAVE_FREQ="${LAB01_LINGBOT_SAVE_FREQ:-}"
_USER_NUM_WORKERS="${LAB01_LINGBOT_NUM_WORKERS:-}"
_USER_CHANNELS="${LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS:-}"
_USER_RENAME_MAP="${LAB01_LINGBOT_RENAME_MAP:-}"
source "$_LAB01_DIR/_env.sh"

export LAB01_LINGBOT_EEF_MODE=true
export LAB01_DATASET_ROOT="$LAB01_LINGBOT_EEF_DATASET_ROOT"
export LAB01_DATASET_REPO_ID="$LAB01_LINGBOT_EEF_DATASET_REPO_ID"
export LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS="${_USER_CHANNELS:-[0,1,2,3,4,5,6]}"
export LAB01_LINGBOT_RENAME_MAP="${_USER_RENAME_MAP:-{\"observation.images.camera_front\":\"observation.images.image\",\"observation.images.camera_wrist\":\"observation.images.image2\"}}"
export LAB01_LINGBOT_OUTPUT="${_USER_OUTPUT:-$LAB01_LINGBOT_EEF_OUTPUT}"
export LAB01_LINGBOT_STEPS="${_USER_STEPS:-$LAB01_LINGBOT_EEF_STEPS}"
export LAB01_LINGBOT_BATCH_SIZE="${_USER_BATCH_SIZE:-$LAB01_LINGBOT_EEF_BATCH_SIZE}"
export LAB01_LINGBOT_SAVE_FREQ="${_USER_SAVE_FREQ:-$LAB01_LINGBOT_EEF_SAVE_FREQ}"
export LAB01_LINGBOT_NUM_WORKERS="${_USER_NUM_WORKERS:-$LAB01_LINGBOT_EEF_NUM_WORKERS}"

if [[ ! -f "$LAB01_DATASET_ROOT/meta/info.json" ]]; then
    "$REPO_ROOT/.venv-rocm/bin/python" "$_LAB01_DIR/build_lingbot_eef_dataset.py" \
        --source "$REPO_ROOT/datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2" \
        --output "$LAB01_DATASET_ROOT"
fi

exec "$_LAB01_DIR/train_lingbot_va.cmd" "$@"
