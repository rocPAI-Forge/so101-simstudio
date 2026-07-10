#!/bin/bash
# Replay keyboard smoke-test dataset in MuJoCo.
# Usage: keyboard_replay.sh [episode|all]
#   episode: 0, 1, 2, ... (default 0)
#   all: replay every episode sequentially

set -e
source "$(dirname "$0")/_common.sh"

TARGET=${1:-0}
ROOT="./datasets/keyboard-record-test"
REPO_ID="alexhegit/so101_mujoco_keyboard_record_test"

echo "=== Keyboard Replay Smoke Test ==="
echo "Dataset: $ROOT"
echo "Target: $TARGET"
echo ""

if [[ "$TARGET" == "all" ]]; then
    "$PYTHON" -m simstudio.scripts.replay_multi \
        --config configs/so101_mujoco_replay_multi.yaml \
        --dataset.root "$ROOT" \
        --dataset.repo_id "$REPO_ID" \
        --dataset.episodes all
else
    "$PYTHON" -m simstudio.scripts.replay \
        --config configs/so101_mujoco_replay.yaml \
        --dataset.root "$ROOT" \
        --dataset.repo_id "$REPO_ID" \
        --dataset.episode "$TARGET"
fi
