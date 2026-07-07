#!/bin/bash
# Keyboard recording smoke test.
# Usage: keyboard_record.sh [episodes] [resume] [view_mode]

set -e
source "$(dirname "$0")/_common.sh"

EPISODES=${1:-3}
RESUME=${2:-false}
VIEW_MODE=${3:-mujoco}
ROOT="./datasets/keyboard-record-test"
REPO_ID="alexhegit/so101_mujoco_keyboard_record_test"

echo "=== Keyboard Record Smoke Test ==="
echo "Episodes: $EPISODES"
echo "Resume: $RESUME"
echo "View mode: $VIEW_MODE"
echo "Dataset: $ROOT"
echo ""
echo "Controls:"
echo "  W/S: Y-axis (forward/backward)"
echo "  A/D: X-axis (left/right)"
echo "  Z/X: Z-axis (up/down)"
echo "  I/K: Wrist flex"
echo "  [/]: Wrist rotation"
echo "  C: Gripper close"
echo "  O: Gripper open"
echo ""
echo "Recording shortcuts:"
echo "  Left arrow: Cancel current episode"
echo "  Right arrow: Save episode, next one"
echo "  ESC: Stop recording"
echo ""

"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME" \
    --view_mode "$VIEW_MODE"
