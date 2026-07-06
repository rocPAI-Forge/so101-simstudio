#!/bin/bash
# Keyboard 录制测试脚本
# 用法: ./test_keyboard_record.sh [episodes] [resume]

set -e

EPISODES=${1:-3}
RESUME=${2:-false}
ROOT="./datasets/keyboard-record-test"
REPO_ID="alexhegit/so101_mujoco_keyboard_record_test"

echo "=== Keyboard Record Test ==="
echo "Episodes: $EPISODES"
echo "Resume: $RESUME"
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

.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME"
