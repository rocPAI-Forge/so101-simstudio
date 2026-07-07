#!/bin/bash
# Joy-Con 录制测试脚本
# 用法: ./test_joycon_record.sh [episodes] [resume] [side] [view_mode]

set -e

EPISODES=${1:-3}
RESUME=${2:-false}
SIDE=${3:-right}
VIEW_MODE=${4:-mujoco}
ROOT="./datasets/joycon-${SIDE}-record-test"
REPO_ID="alexhegit/so101_mujoco_joycon_${SIDE}_record_test"

echo "=== Joy-Con Record Test ==="
echo "Side: $SIDE"
echo "Episodes: $EPISODES"
echo "Resume: $RESUME"
echo "View mode: $VIEW_MODE"
echo "Dataset: $ROOT"
echo ""
echo "Controls ($SIDE Joy-Con):"
if [ "$SIDE" = "right" ]; then
    echo "  Stick: X/Y movement"
    echo "  R: Z-axis up"
    echo "  Stick press: Z-axis down"
    echo "  Tilt: Wrist rotation"
    echo "  ZR hold: Gripper close"
    echo "  Plus: Quit"
else
    echo "  Stick: X/Y movement"
    echo "  L: Z-axis up"
    echo "  Stick press: Z-axis down"
    echo "  Tilt: Wrist rotation"
    echo "  ZL hold: Gripper close"
    echo "  Minus: Quit"
fi
echo ""

.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_joycon.yaml \
    --teleop.side "$SIDE" \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME" \
    --view_mode "$VIEW_MODE"
