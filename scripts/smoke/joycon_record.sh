#!/bin/bash
# Joy-Con recording smoke test.
# Usage: joycon_record.sh [episodes] [resume] [side] [view_mode]

set -e
source "$(dirname "$0")/_common.sh"

EPISODES=${1:-3}
RESUME=${2:-false}
SIDE=${3:-right}
VIEW_MODE=${4:-mujoco}
ROOT="./datasets/joycon-${SIDE}-record-test"
REPO_ID="alexhegit/so101_mujoco_joycon_${SIDE}_record_test"

if [[ "$SIDE" != "left" && "$SIDE" != "right" ]]; then
    echo "side must be 'left' or 'right', got: $SIDE" >&2
    exit 1
fi

CONFIG="configs/so101_mujoco_joycon.yaml"
if [[ "$SIDE" == "left" ]]; then
    CONFIG="configs/so101_mujoco_joycon_left.yaml"
fi

echo "=== Joy-Con Record Smoke Test ==="
echo "Side: $SIDE"
echo "Episodes: $EPISODES"
echo "Resume: $RESUME"
echo "View mode: $VIEW_MODE"
echo "Config: $CONFIG"
echo "Dataset: $ROOT"
echo ""
echo "Controls ($SIDE Joy-Con):"
if [[ "$SIDE" == "right" ]]; then
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

"$PYTHON" -m simstudio.scripts.record \
    --config "$CONFIG" \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME" \
    --view_mode "$VIEW_MODE"
