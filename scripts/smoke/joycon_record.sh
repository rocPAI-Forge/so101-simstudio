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
    echo "  Stick forward/back: reach in/out; left/right: base swing"
    echo "  R: Z up; stick press: Z down"
    echo "  Tilt: wrist flex / roll"
    echo "  ZR: toggle gripper (default)"
    echo "  A/Y/+: save & next / re-record / stop (one-handed recording)"
else
    echo "  Stick forward/back: reach in/out; left/right: base swing"
    echo "  L: Z up; stick press: Z down"
    echo "  Tilt: wrist flex / roll"
    echo "  ZL: toggle gripper (default)"
    echo "  d-pad Down/Up, Minus: save & next / re-record / stop"
fi
echo "  Keyboard (evdev): N/Right save, R/Left re-record, Q/ESC stop"
echo ""

"$PYTHON" -m simstudio.scripts.record \
    --config "$CONFIG" \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME" \
    --view_mode "$VIEW_MODE"
