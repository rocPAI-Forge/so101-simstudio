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
echo "Movement (hold): W/S Y | A/D X | Z/X Z | I/K wrist | [/] roll | O/C gripper"
echo ""
echo "Recording (press once; same in mujoco and rerun):"
echo "  Left arrow / R : Cancel current episode"
echo "  Right arrow / N: Save episode, next one"
echo "  ESC / Q        : Stop recording"
echo ""

"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME" \
    --view_mode "$VIEW_MODE"
