#!/bin/bash
# Keyboard teleop on the kitchen_counter workcell (no recording). Ctrl+C to quit.

set -e
source "$(dirname "$0")/_common.sh"

echo "=== Keyboard Teleop — kitchen_counter (Text2MuJoCo 01 workcell) ==="
echo "Pick: green cube, red cube, blue cylinder  →  place in the blue box"
echo "Controls:"
echo "  W/S: Y-axis (forward/backward)"
echo "  A/D: X-axis (left/right)"
echo "  Z/X: Z-axis (up/down)"
echo "  I/K: Wrist flex"
echo "  [/]: Wrist rotation"
echo "  C: Gripper close"
echo "  O: Gripper open"
echo ""
echo "Press Ctrl+C to quit."
echo ""

"$PYTHON" -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_keyboard_teleop_kitchen.yaml
