#!/bin/bash
# Keyboard teleop smoke test (no recording). Ctrl+C to quit.

set -e
source "$(dirname "$0")/_common.sh"

echo "=== Keyboard Teleop Smoke Test ==="
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
    --config configs/so101_mujoco_keyboard_teleop.yaml
