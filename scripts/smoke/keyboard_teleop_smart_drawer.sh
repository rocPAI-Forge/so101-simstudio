#!/bin/bash
# Keyboard teleop on the smart_drawer scene (Text2MuJoCo 02). Ctrl+C to quit.

set -e
source "$(dirname "$0")/_common.sh"

echo "=== Keyboard Teleop — smart_drawer (Text2MuJoCo 02) ==="
echo "Task: grasp lower handle, hold C, then hold W to pull drawer open (+Y)"
echo "Controls:"
echo "  W: pull toward arm / open drawer (+Y)"
echo "  S: push into cabinet / close drawer (-Y)"
echo "  A/D: X-axis (left/right)"
echo "  Z/X: Z-axis (up/down)"
echo "  I/K: Wrist flex"
echo "  [/]: Wrist rotation"
echo "  C: Gripper close (hold while pulling)"
echo "  O: Gripper open"
echo ""
echo "No mid-session reset in teleop-only mode."
echo "Stuck arm / drawer: Ctrl+C then re-run this script (loads home)."
echo "Record mode resets between episodes (Left/R = rerecord)."
echo ""
echo "Press Ctrl+C to quit."
echo ""

"$PYTHON" -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_keyboard_teleop_smart_drawer.yaml
