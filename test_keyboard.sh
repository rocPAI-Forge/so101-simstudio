#!/bin/bash
# Keyboard teleop 测试脚本
# 按 ESC 退出

echo "=== Keyboard Teleop Test ==="
echo "Controls:"
echo "  W/S: Y-axis (forward/backward)"
echo "  A/D: X-axis (left/right)"
echo "  Z/X: Z-axis (up/down)"
echo "  I/K: Wrist flex"
echo "  [/]: Wrist rotation"
echo "  C: Gripper close"
echo "  O: Gripper open"
echo ""
echo "Recording shortcuts (record mode only):"
echo "  Left arrow: Cancel current episode"
echo "  Right arrow: Save episode, next one"
echo "  ESC: Stop recording"
echo ""

.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_keyboard_teleop.yaml \
    --view_mode mujoco
