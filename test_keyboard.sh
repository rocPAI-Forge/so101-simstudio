#!/bin/bash
# Keyboard teleop 测试脚本
# 按 ESC 退出

echo "=== Keyboard Teleop Test ==="
echo "Controls:"
echo "  Arrow keys: X/Y movement"
echo "  Z/X: Z-axis up/down"
echo "  1/2: Wrist rotation"
echo "  Space: Gripper close/open"
echo "  ESC: Quit"
echo ""

.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_keyboard_teleop.yaml \
    --view_mode mujoco
