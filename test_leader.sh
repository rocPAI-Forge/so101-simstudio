#!/bin/bash
# Leader arm teleop 测试脚本
# Ctrl+C 退出

echo "=== Leader Arm Teleop Test ==="
echo "Controls:"
echo "  Move leader arm to control robot"
echo "  Ctrl+C: Quit"
echo ""

.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_leader_teleop.yaml \
    --view_mode mujoco
