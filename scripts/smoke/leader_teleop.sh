#!/bin/bash
# Leader arm teleop smoke test (no recording). Ctrl+C to quit.

set -e
source "$(dirname "$0")/_common.sh"

echo "=== Leader Arm Teleop Smoke Test ==="
echo "Controls:"
echo "  Move leader arm to control robot"
echo "  Ctrl+C: Quit"
echo ""

"$PYTHON" -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_leader_teleop.yaml
