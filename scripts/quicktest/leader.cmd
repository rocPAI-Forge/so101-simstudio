#!/bin/bash
# Real leader arm record quick-test (mujoco / 2 episodes).
# Prereq: leader on /dev/ttyACM0, dialout group, calibrated.
# Control: 1:1 position; reset_arm=follow, cube=random.
# Recording: N/→, R/←, Q/ESC (evdev). Other port: --teleop.port /dev/ttyACM1
set -euo pipefail
source "$(dirname "$0")/_common.sh"

rm -rf ./datasets/leader-quicktest
"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_leader.yaml \
    --dataset.root ./datasets/leader-quicktest \
    --dataset.repo_id alexhegit/so101_leader_quicktest \
    --dataset.num_episodes 2 --resume false --view_mode mujoco \
    "$@" \
    2>&1 | tee test.log
