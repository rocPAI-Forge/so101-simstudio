#!/bin/bash
# Joy-Con record quick-test (right Joy-Con / mujoco / 2 episodes).
# Stick: forward/back=reach, left/right=base swing; R=up, stick press=down; ZR=toggle gripper.
# Recording: A=next, Y=re-record, +=stop; keyboard N/R/Q still works (evdev).
# Debug: SO101_JOYCON_DEBUG=1 ./scripts/quicktest/joycon.cmd
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export SO101_JOYCON_DEBUG="${SO101_JOYCON_DEBUG:-0}"
rm -rf ./datasets/joycon-quicktest
"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_joycon.yaml \
    --dataset.root ./datasets/joycon-quicktest \
    --dataset.repo_id alexhegit/so101_joycon_quicktest \
    --dataset.num_episodes 2 --resume false --view_mode mujoco \
    2>&1 | tee test.log
