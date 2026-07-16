#!/bin/bash
# Keyboard record quick-test (mujoco / 2 episodes).
# Movement: W/S/A/D/Z/X, I/K, [ / ], O/C; recording: N/→, R/←, Q/ESC (evdev).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

rm -rf ./datasets/keyboard-quicktest
"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml \
    --dataset.root ./datasets/keyboard-quicktest \
    --dataset.repo_id alexhegit/so101_keyboard_quicktest \
    --dataset.num_episodes 2 --resume false --view_mode mujoco \
    2>&1 | tee test.log
