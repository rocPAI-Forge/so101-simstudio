#!/bin/bash
# Lab 01 — leader-arm pick-and-place recording (30 episodes, 60s each, 5s reset).
#
# Prereq: leader on /dev/ttyACM0, dialout group, feetech-servo-sdk installed.
# Control: N/→ save & next, R/← re-record, Q/ESC stop (evdev).
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/record.cmd
#
# Options:
#   ./labs/lab01_pnp/record.cmd --view_mode rerun
#   ./labs/lab01_pnp/record.cmd --teleop.port /dev/ttyACM1
#   ./labs/lab01_pnp/record.cmd --resume true
#
# Fresh run: rm -rf ./datasets/so101-simstudio-pnp
set -euo pipefail
source "$(dirname "$0")/../../scripts/quicktest/_common.sh"

echo "=== Lab 01: SO-101 pick-and-place recording ==="
echo "Dataset: ./datasets/so101-simstudio-pnp (30 x 60s, reset 5s)"
echo "Log:     $REPO_ROOT/test.log"
echo ""

set +e
"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_pick_leader.yaml \
    --view_mode mujoco \
    --dataset.root ./datasets/so101-simstudio-pnp \
    --dataset.repo_id alexhegit/so101-simstudio-pnp \
    --dataset.num_episodes 30 \
    --dataset.episode_time_s 60 \
    --dataset.reset_time_s 5 \
    --resume false \
    "$@" \
    2>&1 | tee test.log
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Recording exited with error ($status). Last lines of test.log:"
    tail -20 test.log
    echo ""
    echo "Fix the error above, then re-run: ./labs/lab01_pnp/record.cmd"
    read -r -p "Press Enter to close this window..."
fi
exit "$status"
