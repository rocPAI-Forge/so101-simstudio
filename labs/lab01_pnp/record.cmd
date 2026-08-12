#!/bin/bash
# Lab 01 — leader-arm pick-and-place recording.
#
# Defaults: labs/lab01_pnp/_env.sh (override via LAB01_* env vars).
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
#   LAB01_DATASET_NAME=my-run ./labs/lab01_pnp/record.cmd
#
# Fresh run: rm -rf "$LAB01_DATASET_ROOT"  (see _env.sh for path)
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

echo "=== Lab 01: SO-101 pick-and-place recording ==="
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Episodes: $LAB01_NUM_EPISODES x ${LAB01_EPISODE_TIME_S}s (reset ${LAB01_RESET_TIME_S}s)"
echo "Log:      $REPO_ROOT/test.log"
echo ""

set +e
"$PYTHON" -m simstudio.scripts.record \
    --config "$LAB01_RECORD_CONFIG" \
    --view_mode mujoco \
    --dataset.root "$LAB01_DATASET_ROOT" \
    --dataset.repo_id "$LAB01_DATASET_REPO_ID" \
    --dataset.num_episodes "$LAB01_NUM_EPISODES" \
    --dataset.episode_time_s "$LAB01_EPISODE_TIME_S" \
    --dataset.reset_time_s "$LAB01_RESET_TIME_S" \
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
