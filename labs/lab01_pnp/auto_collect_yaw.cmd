#!/bin/bash
# Lab 01 — yaw-aware scripted expert (reusable collector).
# Run on the .43 desktop (needs DISPLAY + GLFW).
#
# Visual smoke (3 attempts, no dataset write):
#   ./labs/lab01_pnp/auto_collect_yaw.cmd
#
# One-click record / expand (always writes; prefers this wrapper):
#   ./labs/lab01_pnp/collect_yaw.cmd
#   ./labs/lab01_pnp/collect_yaw.cmd expand
#
# Manual record still works:
#   LAB01_YAW_RECORD=1 LAB01_YAW_TARGET_SUCCESSES=50 ./labs/lab01_pnp/auto_collect_yaw.cmd
# Do not delete the dataset root before a top-up (meta/info.json → resume).
#
#   LAB01_YAW_EPISODES=1 ./labs/lab01_pnp/auto_collect_yaw.cmd
#   LAB01_YAW_HEADLESS=1 MUJOCO_GL=osmesa ./labs/lab01_pnp/auto_collect_yaw.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

HEADLESS="${LAB01_YAW_HEADLESS:-0}"
if [[ "$HEADLESS" != "1" && "$HEADLESS" != "true" && -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset — GLFW needs the .43 graphical session (or ssh -X)." >&2
    echo "Headless: LAB01_YAW_HEADLESS=1 MUJOCO_GL=osmesa $0" >&2
    exit 1
fi

if [[ "$HEADLESS" == "1" || "$HEADLESS" == "true" ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
    unset DISPLAY || true
else
    export MUJOCO_GL="${MUJOCO_GL:-glfw}"
fi

echo "=== Lab 01: yaw-aware scripted expert (no LLM / no ACT) ==="
echo "Align wrist to cube faces, standoff, creep, close, lift, place."
echo "GL:       $MUJOCO_GL  DISPLAY=${DISPLAY:-unset}"
echo "episodes: ${LAB01_YAW_EPISODES:-3}  random_cube=1"
echo "target:   ${LAB01_YAW_TARGET_SUCCESSES:-0} successful episodes (0=attempt count)"
echo "record:   ${LAB01_YAW_RECORD:-0}  root=${LAB01_YAW_DATASET_ROOT}"
echo ""

EXTRA=()
if [[ "$HEADLESS" == "1" || "$HEADLESS" == "true" ]]; then
    EXTRA+=(--headless)
fi
if [[ "${LAB01_YAW_RECORD:-0}" == "1" || "${LAB01_YAW_RECORD:-}" == "true" ]]; then
    EXTRA+=(--record)
fi
if [[ "${LAB01_YAW_SAVE_FAILURES:-0}" == "1" || "${LAB01_YAW_SAVE_FAILURES:-}" == "true" ]]; then
    EXTRA+=(--save-failures)
fi
if [[ "${LAB01_YAW_TARGET_SUCCESSES:-0}" != "0" ]]; then
    EXTRA+=(--target-successes "${LAB01_YAW_TARGET_SUCCESSES}")
fi
EXTRA+=(--dataset-root "${LAB01_YAW_DATASET_ROOT}")
EXTRA+=(--repo-id "${LAB01_YAW_REPO_ID}")

exec "$PYTHON" "$_LAB01_DIR/auto_collect_yaw.py" \
    --episodes "${LAB01_YAW_EPISODES:-3}" \
    "${EXTRA[@]}" \
    "$@"
