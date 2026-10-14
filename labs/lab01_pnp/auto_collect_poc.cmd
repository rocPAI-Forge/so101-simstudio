#!/bin/bash
# Lab 01 — visual PoC of scripted keyframe grasp (no LLM / no ACT).
# Run on the .43 desktop (needs DISPLAY + GLFW).
#
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/auto_collect_poc.cmd
#
#   LAB01_POC_EPISODES=1 LAB01_POC_RANDOM=1 ./labs/lab01_pnp/auto_collect_poc.cmd
#   LAB01_POC_HEADLESS=1 MUJOCO_GL=osmesa ./labs/lab01_pnp/auto_collect_poc.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

HEADLESS="${LAB01_POC_HEADLESS:-0}"
if [[ "$HEADLESS" != "1" && "$HEADLESS" != "true" && -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset — GLFW needs the .43 graphical session (or ssh -X)." >&2
    echo "Headless: LAB01_POC_HEADLESS=1 MUJOCO_GL=osmesa $0" >&2
    exit 1
fi

if [[ "$HEADLESS" == "1" || "$HEADLESS" == "true" ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
    unset DISPLAY || true
else
    export MUJOCO_GL="${MUJOCO_GL:-glfw}"
fi

echo "=== Lab 01: scripted grasp PoC (no LLM / no ACT) ==="
echo "Decelerate into cube, creep, close, lift, place."
echo "GL:       $MUJOCO_GL  DISPLAY=${DISPLAY:-unset}"
echo "episodes: ${LAB01_POC_EPISODES:-3}  random_cube=${LAB01_POC_RANDOM:-0}"
echo "record:   ${LAB01_POC_RECORD:-0}  root=${LAB01_POC_DATASET_ROOT:-.tmp/lab01_scripted_poc}"
echo ""

EXTRA=()
if [[ "$HEADLESS" == "1" || "$HEADLESS" == "true" ]]; then
    EXTRA+=(--headless)
fi
if [[ "${LAB01_POC_RECORD:-0}" == "1" || "${LAB01_POC_RECORD:-}" == "true" ]]; then
    EXTRA+=(--record)
fi
if [[ "${LAB01_POC_SAVE_FAILURES:-0}" == "1" || "${LAB01_POC_SAVE_FAILURES:-}" == "true" ]]; then
    EXTRA+=(--save-failures)
fi

exec "$PYTHON" "$_LAB01_DIR/auto_collect_poc.py" \
    --episodes "${LAB01_POC_EPISODES:-3}" \
    "${EXTRA[@]}" \
    "$@"
