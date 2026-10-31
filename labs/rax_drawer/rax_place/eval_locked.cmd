#!/bin/bash
# Replay the frozen open → place → close controller on the locked 20 scenes.
#
# Reported score: 15/20, freeze 20261031T093325Z (session/CONTROLLER_FROZEN_3).
# This replays that program. It is not a tuning split.
#
#   ./labs/rax_drawer/rax_place/eval_locked.cmd
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${ROOT}/src:${ROOT}/labs${PYTHONPATH:+:$PYTHONPATH}"
PY="${ROOT}/.venv-rocm/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Missing $PY. Run make rocm-sync first." >&2
    exit 1
fi
exec "$PY" -m labs.rax_drawer.rax_place.run_test --round 3
