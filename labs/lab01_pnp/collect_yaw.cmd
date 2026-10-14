#!/bin/bash
# Lab 01 — one-click yaw-aware expert dataset (create or expand).
# Always records. Run on the .43 desktop (DISPLAY + GLFW).
#
#   ./labs/lab01_pnp/collect_yaw.cmd
#       no dataset     → create, fill 50 successes (yaw + 2×2 XY quotas)
#       1–99 successes → resume, fill to 100 (empty XY cells first)
#       already ≥100   → exit 0 (override with expand N)
#
#   ./labs/lab01_pnp/collect_yaw.cmd init          # fill to 50
#   ./labs/lab01_pnp/collect_yaw.cmd expand        # fill to 100
#   ./labs/lab01_pnp/collect_yaw.cmd expand 150    # fill to 150
#
# Low-level smoke (no write): ./labs/lab01_pnp/auto_collect_yaw.cmd
# Headless: LAB01_YAW_HEADLESS=1 MUJOCO_GL=osmesa ./labs/lab01_pnp/collect_yaw.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

MODE="${1:-auto}"
TARGET_ARG="${2:-}"

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
    grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
    exit 0
fi

INIT_N="${LAB01_YAW_INIT_TARGET:-50}"
EXPAND_N="${LAB01_YAW_EXPAND_TARGET:-100}"

_count_successes() {
    local log="$1"
    "$PYTHON" - "$log" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
n = 0
if p.is_file():
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("success"):
            n += 1
print(n)
PY
}

SPAWN_LOG="${LAB01_YAW_DATASET_ROOT}/spawns.jsonl"
INFO="${LAB01_YAW_DATASET_ROOT}/meta/info.json"
EXISTING="$(_count_successes "$SPAWN_LOG")"

case "$MODE" in
    auto)
        if [[ "$EXISTING" -eq 0 ]]; then
            TARGET="$INIT_N"
            ACTION="create"
        elif [[ "$EXISTING" -lt "$EXPAND_N" ]]; then
            TARGET="$EXPAND_N"
            ACTION="expand"
        else
            echo "Dataset already has $EXISTING successes at $LAB01_YAW_DATASET_ROOT"
            echo "To grow further: $0 expand $((EXISTING + 50))"
            exit 0
        fi
        ;;
    init)
        TARGET="$INIT_N"
        ACTION="create-or-fill"
        ;;
    expand)
        if [[ -n "$TARGET_ARG" ]]; then
            if [[ ! "$TARGET_ARG" =~ ^[0-9]+$ ]]; then
                echo "expand count must be an integer, got: $TARGET_ARG" >&2
                exit 2
            fi
            TARGET="$TARGET_ARG"
        else
            TARGET="$EXPAND_N"
        fi
        ACTION="expand"
        if [[ "$TARGET" -le "$EXISTING" ]]; then
            echo "Already at $EXISTING successes (target $TARGET). Nothing to do."
            exit 0
        fi
        ;;
    *)
        echo "Unknown mode: $MODE (use auto | init | expand [N])" >&2
        exit 2
        ;;
esac

if [[ "$ACTION" == "expand" && ! -f "$INFO" ]]; then
    echo "expand needs an existing LeRobot root (missing $INFO)." >&2
    echo "First run: $0 init" >&2
    exit 1
fi

export LAB01_YAW_RECORD=1
export LAB01_YAW_TARGET_SUCCESSES="$TARGET"

echo "=== Lab 01: one-click yaw-aware collect ==="
echo "mode:     $MODE ($ACTION)"
echo "root:     $LAB01_YAW_DATASET_ROOT"
echo "existing: $EXISTING successes"
echo "target:   $TARGET successes (resume if meta/info.json exists)"
echo "need:     $((TARGET - EXISTING)) more"
echo "log:      quotas printed by collector (yaw bins + XY cells)"
echo ""

exec "$_LAB01_DIR/auto_collect_yaw.cmd"
