#!/bin/bash
# ACT RGB-only on place100d (drops camera_top_depth). Same steps as train_act_rgbd.cmd.
set -euo pipefail
_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$_DIR/_env.sh"
export RAX_ACT_USE_DEPTH=0
export RAX_ACT_OUTPUT="${RAX_ACT_RGB_OUTPUT}"
export RAX_ACT_LOG="${RAX_ACT_RGB_LOG}"
export RAX_ACT_JOB_NAME="${RAX_ACT_RGB_JOB_NAME}"
exec "$_DIR/train_act.cmd" "$@"
