#!/bin/bash
# ACT RGB + top-depth gray on place100d. Same steps as train_act_rgb.cmd.
set -euo pipefail
_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$_DIR/_env.sh"
export RAX_ACT_USE_DEPTH=1
export RAX_ACT_OUTPUT="${RAX_ACT_DEPTH_OUTPUT}"
export RAX_ACT_LOG="${RAX_ACT_DEPTH_LOG}"
export RAX_ACT_JOB_NAME="${RAX_ACT_DEPTH_JOB_NAME}"
exec "$_DIR/train_act.cmd" "$@"
