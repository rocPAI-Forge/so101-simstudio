#!/bin/bash
# Collect successful RAX place episodes into a LeRobot v3 dataset (halo / EGL).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/quicktest/_common.sh" 2>/dev/null || true
PYTHON="${PYTHON:-$ROOT/.venv-rocm/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
TARGET="${RAX_TARGET:-100}"
# place100d: RGB top/side/wrist + top-depth gray (3ch) for ACT depth A/B compare.
DS_ROOT="${RAX_DATASET_ROOT:-./datasets/so101-simstudio-rax-drawer-place100d}"
REPO_ID="${RAX_DATASET_REPO_ID:-alexhegit/so101-simstudio-rax-drawer-place100d}"

echo "=== RAX place collect ==="
echo "MUJOCO_GL=$MUJOCO_GL target=$TARGET root=$DS_ROOT"
exec "$PYTHON" -m labs.rax_drawer.collect_rax_place \
  --target "$TARGET" \
  --root "$DS_ROOT" \
  --repo-id "$REPO_ID" \
  "$@"
