#!/bin/bash
# Fetch LeRobot PR #3967 side tree and apply the Lab 01 inverse-slot patch.
# Required before eval_lingbot_vla_v2_home20.cmd (policy code is not in pinned lerobot/).
#
#   ./labs/lab01_pnp/setup_lingbot_vla_v2_side_tree.cmd
#   LAB01_V2_SIDE_ROOT=/path/to/tree ./labs/lab01_pnp/setup_lingbot_vla_v2_side_tree.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_LAB01_DIR/../.." && pwd)"
SIDE_ROOT="${LAB01_V2_SIDE_ROOT:-/tmp/lerobot_vla_v2_src}"
PIN_SHA="${LAB01_V2_SIDE_SHA:-543f4207}"
PATCH="$REPO_ROOT/patches/lerobot-vla-v2-inverse-slot-offset.patch"

echo "=== LingBot-VLA 2.0 side tree ==="
echo "Root: $SIDE_ROOT"
echo "Pin:  $PIN_SHA (PR #3967)"
echo "Patch: $PATCH"

if [[ ! -d "$SIDE_ROOT/.git" ]]; then
    git clone https://github.com/huggingface/lerobot.git "$SIDE_ROOT"
fi

git -C "$SIDE_ROOT" fetch --depth 1 origin "$PIN_SHA" 2>/dev/null \
    || git -C "$SIDE_ROOT" fetch origin pull/3967/head:pr-3967
git -C "$SIDE_ROOT" checkout --force "$PIN_SHA" 2>/dev/null \
    || git -C "$SIDE_ROOT" checkout --force pr-3967

# Idempotent apply: reverse dry-run then apply if needed.
if git -C "$SIDE_ROOT" apply --check --reverse "$PATCH" 2>/dev/null; then
    echo "Patch already applied."
elif git -C "$SIDE_ROOT" apply --check "$PATCH" 2>/dev/null; then
    git -C "$SIDE_ROOT" apply "$PATCH"
    echo "Patch applied."
else
    echo "Patch neither clean nor already applied; inspect $SIDE_ROOT" >&2
    git -C "$SIDE_ROOT" apply --check "$PATCH" || true
    exit 1
fi

echo ""
echo "PYTHONPATH for eval:"
echo "  export PYTHONPATH=$SIDE_ROOT/src:\${PYTHONPATH:-}"
echo "Done."
