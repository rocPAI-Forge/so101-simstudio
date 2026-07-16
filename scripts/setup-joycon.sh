#!/bin/bash
# Install joycon-robotics (submodule) and apply project patch.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMODULE_DIR="$PROJECT_DIR/third_party/joycon-robotics"
PATCH_FILE="$PROJECT_DIR/patches/joycon-robotics.patch"

echo "=== Installing joycon-robotics ==="

cd "$PROJECT_DIR"
uv pip install -e "$SUBMODULE_DIR"

echo "=== Applying patch ==="
cd "$SUBMODULE_DIR"
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "Patch applied"
elif git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "Patch already applied"
else
    echo "Error: patch does not apply; reset submodule to the pinned upstream commit and retry" >&2
    exit 1
fi

echo "=== Done ==="
