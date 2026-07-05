#!/bin/bash
# 安装 joycon-robotics (submodule) 并应用补丁
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMODULE_DIR="$PROJECT_DIR/third_party/joycon-robotics"
PATCH_FILE="$PROJECT_DIR/patches/joycon-robotics.patch"

echo "=== 安装 joycon-robotics ==="

# 安装到项目 venv
cd "$PROJECT_DIR"
uv pip install -e "$SUBMODULE_DIR"

# 应用补丁
echo "=== 应用补丁 ==="
cd "$SUBMODULE_DIR"
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "✓ 补丁已应用"
else
    echo "⚠ 补丁可能已应用或不再适用"
fi

echo "=== 完成 ==="
