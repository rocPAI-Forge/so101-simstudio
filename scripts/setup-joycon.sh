#!/bin/bash
# 安装 joycon-robotics 并应用补丁
set -e

JOYCON_REPO="/home/amd/Repo/joycon-robotics"
PATCH_DIR="$(dirname "$0")/patches"
VENV_DIR="$(dirname "$0")/.venv"

echo "=== 安装 joycon-robotics ==="

# 安装到项目 venv
cd "$(dirname "$0")"
uv pip install -e "$JOYCON_REPO"

# 应用补丁
echo "=== 应用补丁 ==="
cd "$JOYCON_REPO"
if git apply --check "$PATCH_DIR/joycon-robotics.patch" 2>/dev/null; then
    git apply "$PATCH_DIR/joycon-robotics.patch"
    echo "✓ 补丁已应用"
else
    echo "⚠ 补丁可能已应用或不再适用"
fi

echo "=== 完成 ==="
