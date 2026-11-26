#!/bin/bash
# Install joycon-robotics (submodule) and apply project patch into .venv-rocm.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMODULE_DIR="$PROJECT_DIR/third_party/joycon-robotics"
PATCH_FILE="$PROJECT_DIR/patches/joycon-robotics.patch"
ROCM_VENV="$PROJECT_DIR/.venv-rocm"
ROCM_PY="$ROCM_VENV/bin/python"

if [[ ! -x "$ROCM_PY" ]]; then
    echo "No .venv-rocm found. Run: make rocm-sync" >&2
    exit 1
fi

echo "=== Installing joycon-robotics into .venv-rocm ==="

cd "$PROJECT_DIR"
# Match setup-rocm.sh: pin the target interpreter so bare `uv pip` does not
# require an already-activated virtualenv.
uv pip install --python "$ROCM_PY" -e "$SUBMODULE_DIR"

echo "=== Applying patch ==="
cd "$SUBMODULE_DIR"
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "Patch applied"
elif git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "Patch already applied"
else
    echo "Error: patch does not apply; reset submodule to the pinned upstream commit and retry:" >&2
    echo "  git -C third_party/joycon-robotics checkout -- . && make joycon-sync" >&2
    exit 1
fi

# Guard against a silent unpatched tree: upstream device.py returns a bare
# tuple for non-9c:54 serials, which then crashes get_ids_of_type with
# TypeError: 'int' object is not subscriptable.
if grep -q "return (0x057A, product_id, serial)" joyconrobotics/device.py 2>/dev/null; then
    echo "Error: joycon-robotics serial compatibility patch is missing in device.py" >&2
    echo "Reset the submodule and re-run: make joycon-sync" >&2
    exit 1
fi
echo "Patch verified (device.py serial early-return removed)"

# Bluetooth Joy-Con nodes are /dev/hidraw* owned root:root 600 by default.
# Without this rule, joyconrobotics fails with: HIDException unable to open device
UDEV_SRC="$SUBMODULE_DIR/udev/99-nitendo.rules"
UDEV_DST="/etc/udev/rules.d/99-nintendo-joycon.rules"
echo "=== Installing Joy-Con hidraw udev rule ==="
if [[ -f "$UDEV_DST" ]] && cmp -s "$UDEV_SRC" "$UDEV_DST"; then
    echo "udev rule already installed: $UDEV_DST"
elif [[ -r "$UDEV_SRC" ]]; then
    if sudo -n true 2>/dev/null; then
        sudo cp "$UDEV_SRC" "$UDEV_DST"
        sudo udevadm control --reload-rules
        sudo udevadm trigger --subsystem-match=hidraw
        echo "Installed $UDEV_DST"
    else
        echo "Need sudo to allow non-root access to Joy-Con hidraw. Run:"
        echo "  sudo cp $UDEV_SRC $UDEV_DST"
        echo "  sudo udevadm control --reload-rules"
        echo "  sudo udevadm trigger --subsystem-match=hidraw"
        echo "Temporary (until unplug/reconnect):"
        echo "  sudo chmod 666 /dev/hidraw*   # or only the Joy-Con node"
    fi
else
    echo "Warning: missing $UDEV_SRC" >&2
fi

echo "=== Done ==="
