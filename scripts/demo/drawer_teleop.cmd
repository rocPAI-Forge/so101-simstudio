#!/bin/bash
# Live smart_drawer teleop demo: MuJoCo GLFW window + keyboard or Joy-Con.
# Usage:
#   ./scripts/demo/drawer_teleop.cmd              # keyboard (default)
#   ./scripts/demo/drawer_teleop.cmd keyboard
#   ./scripts/demo/drawer_teleop.cmd joycon
#   ./scripts/demo/drawer_teleop.cmd joycon left
# Env:
#   SIDE=left|right   Joy-Con side (default right; overridden by 2nd arg)

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
# Reuse smoke venv resolution.
# shellcheck source=../smoke/_common.sh
source "$REPO_ROOT/scripts/smoke/_common.sh"

DEVICE="${1:-keyboard}"
SIDE_ARG="${2:-${SIDE:-right}}"

case "$DEVICE" in
  keyboard|kb|key)
    DEVICE=keyboard
    CONFIG=configs/so101_mujoco_keyboard_teleop_smart_drawer.yaml
    ;;
  joycon|jc|joy)
    DEVICE=joycon
    case "$SIDE_ARG" in
      right)
        CONFIG=configs/so101_mujoco_joycon_teleop_smart_drawer.yaml
        ;;
      left)
        # Same frozen stick/IK as right; d-pad wrists (no A/B/X/Y on left).
        CONFIG=configs/so101_mujoco_joycon_teleop_smart_drawer_left.yaml
        ;;
      *)
        echo "Joy-Con side must be left or right (got: $SIDE_ARG)" >&2
        exit 1
        ;;
    esac
    ;;
  -h|--help|help)
    cat <<'EOF'
Live smart_drawer teleop demo (MuJoCo GLFW; not Rerun).

  ./scripts/demo/drawer_teleop.cmd [keyboard|joycon] [left|right]

Defaults: keyboard, Joy-Con side=right.
Requires .venv-rocm (make rocm-sync). Joy-Con also needs make joycon-sync + Bluetooth pair.
EOF
    exit 0
    ;;
  *)
    echo "Unknown device '$DEVICE'. Use keyboard or joycon." >&2
    exit 1
    ;;
esac

# Desktop / Wayland: pynput often starts but captures nothing unless the
# terminal stays focused. Prefer focus-independent evdev (same as record).
if [[ "$DEVICE" == "keyboard" ]]; then
  export SO101_PREFER_EVDEV=1
fi

echo "=== smart_drawer live demo ==="
echo "Display : MuJoCo GLFW (operator view). Screen-share this window for the audience."
echo "Scene   : SO101/scenes/smart_drawer"
echo "Device  : $DEVICE"
if [[ "$DEVICE" == "joycon" ]]; then
  echo "Joy-Con : $SIDE_ARG"
fi
echo ""
echo "Task: grasp the lower drawer handle, close the gripper, pull toward the arm (+Y) to open (~8 cm)."
echo "Stuck / reset: Ctrl+C, then re-run this script (loads home)."
echo ""

if [[ "$DEVICE" == "keyboard" ]]; then
  cat <<'EOF'
Keyboard (evdev — works without focusing this terminal or the MuJoCo window):
  W/S  pull open (+Y) / push close (-Y)
  A/D  left / right (X)
  Z/X  up / down (Z)
  I/K  wrist flex
  [/]  wrist rotate
  C    close gripper (hold while pulling)
  O    open gripper
EOF
else
  GRIP_BTN="ZR"
  UP_BTN="R"
  WRIST_FLEX="X / B"
  WRIST_ROLL="Y / A"
  if [[ "$SIDE_ARG" == "left" ]]; then
    GRIP_BTN="ZL"
    UP_BTN="L"
    WRIST_FLEX="D-pad Up / Down"
    WRIST_ROLL="D-pad Left / Right"
  fi
  cat <<EOF
Joy-Con ($SIDE_ARG) — cylindrical stick (stand behind the arm; same scheme as right):
  Stick forward/back  extend / retract arm (IK → lift + elbow)
  Stick left/right    base swing ≈ shoulder_pan
  ${UP_BTN} / stick-press   up / down (IK height)
  ${WRIST_FLEX}       wrist_flex ± (hold)
  ${WRIST_ROLL}       wrist_roll ± (hold)
  ${GRIP_BTN}                 toggle gripper close/open
EOF
  echo ""
  echo "If connect fails: make joycon-sync, pair over Bluetooth, and fix hidraw perms"
  echo "  (sudo cp third_party/joycon-robotics/udev/99-nitendo.rules /etc/udev/rules.d/99-nintendo-joycon.rules)."
  # Preflight: Bluetooth can be Connected while /dev/hidraw* stays root-only.
  if ! "$PYTHON" -c "
import hid, sys
found = False
for d in hid.enumerate(0x057e):
    found = True
    path = d.get('path')
    label = path.decode() if isinstance(path, bytes) else path
    try:
        h = hid.Device(path=path)
        h.close()
        print('hidraw_ok', label)
        sys.exit(0)
    except Exception as e:
        print('hidraw_blocked', label, e, file=sys.stderr)
        sys.exit(2)
if not found:
    print('No Nintendo HID device. Pair/connect the Joy-Con over Bluetooth first.', file=sys.stderr)
    sys.exit(3)
" 2>&1; then
    echo ""
    echo "Joy-Con HID open failed. Typical fix on Ubuntu:"
    echo "  sudo chmod 666 /dev/hidraw10   # use the node from the error / ls -l /dev/hidraw*"
    echo "  # permanent:"
    echo "  sudo cp third_party/joycon-robotics/udev/99-nitendo.rules /etc/udev/rules.d/99-nintendo-joycon.rules"
    echo "  sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=hidraw"
    exit 1
  fi
fi

echo ""
echo "Press Ctrl+C to quit."
echo ""

# Side-specific Joy-Con YAMLs already set teleop.side; do not override.
exec "$PYTHON" -m simstudio.scripts.teleoperate --config "$CONFIG"