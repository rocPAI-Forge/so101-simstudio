# Joy-Con teleop guide

Canonical setup and control map for Nintendo Switch Joy-Con on Ubuntu 24.04
(ROCm). Frozen feel reference: `configs/so101_mujoco_joycon_teleop_smart_drawer.yaml`
(`release-v0.1.7`). Live drawer demo: [scripts/demo/README.md](../scripts/demo/README.md).

## 1. Install

```bash
make rocm-sync          # once (or after dep refresh)
make joycon-sync        # every fresh clone / submodule reset / Joy-Con connect fail
```

`make joycon-sync` (`scripts/setup-joycon.sh`) will:

1. Editable-install `third_party/joycon-robotics` into `.venv-rocm`.
2. Apply `patches/joycon-robotics.patch` (serial compatibility + English connect
   messages). **Do not** commit project edits inside the submodule.
3. Try to install a hidraw **udev** rule so non-root can open `/dev/hidraw*`.

System packages often required (once per machine):

```bash
sudo apt-get install -y libhidapi-hidraw0 libhidapi-libusb0
# optional: bluetoothctl lives in bluez
sudo apt-get install -y bluez
```

## 2. Bluetooth pair / connect

Pair **one** Joy-Con at a time (right or left). Sync button is the small round
button on the rail (inside the long edge).

### GNOME Settings (usual path)

1. Settings → Bluetooth → On.
2. Hold the Joy-Con **Sync** button until the player LEDs chase.
3. Select **Joy-Con (R)** or **Joy-Con (L)** → Connect.
4. Status should show **Connected** (not merely Paired).

### CLI (`bluetoothctl`)

```bash
bluetoothctl
power on
agent on
default-agent
scan on
# wait for: Joy-Con (R) or Joy-Con (L)  XX:XX:XX:XX:XX:XX
pair XX:XX:XX:XX:XX:XX
trust XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
scan off
quit
```

Re-pair tip: if connect flaps, `bluetoothctl remove <MAC>` then pair again, or
toggle Bluetooth off/on and re-hold Sync.

## 3. hidraw permissions (most common failure)

Bluetooth can show **Connected** while Python still fails with:

```text
HIDException: unable to open device
```

Default nodes are often `root:root 600`. Fix permanently:

```bash
# same files as make joycon-sync prints when sudo is needed
sudo cp third_party/joycon-robotics/udev/99-nitendo.rules \
        /etc/udev/rules.d/99-nintendo-joycon.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hidraw
```

Then reconnect the Joy-Con (or reboot). Temporary:

```bash
sudo chmod 666 /dev/hidraw*    # or only the Joy-Con node
```

Sanity check:

```bash
ls -l /dev/hidraw*
# after udev + reconnect, group/mode should allow your user (often plugdev / 660)
```

## 4. Control scheme (current default)

All project Joy-Con configs use this scheme unless a YAML explicitly sets
`wrist_control: imu`. Stick is cylindrical reach/swing; wrists are **buttons**
(not IMU tilt).

### Right Joy-Con (frozen reference)

| Input | Action |
|-------|--------|
| Stick forward / back | Reach out / retract |
| Stick left / right | Base swing (`shoulder_pan`) |
| **R** | Up (+Z) |
| Stick press | Down (−Z) |
| Hold **B** / **X** | Wrist flex ± |
| Hold **Y** / **A** | Wrist roll ± |
| **ZR** | Toggle gripper (`gripper_toggle: true`) |
| **+** (Plus) | Teleop scene hot-reset (arm home, cube, drawer closed) |

### Left Joy-Con

Same stick / Z / gripper (**ZL**). Wrists use the **d-pad**:

| Input | Action |
|-------|--------|
| Hold **Up** / **Down** | Wrist flex ± |
| Hold **Left** / **Right** | Wrist roll ± |

### Recording (one-handed) — remapped off wrist keys

Keyboard **N/R/Q** (or arrows) always works via evdev.

| | Right | Left |
|--|-------|------|
| Save & next | **Home** | **Capture** |
| Re-record | **Capture** | **Home** |
| Stop session | **+** | **Minus** |

### Scene-specific stick sign

| Scene | Typical `invert_y` | Why |
|-------|--------------------|-----|
| `simple_pick` | `false` | Arm faces +Y |
| `smart_drawer` | `true` | Arm faces −Y; stick left = operator left |

Other knobs: `invert_x` / `invert_z`, `wrist_button_rate`, `ik_joint_weights`
(robot), `translation_scale` / `z_scale`.

## 5. Launch commands

```bash
source .venv-rocm/bin/activate   # optional; cmds below use .venv-rocm/bin/python

# Live teleop — simple_pick
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
  --config configs/so101_mujoco_joycon_teleop.yaml

# Live teleop — smart_drawer demo (preferred for demos)
./scripts/demo/drawer_teleop.cmd joycon          # right
./scripts/demo/drawer_teleop.cmd joycon left
# or: make demo-drawer DEVICE=joycon SIDE=right

# Record
.venv-rocm/bin/python -m simstudio.scripts.record \
  --config configs/so101_mujoco_joycon.yaml --view_mode mujoco
# left: configs/so101_mujoco_joycon_left.yaml
# pick template: configs/so101_mujoco_pick_joycon.yaml

# Smoke
make smoke-joycon-record SIDE=right VIEW_MODE=mujoco
```

| Config | Role |
|--------|------|
| `configs/so101_mujoco_joycon_teleop_smart_drawer.yaml` | Right drawer teleop (**reference**) |
| `configs/so101_mujoco_joycon_teleop_smart_drawer_left.yaml` | Left drawer teleop |
| `configs/so101_mujoco_joycon_teleop.yaml` | Right simple_pick teleop |
| `configs/so101_mujoco_joycon.yaml` / `_left.yaml` | Record |
| `configs/so101_mujoco_pick_joycon.yaml` | Pick-and-place record template |

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `HIDException unable to open device` | udev rule (§3); reconnect Joy-Con |
| Bluetooth Connected but no hidraw change | Re-hold Sync; `remove` + re-pair |
| `TypeError: 'int' object is not subscriptable` on connect | Patch missing → `git -C third_party/joycon-robotics checkout -- . && make joycon-sync` |
| Import / shared-lib errors mentioning hidapi | `sudo apt-get install -y libhidapi-hidraw0` |
| Stick axes feel mirrored | Flip `invert_x` / `invert_y` in YAML for that scene |
| Wrist moves while saving episodes | Recording keys collide with wrists — use Home/Capture (right) or Capture/Home (left) |
| Debug stick values | `SO101_JOYCON_DEBUG=1` before launch |

## 7. Pinning a known-good tree

```bash
git fetch --tags
git checkout release-v0.1.7
git submodule update --init --recursive
make rocm-sync && make joycon-sync
```

That tag freezes the drawer demo + button-wrist scheme even if `main` moves on.
