# Live demos

On-site interactive demos. These are **not** CI tests and stay on the
`demo/drawer-teleop` branch until deliberately promoted.

## smart_drawer teleop

Operator drives the SO-101 in the Text2MuJoCo `smart_drawer` workcell.
**Display is the MuJoCo GLFW window** (same path as
`make smoke-keyboard-teleop-smart-drawer`). Rerun is not used: teleop does not
wire `--view_mode`, and the operator should look at the same low-latency window
they are controlling. For the audience, screen-share that GLFW window.

```bash
# after make rocm-sync (and make joycon-sync for Joy-Con)
./scripts/demo/drawer_teleop.cmd              # keyboard
./scripts/demo/drawer_teleop.cmd joycon       # right Joy-Con (frozen reference)
./scripts/demo/drawer_teleop.cmd joycon left  # left twin (same stick/IK; d-pad wrists)
# or
make demo-drawer
make demo-drawer DEVICE=joycon SIDE=right
make demo-drawer DEVICE=joycon SIDE=left
```

**Talk track:** grasp the lower handle → close gripper → pull toward the arm
(+Y) to open (~8 cm). Green button is decorative.

**Scene reset (no restart):** keyboard **Backspace**, Joy-Con **Plus** — arm
home, cube to episode-0 pose, drawer closed. Debounced ~0.4 s.

**Keyboard from the Desktop icon:** the launcher sets `SO101_PREFER_EVDEV=1`
so keys work without focusing the terminal or MuJoCo window (user must be in
the `input` group). If the arm still ignores keys, check the terminal log for
`Using evdev keyboard listener`; `pynput` alone often fails on Wayland.

**Joy-Con `HIDException unable to open device`:** Bluetooth can show Connected
while `/dev/hidraw*` stays `root:root 600`. Install the udev rule (also printed
by `make joycon-sync`) or temporarily `sudo chmod 666` that hidraw node, then
relaunch.

### GLFW panes (smart_drawer)

| Pane | Camera | Role |
|------|--------|------|
| Top-left | `top` | Overhead |
| Top-right | `front` | Near handle |
| Bottom-left | `wrist` | Gripper |
| Bottom-right | `overview` | Operator view: behind the arm, rear-upper looking at cube + arm + drawer |

### Joy-Con → motors (drawer demo)

SO-101 has six actuated joints. Joy-Con does **not** expose one stick per joint;
Cartesian velocity IK drives the first three, tilt drives the wrists, ZR the gripper.

Frozen reference = **right** Joy-Con. Left uses the same stick/IK; only wrist
buttons differ (d-pad).

| Motor | Right Joy-Con | Left Joy-Con |
|-------|---------------|--------------|
| `shoulder_pan` | Stick L/R ≈ swing | same |
| `shoulder_lift` | Stick F/B + R / stick-press (weighted IK) | same + **L** / stick-press |
| `elbow_flex` | Same reach channel (IK prefers elbow) | same |
| `wrist_flex` | Hold **B** / **X** | Hold **Up** / **Down** |
| `wrist_roll` | Hold **Y** / **A** | Hold **Left** / **Right** |
| `gripper` | **ZR** toggle | **ZL** toggle |

One stick axis cannot independently set lift and elbow; cylindrical IK maps
forward/back to **reach** and left/right to **swing** (mostly pan).

| Device | Config |
|--------|--------|
| Keyboard | `configs/so101_mujoco_keyboard_teleop_smart_drawer.yaml` |
| Joy-Con right (reference) | `configs/so101_mujoco_joycon_teleop_smart_drawer.yaml` |
| Joy-Con left | `configs/so101_mujoco_joycon_teleop_smart_drawer_left.yaml` |
