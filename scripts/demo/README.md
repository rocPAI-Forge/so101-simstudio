# Live demos

On-site interactive demos (not CI). **`smart_drawer` teleop is on `main`**
and pinned at tag `release-v0.1.7`. Full Joy-Con Bluetooth / udev / button map:
[docs/joycon-teleop.md](../../docs/joycon-teleop.md).

## smart_drawer teleop

Operator drives the SO-101 in the Text2MuJoCo `smart_drawer` workcell.
**Display is the MuJoCo GLFW window** (same path as
`make smoke-keyboard-teleop-smart-drawer`). Rerun is not used: teleop does not
wire `--view_mode`; screen-share the GLFW window for the audience.

```bash
# after make rocm-sync (and make joycon-sync for Joy-Con — see docs/joycon-teleop.md)
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

**Keyboard from a Desktop launcher:** set `SO101_PREFER_EVDEV=1` so keys work
without focusing the terminal or MuJoCo window (user must be in the `input`
group). Confirm the log shows `Using evdev keyboard listener`; plain `pynput`
often fails on Wayland.

### GLFW panes (smart_drawer)

| Pane | Camera | Role |
|------|--------|------|
| Top-left | `top` | Overhead |
| Top-right | `front` | Near handle |
| Bottom-left | `wrist` | Gripper |
| Bottom-right | `overview` | Operator view: behind the arm, looking at cube + arm + drawer |

### Joy-Con → motors (drawer demo)

SO-101 has six actuated joints. Cartesian velocity IK drives the first three;
face buttons (or left d-pad) drive the wrists; ZR/ZL the gripper.

Frozen reference = **right** Joy-Con. Left uses the same stick/IK; only wrist
buttons differ (d-pad). Full tables + Bluetooth: [docs/joycon-teleop.md](../../docs/joycon-teleop.md).

| Motor | Right Joy-Con | Left Joy-Con |
|-------|---------------|--------------|
| `shoulder_pan` | Stick L/R ≈ swing | same |
| `shoulder_lift` | Stick F/B + R / stick-press (weighted IK) | same + **L** / stick-press |
| `elbow_flex` | Same reach channel (IK prefers elbow) | same |
| `wrist_flex` | Hold **B** / **X** | Hold **Up** / **Down** |
| `wrist_roll` | Hold **Y** / **A** | Hold **Left** / **Right** |
| `gripper` | **ZR** toggle | **ZL** toggle |
| scene reset | **Plus** | (right Plus; left has no Plus) |

| Device | Config |
|--------|--------|
| Keyboard | `configs/so101_mujoco_keyboard_teleop_smart_drawer.yaml` |
| Joy-Con right (reference) | `configs/so101_mujoco_joycon_teleop_smart_drawer.yaml` |
| Joy-Con left | `configs/so101_mujoco_joycon_teleop_smart_drawer_left.yaml` |
