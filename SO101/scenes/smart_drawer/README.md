# smart_drawer

ROSCon teleop workcell adapted from Text2MuJoCo showcase
[`02-smart-drawer`](https://github.com/ShawnJoeng/Text2Mujoco) (MIT).
**Keyboard teleop pull validated** on the local sim host.

The cabinet, green button, and lower drawer are one mocap body, `drawer_rig`.
The identity pose keeps their nominal world positions. SO-101 is remounted via
`SO101/so101_drawer_mount.xml` at `(0.18, 0.48)`, **facing −Y** — standing in
front of the drawer handle. The drawer opens **+Y** (toward the arm).

**Cameras:** the control set is `top`, `side`, and `wrist`. `wrist` stays on
the gripper. `top` is a fixed overhead Gemini 336L depth camera at
`(0.18, 0.22, 0.75)`, looking straight down, 1280×800, 65° vertical field
(about 91° horizontal). `side` is a fixed color camera at
`(-0.8255, 0.2800, 0.3970)`, 640×480, 45° vertical field, looking across the
drawer, the cube, and the arm from 1.15 m away. `front` and `overview` remain
for human viewing. See
[../README.md#cameras-record-vs-teleop-vs-train](../README.md#cameras-record-vs-teleop-vs-train).

## Run (keyboard teleop)

```bash
make smoke-keyboard-teleop-smart-drawer
```

Task: grasp the lower handle, hold **C**, hold **W** to pull (+Y, open).
Drawer slides up to 8 cm. Green unlock button is static (no lock logic).
Red/yellow tray tools are visual-only (no collision).

```bash
.venv-rocm/bin/python -m mujoco.viewer --mjcf=SO101/scenes/smart_drawer/scene.xml
```
