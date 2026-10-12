# smart_drawer

ROSCon teleop workcell adapted from Text2MuJoCo showcase
[`02-smart-drawer`](https://github.com/ShawnJoeng/Text2Mujoco) (MIT).
**Keyboard teleop pull validated** on the local sim host.

Cabinet **front faces +Y** (visible to `front` / `top`). SO-101 is remounted via
`SO101/so101_drawer_mount.xml` at `(0.18, 0.48)`, **facing −Y** — standing in
front of the drawer handle. The drawer opens **+Y** (toward the arm).

**Cameras:** see [../README.md#cameras-record-vs-teleop-vs-train](../README.md#cameras-record-vs-teleop-vs-train).
Default record is `front` / `top` / `wrist`. MJCF `overview` is GLFW teleop-only
unless added to `camera_names`; training can subset (e.g. wrist + overview) with
`--rename_map` without deleting other streams from the dataset.

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
