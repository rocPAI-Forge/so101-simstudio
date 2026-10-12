# kitchen_counter

ROSCon teleop workcell. The environment is adapted from Text2MuJoCo showcase
[`01-button-cube-box`](https://github.com/ShawnJoeng/Text2Mujoco) (MIT): steel
workbench, green start button, translucent-front place box. SO-101 and the three
calibrated cameras (`front`, `top`, `wrist`) replace their industrial arm.

Text2MuJoCo has no kitchen or food meshes — only primitive furniture. Scan2Sim
ginger/garlic were dropped because decimation cracked the surface and there is
no equivalent asset in Text2MuJoCo.

| Item | Source | Notes |
|------|--------|-------|
| Workbench, button, open box | Text2MuJoCo `01-button-cube-box` | Reseated so the top face is **z = 0** |
| Green cube | Lab `simple_pick` | Joint name `block` |
| Red cube, blue cylinder | Text2MuJoCo primitive recipe | Sized for the SO-101 jaw (~2.5–3 cm) |
| Cameras (recorded) | `simple_pick` | `front` / `top` / `wrist` — these are in `robot.camera_names` |
| `overview` (+ extras) | Scene MJCF | **Teleop GLFW only** — not recorded unless you add the name to `camera_names` |

The MuJoCo window may show a fourth **overview** pane for operator situational
awareness. Recording follows `camera_names` (default three). To record overview
or train on a subset (e.g. wrist + overview), see
[../README.md#cameras-record-vs-teleop-vs-train](../README.md#cameras-record-vs-teleop-vs-train).

## Run (keyboard teleop)

```bash
make smoke-keyboard-teleop-kitchen
```

```bash
source .venv-rocm/bin/activate
export MUJOCO_GL=glfw
./scripts/smoke/keyboard_teleop_kitchen.sh
# equivalent:
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
  --config configs/so101_mujoco_keyboard_teleop_kitchen.yaml
```

## Preview MJCF only

```bash
.venv-rocm/bin/python -m mujoco.viewer --mjcf=SO101/scenes/kitchen_counter/scene.xml
```
