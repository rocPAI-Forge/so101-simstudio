---
tags: [mujoco, simulation, rendering]
platform: [linux, macos]
update-check: 2026-07
---

# MuJoCo Setup

MuJoCo is a physics engine for robot simulation. Used for sim-to-real workflows and teleoperation testing.

## Install

```bash
uv pip install "mujoco>=3.0.0,<4.0.0"
```

## Preview scene

```bash
python -m mujoco.viewer --mjcf=SO101/pick_scene.xml
```

## Headless rendering

For recording without visible window:

```yaml
# In robot config YAML
robot:
  render_window: false
```

## Python API basics

```python
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

## Common issues

| Issue | Fix |
|-------|-----|
| Window invisible on Ubuntu | See `platform-quirks/glfw-window.md` |
| Slow rendering on CPU | Normal; use GPU machine for real-time |
| `mujoco.viewer` crashes | Check GLFW, try `render_window: false` |

## When NOT to use MuJoCo

- Need contact-rich manipulation with deformable objects → consider Isaac Sim
- Need photorealistic rendering → consider NVIDIA Omniverse
- Simple kinematic testing → PyBullet may be lighter
