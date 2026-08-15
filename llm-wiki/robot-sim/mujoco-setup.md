---
tags: [mujoco, simulation, rendering]
platform: [linux, macos]
update-check: 2026-08
---

# MuJoCo Setup

MuJoCo is a physics engine for robot simulation. Used for sim-to-real workflows and teleoperation testing.

## Install

```bash
uv pip install "mujoco>=3.0.0,<4.0.0"
```

## Preview scene

```bash
python -m mujoco.viewer --mjcf=SO101/scenes/simple_pick/scene.xml
```

## Headless rendering

For recording / eval without a visible window, set in robot config YAML:

```yaml
robot:
  render_window: false
```

Choose the OpenGL backend with `MUJOCO_GL` (or Lab 01 `LAB01_MUJOCO_GL`):

| Backend | Use when |
|---------|----------|
| `egl` | Headless GPU (preferred for batch eval on AMD/NVIDIA workstations) |
| `glfw` | Interactive MuJoCo window (needs `DISPLAY`) |
| `osmesa` | CPU software GL (compute nodes without EGL; very slow) |

### Diagnose camera images per backend

`scripts/render_headless_cameras.py` loads `SO101/scenes/simple_pick/scene.xml`, poses the arm at home with a sample cube, and writes one PNG per camera (`front`, `top`, `wrist`). Use it to verify that a given `MUJOCO_GL` backend produces sane images (not black / garbage) before long eval runs.

```bash
# from repo root, with .venv-rocm active
python scripts/render_headless_cameras.py egl /tmp/cam_egl
python scripts/render_headless_cameras.py osmesa /tmp/cam_osmesa
# optional: glfw (needs a display)
python scripts/render_headless_cameras.py glfw /tmp/cam_glfw
```

Outputs look like `/tmp/cam_egl/egl_front.png`, `egl_top.png`, `egl_wrist.png`. Compare mean pixel values or open the PNGs side by side across backends.

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
| Slow rendering on CPU | Normal for `osmesa`; prefer `egl` on GPU |
| Black / empty camera frames headless | Run `scripts/render_headless_cameras.py` for the backend you use |
| `mujoco.viewer` crashes | Check GLFW, try `render_window: false` |

## When NOT to use MuJoCo

- Need contact-rich manipulation with deformable objects → consider Isaac Sim
- Need photorealistic rendering → consider NVIDIA Omniverse
- Simple kinematic testing → PyBullet may be lighter
