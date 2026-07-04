---
tags: [glfw, mujoco, window, rendering]
platform: [linux]
update-check: 2026-07
---

# GLFW Window Visibility (MuJoCo)

MuJoCo uses GLFW for windowed rendering. On Ubuntu 24.04 / GNOME, certain GLFW hints make the window invisible.

## Problem

Setting these GLFW hints to `FALSE` causes the window to not appear:

```python
# DO NOT set these to FALSE on Ubuntu/GNOME
os.environ["GLFW_FOCUSED"] = "FALSE"
os.environ["GLFW_FOCUS_ON_SHOW"] = "FALSE"
```

## Solution

Leave GLFW hints at defaults (do not set them). Or set to `TRUE`.

For headless/CI runs, disable the GLFW window entirely in MuJoCo config:

```yaml
robot:
  render_window: false
```

## Affected environments

- Ubuntu 24.04 + GNOME (Wayland and X11)
- Likely affects other GNOME-based desktops

## Not affected

- Headless servers (no display)
- KDE / other desktop environments (not confirmed)
- macOS (uses native windowing)

## Verify

```bash
python -m mujoco.viewer --mjcf=SO101/pick_scene.xml
```

If window appears, GLFW is working. If not, check environment variables.
