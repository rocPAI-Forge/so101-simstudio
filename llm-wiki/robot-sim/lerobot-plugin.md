---
tags: [lerobot, plugin, registration, teleoperator, robot]
platform: [linux, macos]
update-check: 2026-07
---

# LeRobot Plugin Registration

LeRobot uses `ChoiceRegistry` for robot/teleoperator configs. Plugins are discovered via entry points in installed distributions.

## Problem

When a plugin package is installed as editable in the same distribution as the main project, LeRobot's distribution-based discovery does NOT pick it up automatically.

## Solution

Import config classes explicitly in the wrapper script BEFORE importing LeRobot's entry points:

```python
# 1. Import project plugins (registers them in ChoiceRegistry)
from my_project.robots.my_robot import MyRobotConfig
from my_project.teleoperators.my_teleop import MyTeleopConfig

# 2. Now import LeRobot's record/replay
from lerobot.scripts.lerobot_record import record
```

## Why this works

Importing a config class triggers its `register()` classmethod (via `__init_subclass__`), which adds it to the global registry. LeRobot's draccus parser then finds it.

## Naming convention

- Robot plugins: `lerobot_robot_<name>`
- Teleoperator plugins: `lerobot_teleoperator_<name>`

## Avoid name collisions

If your plugin name conflicts with a LeRobot built-in (e.g., `so101_leader` vs LeRobot's `so101_leader`), rename your plugin. The built-in takes priority in the registry.

## Verify

```bash
python -c "from my_project.robots.my_robot import MyRobotConfig; print('OK')"
```
