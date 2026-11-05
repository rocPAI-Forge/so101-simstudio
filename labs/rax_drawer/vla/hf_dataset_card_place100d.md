---
license: apache-2.0
task_categories:
  - robotics
tags:
  - robotics
  - lerobot
  - so101
  - simstudio
  - mujoco
  - drawer
  - rgb-d
---

# SO-101 RAX drawer place100d

~100 successful open → place cube → close demos on the SO-101 `smart_drawer` MuJoCo scene, collected with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** for an ACT RGB vs RGB-D ablation.

**Cameras:** `camera_top`, `camera_side`, `camera_wrist` (RGB) plus `camera_top_depth` (metric top depth as 3-channel gray for ResNet/ACT).

**Actions / state:** absolute joint `.pos` (6-D) for imitation; recorded observation state may be richer — train ACT with 6-D joint positions.

## Companion models

| Policy | Hub |
|--------|-----|
| ACT RGB | [`alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d) |
| ACT RGB-D | [`alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d) |

Ablation write-up: [`labs/rax_drawer/vla/DEPTH_ABLATION.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/rax_drawer_vla/labs/rax_drawer/vla/DEPTH_ABLATION.md)

## Load

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset("alexhegit/so101-simstudio-rax-drawer-place100d")
```
