---
license: apache-2.0
task_categories:
  - robotics
tags:
  - lerobot
  - so101
  - simstudio
  - mujoco
  - pick-and-place
  - teleoperation
size_categories:
  - 10K<n<100K
---

# SO-101 Lab01 Pick-and-Place Dataset

Expert demonstrations for MuJoCo sim2sim pick-and-place, recorded with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** (real leader arm → MuJoCo follower, LeRobot v3.0).

| | |
|---|---|
| Episodes | 50 @ 20 Hz |
| Task | Pick up the cube and place it in the box. |
| Robot | `so101_mujoco` (6-D joint position actions) |
| Cameras | `camera_front`, `camera_top`, `camera_wrist` (640×480) |

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 (record → validate → train → eval) | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

For recording setup, validation, replay, and training commands, see **Lab 01** in the repo above.

## Fine-tuned policy

ACT checkpoint trained on this dataset: [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act)

## Quick load

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset("alexhegit/so101-simstudio-lab01-pnp")
sample = dataset[0]
```
