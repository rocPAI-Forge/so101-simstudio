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
  - scripted-expert
size_categories:
  - 10K<n<100K
---

# SO-101 Lab01 Pick-and-Place Dataset (scripted agent)

Privileged-state scripted expert (cube pose + joint lerp + Jacobian IK) recorded with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** into LeRobot v3.0. Same MuJoCo scene and cube random box as the leader-teleop set; **not** mixed with `alexhegit/so101-simstudio-lab01-pnp`.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent` |
| Task | Pick up the cube and place it in the box. |
| Robot | `so101_mujoco` (6-D joint position actions) |
| Cameras | `camera_front`, `camera_top`, `camera_wrist` (640×480) |
| Cube random | x [0.26, 0.34], y [0.165, 0.235], z 0.0125, yaw ±45° |

Teleop counterpart: [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp).

**Closed-loop (do not treat as a 64% ACT set):** ACT 80K on this dataset, full-range eval on .43 GLFW, 30 s/ep, `n_action_steps=50`: **9/50 (18%)**. The expert **locks wrist** at close (`wrist_roll=0.17` on all 100 episodes) and does not map cube yaw to jaw alignment. After-action: `labs/lab01_pnp_agent/agent_act_80k.md`. A follow-up yaw-aware collector will use a **new** repo id.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |
| Collector | `labs/lab01_pnp/auto_collect_poc.py` |

## Quick load

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset("alexhegit/so101-simstudio-lab01-pnp-agent")
sample = dataset[0]
```
