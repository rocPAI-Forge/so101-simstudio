---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - molmoact2
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
base_model: lerobot/MolmoAct2-SO100_101-LeRobot
---

# SO-101 Lab01 Pick-and-Place — MolmoAct2

MolmoAct2 policy fine-tuned from [`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) on expert demonstrations collected and validated with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** (MuJoCo sim2sim, leader-arm teleop).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

This Hub revision is the **MI300X** run: batch **32**, **10 000** steps (~320K sample updates, ~15 epochs), VLM LoRA + trainable action expert, final train loss **0.009**, ~30 GB of 192 GB HBM. Wall time ~**13 h 40 m** on AMD Instinct MI300X (DORobot). Checkpoint `010000` / `last`.

Cameras used for this fine-tune (2-view warm start matching the SO100_101-LeRobot base): `camera_top` → `cam0`, `camera_wrist` → `cam1`. Lab01 joints are **radians**; joint frame transform is identity (no SO100 degree offsets).

Closed-loop MuJoCo success rates are **not** published for this checkpoint yet.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

## Quick load

```python
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

policy = MolmoAct2Policy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-molmoact2")
```
