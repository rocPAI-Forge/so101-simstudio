---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - vla-jepa
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
base_model: lerobot/VLA-JEPA-LIBERO
---

# SO-101 Lab01 Pick-and-Place — VLA-JEPA

VLA-JEPA policy fine-tuned from [`lerobot/VLA-JEPA-LIBERO`](https://huggingface.co/lerobot/VLA-JEPA-LIBERO) on expert demonstrations collected and validated with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** (MuJoCo sim2sim, leader-arm teleop).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

This Hub revision is the **MI300X** run: batch **16**, **20 000** steps (resume 10K→20K), world-model co-training on, `chunk_size` / `n_action_steps` **7**, final train loss **~0.115**. Wall time ~**3.5 h** (first 10K) + **~4.2 h** (10K→20K) on AMD Instinct MI300X (DORobot). Checkpoint `020000` / `last`.

Cameras: `camera_top` → `image`, `camera_wrist` → `image2`. Action is **6-D** joint position. Proprio is Lab 01’s **15-D** `observation.state` (6 pos + 6 vel + 3 EE); the LIBERO base config still lists 8-D `input_features` — load with `state_dim=15` (see SimStudio `eval.py`).

Closed-loop MuJoCo (10K, fixed spawn, `reset_arm: home`, sync, GLFW): **0/10**. The 20K weights on this Hub page have not been eval’d in sim yet.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

## Quick load

```python
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

policy = VLAJEPAPolicy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-vla-jepa")
```
