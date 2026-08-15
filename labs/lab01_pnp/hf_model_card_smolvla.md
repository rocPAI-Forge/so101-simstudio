---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - smolvla
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
base_model: lerobot/smolvla_base
---

# SO-101 Lab01 Pick-and-Place — SmolVLA

SmolVLA policy fine-tuned from [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) on expert demonstrations collected and validated with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** (MuJoCo sim2sim, leader-arm teleop).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

This Hub revision is the **MI300X** run: batch **64**, **50 000** steps (~3.2M sample updates), final train loss **0.018**, ~26 GB of 192 GB HBM. Wall time ~**7h 45m** on AMD Instinct MI300X (DORobot).

A shorter Strix Halo / 8060S iGPU schedule (`batch_size=4`, 7500 steps) is documented in Lab 01; it is **not** this checkpoint.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough (record → train → eval) | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

Camera keys in the dataset (`camera_top` / `camera_front` / `camera_wrist`) map to SmolVLA `camera1` / `camera2` / `camera3`. See Lab 01 §5 for `rename_map` and the MI300X training recipe (`batch_size=64`, `steps=50000`).

## Quick load

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-smolvla")
```

Sim2sim eval in MuJoCo: follow **§6 Policy eval** in [lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) (SmolVLA configs under `labs/lab01_pnp/configs/rollout_smolvla*.yaml`).
