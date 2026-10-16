---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - act
  - so101
  - simstudio
  - yaw-aware
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4
---

# SO-101 Lab01 Pick-and-Place — ACT (agent-yaw v4, 6-D)

ACT policy trained on the yaw-aware agent **v4** dataset with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)**.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4). The on-disk dataset `observation.state` is 15-D (6 joint pos + 6 vel + 3 EE). This checkpoint was trained on the **first 6 dims only** (joint `.pos`).

This repo previously held a **15-D** ACT (MI300X 80K, home n=20 → 17/20). That revision was **deleted**. Do not cite 17/20 for the weights in this repo.

**Lab01-PnP-Home20** for this 6-D checkpoint (`050000`, trained on the 140-episode v4 set, not the later 193-episode follow2 set): **18/20, 17/20, 18/20 = 53/60**. Arm `home`, gripper start 0.85, seeds 1000/1001/1002. Quote `final_success`. Comparison with the 193-episode runs: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).

## Recipe

| | |
|---|---|
| Policy | ACT, **6-D** joint `.pos` |
| Host | acc / NVIDIA A30 |
| Batch / steps | 16 / **50000** |
| Train loss | 0.032 |
| Checkpoint | `050000/pretrained_model` |

## Quick load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4"
)
```

Lab walkthrough: [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md).
