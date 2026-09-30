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

# SO-101 Lab01 Pick-and-Place — ACT (agent-yaw v4)

ACT policy trained on the yaw-aware agent **v4** dataset (v3 base + 20 full
DAgger episodes) with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)**.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4)

## Recipe

| | |
|---|---|
| Policy | ACT, 15-D state (pos + vel + EE XYZ @ `gripperframe`) |
| Host | DORobot / MI300X |
| Batch / steps | 64 / **80000** (from scratch; ckpts every 20K) |
| Checkpoint | `080000/pretrained_model` |

## Eval (sim, home reset, n=20)

Protocol: `home`, `n_action_steps=50`, EGL, Lab 01 agent-yaw rollout config.

| Checkpoint | success | lift | close | dropped |
|---|---|---|---|---|
| 40K | 14/20 (70%) | 17/20 | 20/20 | 4/20 |
| **80K** | **17/20 (85%)** | 17/20 | 19/20 | 0/20 |

Same protocol on **v3** ACT 80K: 13/20 (65%).

## Quick load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4"
)
```

Lab walkthrough: [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md).
