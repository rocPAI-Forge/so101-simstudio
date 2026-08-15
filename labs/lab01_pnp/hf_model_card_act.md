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
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
---

# SO-101 Lab01 Pick-and-Place — ACT

ACT policy fine-tuned on expert demonstrations collected and validated with **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** (MuJoCo sim2sim, leader-arm teleop).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough (record → train → eval) | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

For dataset layout, training commands, eval configs, and reference metrics, see **Lab 01** in the repo above.

## Quick load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-act")
```

Sim2sim eval in MuJoCo: follow **§6 ACT eval** in [lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md).
