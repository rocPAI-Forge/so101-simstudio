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
  - mujoco
  - drawer
datasets:
  - alexhegit/so101-simstudio-rax-drawer-place100d
---

# SO-101 RAX drawer — ACT RGB (place100d)

ACT trained **without** `camera_top_depth` on
[`alexhegit/so101-simstudio-rax-drawer-place100d`](https://huggingface.co/datasets/alexhegit/so101-simstudio-rax-drawer-place100d).
Cameras: top / side / wrist. Absolute **6-D** joint `.pos`. **30 000** steps.

Repo write-up (canonical): [DEPTH_ABLATION.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/release-v0.1.6/labs/rax_drawer/vla/DEPTH_ABLATION.md)

## Comparison results (closed-loop)

Paired ablation vs the RGB-D sibling on the **same** `place100d` demos:

| Policy | explore10 | test20 |
|--------|-----------|--------|
| **This checkpoint (RGB)** | 4/10 (40%) | 7/20 (35%) |
| ACT RGB-D sibling (+ top depth) | **6/10 (60%)** | **11/20 (55%)** |

RGB-D sibling: [`alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d)

## Load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d")
```
