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

ACT trained **without** `camera_top_depth` on [`alexhegit/so101-simstudio-rax-drawer-place100d`](https://huggingface.co/datasets/alexhegit/so101-simstudio-rax-drawer-place100d). Cameras: top / side / wrist. Absolute 6-D joint `.pos`. 30 000 steps.

Closed-loop (halo, EGL): explore **4/10**, test **7/20**.

RGB-D sibling (same data): [`alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d) — explore 6/10, test 11/20.

Docs: [DEPTH_ABLATION.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/rax_drawer_vla/labs/rax_drawer/vla/DEPTH_ABLATION.md)

## Load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d")
```
