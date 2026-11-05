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
  - rgb-d
datasets:
  - alexhegit/so101-simstudio-rax-drawer-place100d
---

# SO-101 RAX drawer — ACT RGB-D (place100d)

ACT trained **with** overhead `camera_top_depth` (3-channel gray of metric depth) on [`alexhegit/so101-simstudio-rax-drawer-place100d`](https://huggingface.co/datasets/alexhegit/so101-simstudio-rax-drawer-place100d). Cameras: top / side / wrist / top_depth. Absolute 6-D joint `.pos`. 30 000 steps (MI300X).

Closed-loop (halo, EGL): explore **6/10 (60%)**, test **11/20 (55%)** — **+20 pp** vs the RGB-only sibling on the same dataset.

RGB sibling: [`alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d)

Docs: [DEPTH_ABLATION.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/rax_drawer_vla/labs/rax_drawer/vla/DEPTH_ABLATION.md)

## Load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d")
```
