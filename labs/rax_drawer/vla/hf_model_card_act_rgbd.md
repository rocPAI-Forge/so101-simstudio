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

ACT trained **with** overhead `camera_top_depth` (metric top depth as 3-channel gray for ResNet) on
[`alexhegit/so101-simstudio-rax-drawer-place100d`](https://huggingface.co/datasets/alexhegit/so101-simstudio-rax-drawer-place100d).
Cameras: `camera_top` / `camera_side` / `camera_wrist` / `camera_top_depth`. Absolute **6-D** joint `.pos`. **30 000** steps on AMD Instinct MI300X.

Repo write-up (canonical): [DEPTH_ABLATION.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/release-v0.1.6/labs/rax_drawer/vla/DEPTH_ABLATION.md) · tag [`release-v0.1.6`](https://github.com/rocPAI-Forge/so101-simstudio/releases/tag/release-v0.1.6)

## Comparison results (closed-loop)

Same demos (`place100d`), same ACT budget, eval on halo with `MUJOCO_GL=egl`, locked manifests, state 6-D.

| Policy | explore10 | test20 | Δ vs place100d RGB |
|--------|-----------|--------|--------------------|
| **This checkpoint (RGB-D)** | **6/10 (60%)** | **11/20 (55%)** | **+20 pp / +20 pp** |
| ACT RGB sibling (no depth) | 4/10 (40%) | 7/20 (35%) | — |
| Prior place100b RGB (reference) | 5/10 (50%) | 9/20 (45%) | — |

**Takeaway:** matched-data top-depth gray helps ACT open→place→close on this drawer task. Caveats: small n; depth is gray pseudo-RGB (not a dedicated depth backbone).

RGB sibling: [`alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d)

## Load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d")
```
