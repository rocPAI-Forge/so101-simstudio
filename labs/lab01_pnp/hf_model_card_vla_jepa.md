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

# SO-101 Lab01 Pick-and-Place — VLA-JEPA (transfer failure)

VLA-JEPA fine-tuned from [`lerobot/VLA-JEPA-LIBERO`](https://huggingface.co/lerobot/VLA-JEPA-LIBERO) on the Lab 01 50-episode leader dataset. **This is a documented negative result**, not a classroom pick-and-place policy. Same data: ACT 6-D **58%** and MolmoAct2 6-D **54%** full-range; this checkpoint does not reliably grasp.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

This Hub revision is the **best measured Lab 01 JEPA run**: 6-D joint `.pos` BC, freeze Qwen, world model off, `chunk_size` / `n_action_steps` **30**, batch **16** × **10 000** steps on AMD Instinct MI300X (DORobot). Train loss ~**0.020**. Checkpoint `010000`. Do not prefer the 20K BC or world-model 20K weights — both scored **0/10** on the same fixed-pose protocol.

Cameras: `camera_top` → `image`, `camera_wrist` → `image2`. Action and proprio are **6-D** joint position. The LIBERO base config still lists 8-D `input_features`; SimStudio `eval.py` keeps the fine-tune `state_dim=6`.

## Closed-loop (do not treat as a working policy)

Fixed cube `(0.27, 0.20, −8°)`, `reset_arm: home`, sync, `n_action_steps=30`:

| Protocol | Backend | Success |
|----------|---------|---------|
| Demo-mean `home_joints` | EGL | **1/10** |
| Keyboard `home` | EGL | **1/10** |
| Demo-mean start, GLFW GUI | GLFW | **0/10** (grasp-stage failures) |
| Same + proximity gripper oracle | EGL | **0/10** (close fired every episode; cube never lifted) |

Failures concentrate at **grasp**: the arm approaches; the gripper stays open (~0.85–0.93 rad) and the fingertip frame is ~4 cm off the cube in Y. Threshold snap cannot recover a close command that is never issued. Shorter chunks (10 / 5) made opening later or absent.

LIBERO is 7-DoF / 8-D state / 2 cams. Fine-tune reinitializes action/state heads (`reinit_modules`) and turns off LIBERO gripper binarize (`gripper_dim=6` does not apply to SO-101 dim 5).

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough §6.4 | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

## Quick load

```python
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

policy = VLAJEPAPolicy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-vla-jepa")
```
