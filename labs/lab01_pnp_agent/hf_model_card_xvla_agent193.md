---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - xvla
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: lerobot/xvla-base
---

# SO-101 Lab01 Pick-and-Place — xVLA (193 episodes, 10K, Home20 47/60)

Full finetune of [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) on the 193-episode SO-101 agent-yaw dataset. All 879,482,456 parameters were trainable.

## Inputs and action adaptation

- Cameras: `camera_top` → `image`, `camera_wrist` → `image2`, `camera_front` → `image3`
- Images: 224×224
- State: first 6 joint `.pos` values from the on-disk 15-D state; xVLA pads these to its 8-D proprio interface
- Actions: 6-D absolute SO-101 joints with `action_mode=auto`; the pretrained 20-D action head is retained, loss is applied to the 6 real dimensions, and inference is trimmed to 6-D
- Chunk / `n_action_steps`: 30 / 30

Do not send the raw 20-D model output directly to the robot.

## Training

- Dataset: [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2), 193 episodes / 130,835 frames
- Steps: 10,000, approximately 1.22 epochs
- Hardware: 4× NVIDIA A30 with DDP
- Batch: 4 per GPU, effective batch 16
- dtype: bfloat16
- Learning rate: peak `1e-4`, warmup 1000, cosine decay over 10,000 steps to `2.5e-6`
- Final training loss: 0.008

Train and evaluate with LeRobot `24f15b92` via `labs/lab01_pnp/train_xvla.cmd` and `labs/lab01_pnp/eval_home20.cmd` (`LAB01_LEROBOT_SRC`). The pinned v0.6.0 xVLA sources differ from that revision.

## Home20 evaluation

Pure-model Lab01-PnP-Home20:

- Seed 1000: 15/20
- Seed 1001: 17/20
- Seed 1002: 15/20
- Total: **47/60**

Stage totals: within 4 cm 60/60, close 57/60, lift 49/60, over box 48/60, released in box 48/60, final success 47/60, drop 2/60.

Protocol: arm `home`, gripper start 0.85, 20 frozen cube poses, seeds 1000/1001/1002, and `final_success` measured at episode end. This result describes this particular training recipe on Home20; it is not a general ranking of xVLA against other models.

Full comparison and recipe: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md#xvla).

```python
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

policy = XVLAPolicy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-xvla-agent193-10k"
)
```

Weights inherit the Apache-2.0 license from `lerobot/xvla-base`.
