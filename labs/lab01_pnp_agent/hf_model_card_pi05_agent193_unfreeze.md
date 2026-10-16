---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - pi05
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: ant0nh/pi05_pnp_425_25k
---

# SO-101 Lab01 Pick-and-Place — pi05 full unfreeze (193 episodes, Home20 34/60)

pi05 fine-tune from [`ant0nh/pi05_pnp_425_25k`](https://huggingface.co/ant0nh/pi05_pnp_425_25k). The whole PaliGemma stack is trainable: vision tower, projector, language model, and action expert (`train_expert_only=false`, `freeze_vision_encoder=false`). 4,143,404,816 learnable parameters. `use_relative_actions=false`, dtype bfloat16.

Cameras: `camera_wrist` → `observation.images.wrist`, `camera_front` → `observation.images.front`. The recorded top camera is not an input. State and action are 6-D absolute joint `.pos`.

This repo is the pure-model **34/60** checkpoint. It does not replace [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193), which is the earlier expert-only 10K run at **12/60** and maps `camera_top` to `front`.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2) (193 episodes). On disk, `observation.state` is 15-D; training uses the first 6 joint positions.

**Lab01-PnP-Home20** (no expert takeover): **12/20, 11/20, 11/20 = 34/60**. Lift 38/60, close 58/60, drop 3/60. Arm `home`, gripper start 0.85, seeds 1000/1001/1002. Quote `final_success`.

The published weights are the third 5000-step stage. Each stage starts a fresh optimizer at peak lr `2.5e-6`, warmup 200, decay over that stage's 5000 steps down to `2.5e-7`. The checkpoint directory is `005000` because the step counter restarts. Chunk / `n_action_steps` 50.

How this score was reached: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md#pi05).

```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

policy = PI05Policy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze"
)
```
