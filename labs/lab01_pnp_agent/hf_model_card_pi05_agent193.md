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

# SO-101 Lab01 Pick-and-Place — pi05 (193 episodes, 10K)

pi05 fine-tune from [`ant0nh/pi05_pnp_425_25k`](https://huggingface.co/ant0nh/pi05_pnp_425_25k) (itself a fine-tune of `lerobot/pi05_base` on a real SO-follower pick-and-place set). `train_expert_only=true`, `use_relative_actions=false`, dtype bfloat16. Cameras: `camera_wrist` → `observation.images.wrist`, `camera_top` → `observation.images.front`. State and action are 6-D absolute joint `.pos`.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2)

**Lab01-PnP-Home20:** **4/20, 6/20, 2/20 = 12/60**. Reach 60/60, close 39/60, lift 15/60. The hole is lift after close.

Checkpoint `010000`. Chunk / `n_action_steps` 50. Sync inference.

```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

policy = PI05Policy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-pi05-agent193"
)
```
