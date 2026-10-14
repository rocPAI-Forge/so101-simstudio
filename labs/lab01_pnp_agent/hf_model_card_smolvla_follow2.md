---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - smolvla
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: lerobot/smolvla_base
---

# SO-101 Lab01 Pick-and-Place — SmolVLA follow2 (agent-yaw, 193 episodes)

SmolVLA fine-tune on the **193-episode** agent-yaw follow2 set. Actions are absolute 6-D joint `.pos` (`use_delta_joint_actions_aloha=false`). State input is the first 6 joint positions.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2)

**Lab01-PnP-Home20:** **16/20, 18/20, 17/20 = 51/60**. Arm `home`, gripper start 0.85, seeds 1000/1001/1002. This is not a fresh full-range n=50. Comparison with the other runs: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).

Checkpoint `002500`. Chunk / `n_action_steps` 50.

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2"
)
```
