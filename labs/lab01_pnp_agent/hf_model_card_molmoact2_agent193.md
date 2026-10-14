---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - molmoact2
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: lerobot/MolmoAct2-SO100_101-LeRobot
---

# SO-101 Lab01 Pick-and-Place — MolmoAct2 (193 episodes, plus 20K)

MolmoAct2 fine-tune from [`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) on the 193-episode agent-yaw set, then continued from the 10K weights for another 20K steps. VLM LoRA only (`enable_lora_action_expert=false`). Cameras: `camera_top` → `cam0`, `camera_wrist` → `cam1`. `camera_front` is unused. State is the first 6 joint positions.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2)

**Lab01-PnP-Home20** (bf16 eval): the 10K checkpoint was **36/60**. This 20K checkpoint is **13/20, 13/20, 13/20 = 39/60**. The extra successes are releases after a lift. Six far cube poses still almost never close inside 4 cm. Merging this LoRA and full-finetuning another 5000 steps scored **23/60** and was not uploaded. This repo stays the 20K LoRA weights. Comparison: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).

Checkpoint `020000`. Chunk 30. This Hub repo is the 20K weights, not the 10K weights.

```python
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

policy = MolmoAct2Policy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k"
)
```
