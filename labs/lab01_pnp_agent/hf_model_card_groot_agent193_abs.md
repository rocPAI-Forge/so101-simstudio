---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - groot
  - gr00t
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: nvidia/GR00T-N1.7-3B
---

# SO-101 Lab01 Pick-and-Place — GR00T N1.7 absolute joints (193 episodes, 10K)

Fine-tune of [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) on the 193-episode agent-yaw set. `embodiment_tag=new_embodiment`, `use_relative_actions=false`, LLM and vision frozen. The dataset's 6 joint `.pos` values are the training targets; GR00T still pads them into its 132-D slot. Eval uses sync inference (`inference.type=sync`).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2)

**Lab01-PnP-Home20:** **7/20, 2/20, 5/20 = 14/60**. Reach 60/60, within 4 cm 51/60, close 45/60, lift 17/60. Failed closes are mostly beside the cube (near-zero jaw force). A later full unfreeze of these weights scored **15/60** and was not uploaded. This repo stays the frozen-backbone 10K. The comparison is [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).

The relative-action runs on this same set are not in this repo: 10K scored **0/60** (checkpoint deleted) and a fresh 30K scored **1/60** with approach collapsed to 11/60 reach. A later 5K continuation of these absolute weights at lr `1e-5` was worse on seed 1000 (4/20). Do not resume this cosine; it ended at lr 0.

This is a different artifact from the 50-episode relative-action negative result at [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17).

Weights inherit the **NVIDIA Open Model License** from the N1.7 base.

Checkpoint `010000`. Chunk 16.

```python
from lerobot.policies.groot.modeling_groot import GrootPolicy

policy = GrootPolicy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k"
)
```
