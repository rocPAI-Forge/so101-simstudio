---
license: gemma
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - dm05
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: lerobot/dm05_base
---

# SO-101 Lab01 Pick-and-Place — DM0.5 (193 episodes, 10K, Home20 25/60)

Fine-tune of [`lerobot/dm05_base`](https://huggingface.co/lerobot/dm05_base) on the 193-episode agent-yaw set. Full finetune (`use_peft=false`); the VLM token embedding stays frozen. 4,486,607,248 learnable parameters. `add_state=true`, `use_relative_actions=false`, dtype bfloat16.

Cameras: `camera_top` and `camera_wrist`. The recorded front camera is not an input. State and action are 6-D absolute joint `.pos`. Images are resized to 448×448. Chunk / `n_action_steps` 10.

The pinned SO-101 SimStudio submodule is LeRobot v0.6.0, which does not ship DM0.5. Load this repo with a LeRobot revision that includes `lerobot.policies.dm05`.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2) (193 episodes). On disk, `observation.state` is 15-D; training uses the first 6 joint positions.

**Lab01-PnP-Home20** (no expert takeover): **7/20, 7/20, 11/20 = 25/60**. Reach 60/60, within 4 cm 60/60, close 59/60, lift 27/60, drop 1/60. Arm `home`, gripper start 0.85, seeds 1000/1001/1002. Quote `final_success`.

Trained 10000 steps on 4× AMD Instinct MI210, batch 1 per GPU (effective batch 4), peak lr `2.5e-5`, warmup 400, cosine decay over 10000 steps to `2.5e-6`. About 1.65 s/step, 51 GB per GPU, about 4 h 36 min. This repo is checkpoint `010000`. The 5000-step checkpoint was not uploaded.

Comparison: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md#dm05).

Weights inherit the Gemma license from `lerobot/dm05_base`.

```python
from lerobot.policies.dm05.modeling_dm05 import DM05Policy

policy = DM05Policy.from_pretrained(
    "alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k"
)
```
