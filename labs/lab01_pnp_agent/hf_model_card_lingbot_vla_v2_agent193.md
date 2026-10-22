---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - lingbot-vla-v2
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2
base_model: lerobot/lingbot_vla_v2_base
---

# SO-101 Lab01 Pick-and-Place — LingBot-VLA 2.0 (193 episodes, 20K, Home20 25/60)

Fine-tune of [`lerobot/lingbot_vla_v2_base`](https://huggingface.co/lerobot/lingbot_vla_v2_base) on the 193-episode agent-yaw follow2 set with **joint slots** (`arm.position` + `effector.position`). Cameras: top + wrist. Absolute joint `.pos` supervision.

Pinned SO-101 SimStudio LeRobot (`v0.6.0`) does **not** ship `lingbot_vla_v2`. Eval needs a side checkout of Hugging Face LeRobot PR [#3967](https://github.com/huggingface/lerobot/pull/3967) at `543f4207` plus project patch [`lerobot-vla-v2-inverse-slot-offset.patch`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/patches/lerobot-vla-v2-inverse-slot-offset.patch) (gripper read-back bug on unmapped slots). See [`lingbot_vla_v2_adapt.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lingbot_vla_v2_adapt.md).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2).

**Lab01-PnP-Home20** (no expert takeover, **default** flow sampler): **11/20, 8/20, 6/20 = 25/60**. Reach 60/60, within 4 cm 59/60, close 56/60, lift 28/60, over box 27/60. Arm `home`, gripper start 0.85, seeds 1000/1001/1002. Quote `final_success`. Pre-patch closed-loop scores are invalid.

Trained from a 5K FSDP warm start, then +20K full unfreeze on 4× AMD Instinct MI210 (global batch 4). This repo is checkpoint `020000`.

Comparison: [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).

```bash
# Download
hf download alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k \
  --local-dir ./outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000/pretrained_model

# Side tree + patch + Home20 (from so101-simstudio repo root)
./labs/lab01_pnp/setup_lingbot_vla_v2_side_tree.cmd
./labs/lab01_pnp/eval_lingbot_vla_v2_home20.cmd
```
