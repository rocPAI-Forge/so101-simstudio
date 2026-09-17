---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - fastwam
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
base_model: ZibinDong/fastwam_libero_uncond_2cam224
---

# SO-101 Lab01 Pick-and-Place — FastWAM (transfer failure)

FastWAM fine-tuned on the Lab 01 50-episode leader dataset. **This is a documented negative result** for LIBERO→SO-101 **joint-space** transfer, not a classroom pick-and-place policy. Same data: ACT 6-D **58%** and MolmoAct2 6-D **54%** full-range; this checkpoint never approaches the cube (`reach_8cm=0`).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

This Hub revision is the **best measured Lab 01 FastWAM run** (still **0%** closed-loop): start from the LIBERO 2-cam checkpoint [`ZibinDong/fastwam_libero_uncond_2cam224`](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224) (not bare `lerobot/fastwam_base`, which has no action pathway tensors), **pad 6 SO-101 joints into the 7-D action / 8-D proprio slots**, freeze the Wan video expert then **unfreeze the last 4 video MoT layers** for 8K steps. `lambda_video=0` (action-only; not the paper’s video co-training). Checkpoint `008000`. Open-loop RMSE **0.136 rad** (~3.8× ACT). Do **not** prefer the earlier `fastwam_base` 6-D 4K/20K runs (worse open-loop, still 0%).

Cameras: `camera_top` → `image`, `camera_wrist` → `image2` (key-name alignment only; Lab 01 top-down ≠ LIBERO agentview). Eval in SimStudio unpads the 6 real joints (`LAB01_FASTWAM_PAD_ACTION`). Failures are **before approach**: the arm moves, the cube stays at spawn `z≈0.015`. Visual probes still respond to cube pixels; the defect is cross-embodiment action/coordinate grounding, not a dead camera. Ledger: Lab 01 §6.4.2.

## Closed-loop (do not treat as a working policy)

Full-range random spawn, `reset_arm: home`, sync, `n_action_steps=32`:

| Protocol | Backend | Success |
|----------|---------|---------|
| Keyboard `home` | OSMesa | **0/10** |
| Keyboard `home` | GLFW GUI | **0/5** (same: cube never leaves spawn) |
| Matched n=3 funnel | OSMesa | `reach_8cm=0/3` |

Earlier Lab 01 FastWAM recipes (`fastwam_base` freeze-video 4K/20K, pad7 without unfreeze-4) are also **0%** and are **not** this Hub revision.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough §6.4 / §6.4.2 | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

## Quick load

```python
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

policy = FastWAMPolicy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-fastwam")
```

Sim2sim eval: `LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_fastwam.yaml` and the FastWAM pad eval patch in `eval.cmd`. Not a demo policy.
