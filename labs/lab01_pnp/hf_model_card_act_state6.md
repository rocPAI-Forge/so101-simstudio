---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - act
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
---

# SO-101 Lab01 Pick-and-Place — ACT (6-D joint state)

ACT policy fine-tuned on expert demonstrations from **[SO-101 SimStudio](https://github.com/rocPAI-Forge/so101-simstudio)** (MuJoCo, leader-arm teleop).

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp) (on-disk `observation.state` is still 15-D: 6 joint pos + 6 vel + 3 EE). This checkpoint was trained on the **first 6 dims only** (joint `.pos`), matching official real LeRobot SO-101 (`so_follower`) proprioception.

**Why 6-D:** so the policy input matches **real-robot IL** and is easier to **sim2real** / merge with real datasets. Extra sim channels (velocity, end-effector XYZ) are not available on the stock real follower. Units (radians vs degrees, gripper scale) still need a separate alignment.

The 15-D ACT reference (same 50K schedule, pos+vel+ee) is [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act).

## This Hub revision

| Item | Value |
|------|-------|
| GPU | AMD Instinct MI300X (DORobot) |
| Batch / steps | **128 / 50 000** |
| Train loss | **0.054** |
| Sim2sim eval | full-range, `reset_arm: follow`, sync, EGL, `n_action_steps=50` → **29/50 (58%)** |
| 15-D ACT same protocol | **32/50 (64%)** — same level at n=50 |

Wall time ~**17 h**. Checkpoint `050000` / `last`.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 walkthrough | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |

## Quick load

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("alexhegit/so101-simstudio-lab01-pnp-act-state6")
```

```bash
hf download alexhegit/so101-simstudio-lab01-pnp-act-state6 \
  --local-dir ./outputs/hub/lab01_pnp_act_state6
```

Sim2sim eval: `labs/lab01_pnp/configs/rollout_act.yaml` (same YAML as 15-D ACT; rollout already sends joint `.pos`).
