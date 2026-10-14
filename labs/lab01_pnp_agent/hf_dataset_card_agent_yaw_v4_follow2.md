---
license: apache-2.0
task_categories:
  - robotics
tags:
  - lerobot
  - so101
  - simstudio
  - mujoco
  - pick-and-place
  - scripted-expert
  - yaw-aware
size_categories:
  - 100K<n<1M
---

# SO-101 Lab01 Pick-and-Place — agent-yaw v4 actonly-follow2 (193 episodes)

LeRobot v3.0 trajectories for the SO-101 MuJoCo pick-and-place lab. Task string: `Pick up the cube and place it in the box`.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2` |
| Robot / action | `so101_mujoco`, 6-D joint position |
| State on disk | 15-D: 6 joint pos + 6 vel + 3 EE XYZ |
| Cameras | `camera_front`, `camera_top`, `camera_wrist` (640×480, 20 Hz) |
| Size | **193 episodes / 130835 frames** |

Policies that train a sim2real joint layout slice the first 6 state dims (joint `.pos`). The on-disk column stays 15-D.

This repo does not replace `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4` (140 episodes). Episodes 0–139 of this set are that v4 set; 140–192 are later ACT-follow successes.

## Episode groups

| Episodes | Count | What is recorded |
|---|---:|---|
| 0–99 | 100 | Scripted expert, one successful grasp from home |
| 100–109 | 10 | Scripted expert, a small alignment while the gripper is still open, then one grasp |
| 110–119 | 10 | Scripted expert, a partial miss is not labeled; the gripper reopens and grasps again |
| 120–139 | 20 | DAgger. v3 ACT 80K rolls until a close without a lift; the scripted expert finishes the episode |
| 140–170 | 31 | ACT follow. Successes from the 6-D ACT 50K trained only on the first 140 episodes, `reset_arm=follow`, 1200 frames, no expert handoff |
| 171–192 | 22 | More successes from that same ACT checkpoint |

ACT follow episodes use `outputs/train/lab01_pnp_act_state6_agent_yaw_v4/checkpoints/050000`. That checkpoint's own Lab01-PnP-Home20 score is **53/60**. The DAgger block uses an earlier v3 ACT 80K, not this checkpoint.

## Models trained on this set

| Model | Hub | Lab01-PnP-Home20 |
|---|---|---|
| SmolVLA follow2 2.5K | [alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2) | **51/60** |
| MolmoAct2 plus 20K | [alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k) | **39/60** |
| pi05 expert-only 10K | [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193) | **12/60** |
| pi05 full unfreeze | [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze) | **34/60** |
| GR00T N1.7 absolute, backbone frozen, 10K | [alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k) | **14/60** |
| DM0.5 10K | [alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k) | **25/60** |

Home20 is 20 frozen cube poses, arm `home`, gripper start 0.85, seeds 1000/1001/1002. Quote `final_success`. The rows share this protocol and, except the ACT run on the first 140 episodes, this dataset. Step count, learning rate, cameras, which weights were trained, and absolute versus relative actions differ. A higher score is that checkpoint on these 20 poses, not a ranking of the model. This table lists the checkpoints uploaded from this dataset. The full score table, including runs that were not uploaded, is [`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).
