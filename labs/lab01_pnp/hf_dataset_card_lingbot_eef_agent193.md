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
  - lingbot-va
  - eef-delta
  - libero-action
size_categories:
  - 100K<n<1M
---

# SO-101 Lab01 — agent193 LingBot-VA EEF labels (derived)

LeRobot v3.0 **derived** view of the 193-episode agent-yaw follow2 joint dataset.
Videos and proprioception match the source; only **`action`** (and its stats /
feature meta) are rewritten to **LIBERO-style 7-D Cartesian EEF Δ + gripper**
for fair LingBot-VA fine-tuning.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent193-lingbot-eef` |
| Source (joints) | [`alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2) |
| Robot | `so101_mujoco` |
| Action | 7-D: `eef.delta_{x,y,z}`, `eef.delta_axis_angle_{x,y,z}`, `gripper.command` |
| State on disk | 15-D (unchanged from source) |
| Cameras | `camera_front`, `camera_top`, `camera_wrist` (640×480, 20 Hz) |
| Size | **193 episodes / 130835 frames** |

## Action contract

Matches `lerobot/lingbot_va_libero_long` / LIBERO OSC scaling:

- translation channels clipped to `[-1, 1]`, scale **0.05 m**
- rotation channels clipped to `[-1, 1]`, scale **0.5 rad** (world-frame axis-angle)
- gripper: `+1` close, `-1` open (held per frame)

Labels are produced offline with MuJoCo FK (`SO101EEFAdapter` /
`labs/lab01_pnp/build_lingbot_eef_dataset.py`). Closed-loop eval on SO-101 still
needs the project IK postprocessor (`LAB01_LINGBOT_EEF_ADAPTER=true`).

**Do not** train joint-space policies (ACT, MolmoAct2 6-D, …) on this repo — use
the source joint dataset instead.

## Rebuild

```bash
.venv-rocm/bin/python labs/lab01_pnp/build_lingbot_eef_dataset.py \
  --source datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2 \
  --output .tmp/lab01_lingbot_eef_agent193

LAB01_LINGBOT_EEF_DATASET_ROOT=./.tmp/lab01_lingbot_eef_agent193 \
  ./labs/lab01_pnp/push_lingbot_eef_dataset.cmd
```

Pipeline and reuse notes: [`lingbot_va_eef_adapt.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lingbot_va_eef_adapt.md).

## Not a Home20 ranking dataset

This set enables fair EEF-policy transfer experiments. Published Home20 scores
for joint policies stay on the **source** joint Hub id and
[`home20.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp_agent/home20.md).
Do not quote Home20 for a LingBot-VA checkpoint until fixed-cube motion is
clearly task-directed.
