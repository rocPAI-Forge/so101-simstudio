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
  - dagger
size_categories:
  - 10K<n<100K
---

# SO-101 Lab01 Pick-and-Place Dataset (yaw-aware agent, v4)

Privileged-state scripted expert + full-episode DAgger recoveries recorded with
SO-101 SimStudio into LeRobot v3.0. Built on **v3** (same expert fixes); adds
on-policy fail→recover→place episodes. Does not overwrite v1–v3 Hub datasets.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4` |
| Robot/action | `so101_mujoco`, 6-D joint position |
| State | 15-D position + velocity + EE XYZ |
| EE point | fingertip site `gripperframe` |
| Cameras | front, top, wrist (640×480, 20 Hz) |
| Spawn | x [0.26, 0.34], y [0.165, 0.235], yaw ±45° |
| Size | 140 episodes / 67235 frames |

## Composition

| Kind | Count | Notes |
|---|---:|---|
| v3 first-try + recovery base | 120 | Carried from [agent-yaw-v3](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v3) |
| Full DAgger (`policy_dagger_full`) | 20 | Home start; policy rolls until fail; expert recovers through place |

Mid-episode recovery-only handovers (no home start) are **not** included.

## vs v3

- Same spawn range, cameras, 15-D state, and expert quality gates as v3.
- Extra 20 full-home DAgger episodes target close→lift→place distribution shift.
- Reference ACT (15-D, MI300X, 80K): [alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4).

Collectors: `labs/lab01_pnp/auto_collect_yaw.py`,
`labs/lab01_pnp/collect_policy_recovery.py` (`--mode full`).
