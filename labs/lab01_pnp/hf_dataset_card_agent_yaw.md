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
  - 10K<n<100K
---

# SO-101 Lab01 Pick-and-Place Dataset (yaw-aware scripted agent)

Privileged-state scripted expert recorded with SO-101 SimStudio into LeRobot
v3.0. This is a new dataset; it does not overwrite or mix the fixed-wrist
`alexhegit/so101-simstudio-lab01-pnp-agent` set.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw` |
| Robot/action | `so101_mujoco`, 6-D joint position |
| State | 15-D position + velocity + EE XYZ |
| Cameras | front, top, wrist (640×480, 20 Hz) |
| Spawn | x [0.26, 0.34], y [0.165, 0.235], yaw ±45° |
| Coverage | Six yaw bins with balanced success quotas |
| Size | 50 episodes / 22933 frames |
| Collection yield | 50 successes / 68 attempts (73.5%) |

## Expert behavior

The collector uses privileged cube pose only to generate demonstrations:

1. choose a wrist-roll solution aligned with the nearest cube face;
2. approach a face-normal standoff;
3. creep until first light contact, then close;
4. keep the wrist locked during lift/carry;
5. save only verified pick-and-place successes.

`spawns.jsonl` records XYZ, yaw/bin, XY cell, target/actual wrist roll,
pre-close slip, lift status, success, and frame count. Privileged pose is not
part of policy observations.

Success quotas per yaw bin: `[9, 8, 8, 8, 8, 9]`. Saved trajectories span
yaw −44.93° … +44.66°, wrist targets −0.650 … +1.049 rad, and pre-close
cube slip ≤4.15 mm.

Collector: `labs/lab01_pnp/auto_collect_yaw.py`.
Experiment rationale and predecessor failure: `labs/lab01_pnp/agent_act_80k.md`.
