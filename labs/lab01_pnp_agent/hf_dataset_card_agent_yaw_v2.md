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

# SO-101 Lab01 Pick-and-Place Dataset (yaw-aware agent, v2)

Privileged-state scripted expert recorded with SO-101 SimStudio into LeRobot
v3.0. This set extends
`alexhegit/so101-simstudio-lab01-pnp-agent-yaw` with recovery trajectories.
It does not overwrite or mix that 100-episode set.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v2` |
| Robot/action | `so101_mujoco`, 6-D joint position |
| State | 15-D position + velocity + EE XYZ |
| EE point | fingertip site `gripperframe` |
| Cameras | front, top, wrist (640×480, 20 Hz) |
| Spawn | x [0.26, 0.34], y [0.165, 0.235], yaw ±45° |
| Size | 188 episodes / 77471 frames |

## Composition

Saved episodes only. Deliberate misses are not labeled; recording starts at
the correction.

| Kind | Count | Episode index | Recorded from |
|---|---:|---|---|
| First-try success | 78 | 0–79 except 24, 67 | Start; one grasp |
| Natural regrasp | 5 | 24, 67, 80, 81, 82 | Start; includes the real failed first grasp, then regrasp |
| Fixed recovery | 20 | 83–102 | Gripper half-open (+0.50) beside the cube; deliberate miss not recorded |
| Diverse recovery | 40 | 103–142 | Same, with tangent / radial / diagonal 8–16 mm and ±5–10° wrist offsets |
| Closed empty-grasp recovery | 20 | 143–162 | Gripper already closed (−0.08) on nothing; open and regrasp |
| Pre-close correction | 15 | 163–177 | Gripper still open near the cube; small correction, then one grasp |
| Policy handover | 10 | 178–187 | State after v1 ACT 80K closed without lifting; policy actions not recorded |

Only the 5 natural regrasps show a real first-grasp failure. No episode shows
a drop after lift. Episode indices are the LeRobot `episode_index`, matched to
`spawns.jsonl` by order and frame count (188/188). Two extra regrasp rows in
`spawns.jsonl` (620 and 637 frames) were never saved to parquet.

Recovery frames are about 53% of the dataset. `ee.pos` on the 10 handover
episodes was recomputed from the recorded joint angles onto `gripperframe`
so it matches the rest of the set.

## Expert behavior

The collector uses privileged cube pose only to generate demonstrations:
align the wrist to the nearest cube face, approach a face-normal standoff,
creep, close, lift, and place. Only verified successes are kept.
`spawns.jsonl` records spawn, yaw bin, XY cell, recovery style, and success.
Privileged pose is not part of policy observations.

Collector: `labs/lab01_pnp/auto_collect_yaw.py`.
