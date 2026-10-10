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

# SO-101 Lab01 Pick-and-Place Dataset (yaw-aware agent, v3)

Privileged-state scripted expert recorded with SO-101 SimStudio into LeRobot
v3.0. Built with the post-2026-09-28 expert fixes (wrist roll aligned to the
approach face, two-jaw contact gate, far-reach quality rejects). Does not
overwrite v1, v2, or v2-clean.

| | |
|---|---|
| Hub id | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v3` |
| Robot/action | `so101_mujoco`, 6-D joint position |
| State | 15-D position + velocity + EE XYZ |
| EE point | fingertip site `gripperframe` |
| Cameras | front, top, wrist (640×480, 20 Hz) |
| Spawn | x [0.26, 0.34], y [0.165, 0.235], yaw ±45° |
| Size | 120 episodes / 52476 frames |

## Composition

| Kind | Count | Notes |
|---|---:|---|
| First-try success | 100 | Dual quota: 6 yaw bins × 2×2 XY cells (25 each); one grasp from home |
| Pre-close correction | 10 | Gripper still open; small align, then one grasp; full home→place |
| Partial-miss local reclose | 10 | Half-open (+0.50) miss not labeled; open in place and regrasp (no high retreat) |

Deliberate wrong actions are never saved as expert labels. Failed attempts are
discarded; temporary video PNGs are cleared so episode videos match parquet
length (no trailing tails).

## Expert fixes vs v1 / v2

- Wrist roll chosen so the jaw axis is perpendicular to the approach-face normal
  (square 90° ambiguity removed).
- Approach face ranked by roll cost, then standoff reach.
- Close rejected unless both jaws contact the block (≥1 N) and pre-close
  tangent error ≤10 mm.
- Far-table unreachable poses are quality-rejected, not saved as empty grasps.

Collector: `labs/lab01_pnp/auto_collect_yaw.py`.
`spawns.jsonl` records spawn, yaw bin, XY cell, recovery style, and success.
