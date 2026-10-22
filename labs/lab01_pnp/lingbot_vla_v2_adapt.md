# LingBot-VLA 2.0 adaptation for SO-101 (Lab 01)

Experimental joint-slot path for Lab 01. Distinct from LingBot-VA
(`lerobot/lingbot_va_*`, official LIBERO 7-D ΔEEF); that family's fair FK→ΔEEF
adapter was retired. Ranked LingBot work here is **VLA 2.0 + inverse-slot patch**.

| | LingBot-**VA** (not ranked) | LingBot-**VLA 2.0** |
|---|---|---|
| Hub base | `lerobot/lingbot_va_*` | [`lerobot/lingbot_vla_v2_base`](https://huggingface.co/lerobot/lingbot_vla_v2_base) |
| Action language | LIBERO 7-D ΔEEF | **55-D slots** |
| Pin LeRobot `30da8e68` / 0.6.0 | has `lingbot_va` | **no** `lingbot_vla_v2` |

Issue [#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19). Ranked Home20
cell: [`home20.md`](../lab01_pnp_agent/home20.md).

## Joint-slot contract

Reuse the published 193 joint demos (no FK→ΔEEF):

- `arm.position` ← dims `[0:5]`
- `effector.position` ← dim `[5:6]`
- cameras: `camera_top` → `camera_top`, `camera_wrist` → `camera_wrist_left`

Slot file: [`configs/lingbot_vla_v2_so101_slots.json`](configs/lingbot_vla_v2_so101_slots.json).
Dataset: `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`.

## Side checkout + required patch

Policy code is **not** in the pinned `lerobot/` submodule. Use a side tree of
Hugging Face LeRobot PR [#3967](https://github.com/huggingface/lerobot/pull/3967)
(head `543f4207`), e.g. `/tmp/lerobot_vla_v2_src`. Do not commit into `lerobot/`.

PR #3967’s inverse slot step skips unmapped slots when advancing the canonical
read offset. With our layout (`arm` filled → `end` **empty** → `effector`
filled), gripper dim 5 was read from canonical dim **14** instead of **28**.
Closed-loop gripper was unsupervised noise; arm dims were fine. Apply before
any eval:

```bash
git -C /tmp/lerobot_vla_v2_src apply "$PWD/patches/lerobot-vla-v2-inverse-slot-offset.patch"
```

[#4897](https://github.com/huggingface/lerobot/pull/4897) already fixes the
offset logic, but changes processor config shape (`action_spans` /
`canonical_joints`). Migrating the side tree there needs checkpoint processor
conversion first; until then stay on #3967 + this patch.

Teacher-forced check on the MI210 20K weights: gripper MAE ~0.55 → ~0.04 with
the patch. Pre-patch `close` / `lift` / success numbers are invalid.

## Ranked checkpoint and scores

Local (not on the Hub):

`outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000`

Default policy path: `LAB01_V2_POLICY_PATH` in [`_env.sh`](_env.sh).

| Score | Sampler | Notes |
|---|---|---|
| **25/60** | default (noise ×1, 10 steps) | **Official / fair** — same bar as other `home20.md` rows |
| 58/60 | `LAB01_FLOW_NOISE_SCALE=0 LAB01_FLOW_NUM_STEPS=10` | Eval-time note only; optimistic; selected after looking at Home20 |

Default-sampler failures are mostly empty grasps. Lower noise shrinks close
standoff; a seed-1000 sweep showed success rising as noise dropped to ≤0.25,
with three or more denoise steps needed (one step alone was worse). Quote any
`LAB01_FLOW_*` overrides next to the number. There is no V2 eval `.cmd` yet;
Home20 wrappers live under `.tmp/`.

Post-patch finding: the same 20K weights on the same 193 demos are enough for
the ranked cell. Scripted DAgger on top of that set was a pre-patch dead end
and has been removed from the lab.

## Training caveat

Converted checkpoints embed `optimizer_lr = scheduler_decay_lr = 1e-5`. Lab
runs so far did not override that (constant 1e-5, global batch 4). Upstream
side-tree recipes use higher LRs / larger batches — pass those flags explicitly
if matching them.
