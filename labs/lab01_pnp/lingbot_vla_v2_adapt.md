# LingBot-VLA 2.0 on Lab 01 (authoritative path)

Issue [#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19). Ranked
Home20 cell: [`home20.md`](../lab01_pnp_agent/home20.md) — **25/60**.

Same shape as other joint-space finetunes (e.g. GR00T N1.7): **193 follow2 demos
→ finetune → Home20 default eval**. The Lab 01–specific gate is the side-tree
**inverse-slot patch** (upstream gripper read-back bug).

## Contract

| Item | Value |
|---|---|
| Base | [`lerobot/lingbot_vla_v2_base`](https://huggingface.co/lerobot/lingbot_vla_v2_base) |
| Data | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2` (193 joint) |
| Slots | [`configs/lingbot_vla_v2_so101_slots.json`](configs/lingbot_vla_v2_so101_slots.json): `arm.position` ← `[0:5]`, `effector.position` ← `[5:6]`; cams top + wrist |
| Runtime | Side checkout of LeRobot PR [#3967](https://github.com/huggingface/lerobot/pull/3967) (`543f4207`); **not** pinned `lerobot/` |
| Checkpoint | `outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000` (local; `LAB01_V2_POLICY_PATH` in [`_env.sh`](_env.sh)) |
| Eval | Home20, seeds 1000/1001/1002, `home`, gripper 0.85, **default** flow sampler (no `LAB01_FLOW_*`) |

Rollout config: [`configs/rollout_lingbot_vla_v2.yaml`](configs/rollout_lingbot_vla_v2.yaml).

## Required patch (before any eval)

PR #3967 skips unmapped slots when advancing the inverse read offset. With our
layout (`arm` filled → `end` empty → `effector` filled), gripper was read from
canonical dim **14** instead of **28**. Apply:

```bash
git -C /tmp/lerobot_vla_v2_src apply "$PWD/patches/lerobot-vla-v2-inverse-slot-offset.patch"
```

[#4897](https://github.com/huggingface/lerobot/pull/4897) already fixes the
logic but changes processor config shape; migrating needs checkpoint conversion.
Until then: **#3967 + this patch**.

Without the patch, closed-loop gripper is noise and Home20 success is invalid.
Training loss still supervised the right dims; only rollout read-back was wrong.

## Ranked score

**25/60** (11/20 + 8/20 + 6/20) on the MI210 20K weights above, patched side
tree, default sampler.

## Training note

Converted checkpoints embed `optimizer_lr = 1e-5`. Lab runs so far used that
constant LR at global batch 4 unless overridden. Upstream side-tree recipes use
higher LRs — pass flags explicitly if matching them.
