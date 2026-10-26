# LingBot-VLA 2.0 on Lab 01 (authoritative path)

Issue [#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19). Ranked
Home20 cell: [`home20.md`](../lab01_pnp_agent/home20.md) — **25/60**.

Same shape as other joint-space finetunes (e.g. GR00T N1.7): **193 follow2 demos
→ finetune → Home20 default eval**. The Lab 01–specific gate is the side-tree
**inverse-slot patch** (upstream gripper read-back bug).

LingBot-VA (`lingbot_va_*`, LIBERO ΔEEF) is **paused** on SO-101 Lab 01: fair-EEF
Home20 stayed **0/60** (sparse reach / IK); adapter and derived EEF dataset are
gone ([issue #9](https://github.com/rocPAI-Forge/so101-simstudio/issues/9)). That
action language fits **7-DoF Cartesian** arms (e.g. openARM-class dual-arm) better
than SO-101’s 5+1 joint arm — revisit VA there, not as this table’s ranking path.

## Reproduce Home20 25/60 (eval only)

```bash
# 1) Weights (~24 GB)
hf download alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k \
  --local-dir ./outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000/pretrained_model

# 2) Side tree PR #3967 + inverse-slot patch
./labs/lab01_pnp/setup_lingbot_vla_v2_side_tree.cmd

# 3) Home20 seeds 1000/1001/1002 (default sampler — do not set LAB01_FLOW_*)
source .venv-rocm/bin/activate
./labs/lab01_pnp/eval_lingbot_vla_v2_home20.cmd
```

On AMD Instinct hosts without working EGL, use `LAB01_MUJOCO_GL=osmesa`.
Quote `final_success` from the three logs under `outputs/eval/`.

Push local weights (maintainers):

```bash
LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co \
  ./labs/lab01_pnp/push_lingbot_vla_v2_agent193_model.cmd
```

## Contract

| Item | Value |
|---|---|
| Base | [`lerobot/lingbot_vla_v2_base`](https://huggingface.co/lerobot/lingbot_vla_v2_base) |
| Data | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2` (193 joint) |
| Slots | [`configs/lingbot_vla_v2_so101_slots.json`](configs/lingbot_vla_v2_so101_slots.json): `arm.position` ← `[0:5]`, `effector.position` ← `[5:6]`; cams top + wrist |
| Runtime | Side checkout of LeRobot PR [#3967](https://github.com/huggingface/lerobot/pull/3967) (`543f4207`); **not** pinned `lerobot/` |
| Hub finetune | [`alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k`](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k) |
| Local path | `outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000` (`LAB01_V2_POLICY_PATH`) |
| Eval | Home20, seeds 1000/1001/1002, `home`, gripper 0.85, **default** flow sampler (no `LAB01_FLOW_*`) |

Rollout: [`configs/rollout_lingbot_vla_v2.yaml`](configs/rollout_lingbot_vla_v2.yaml).
Scripts: `setup_lingbot_vla_v2_side_tree.cmd`, `eval_lingbot_vla_v2_home20.cmd`.

## Required patch (before any eval)

PR #3967 skips unmapped slots when advancing the inverse read offset. With our
layout (`arm` filled → `end` empty → `effector` filled), gripper was read from
canonical dim **14** instead of **28**. `setup_lingbot_vla_v2_side_tree.cmd`
applies [`patches/lerobot-vla-v2-inverse-slot-offset.patch`](../../patches/lerobot-vla-v2-inverse-slot-offset.patch).

[#4897](https://github.com/huggingface/lerobot/pull/4897) already fixes the
logic but changes processor config shape; migrating needs checkpoint conversion.
Until then: **#3967 + this patch**.

Without the patch, closed-loop gripper is noise and Home20 success is invalid.

## Ranked score

**25/60** (11/20 + 8/20 + 6/20) on the MI210 20K weights above, patched side
tree, default sampler.

## Training note

Converted checkpoints embed `optimizer_lr = 1e-5`. Lab runs so far used that
constant LR at global batch 4 unless overridden. Upstream side-tree recipes use
higher LRs — pass flags explicitly if matching them. Formal train `.cmd` is not
in `labs/` yet; ranked reproduction is **eval of the Hub 20K**.
