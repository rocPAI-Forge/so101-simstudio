# LingBot-VLA 2.0 adaptation for SO-101 (Lab 01)

Status: experimental overnight scaffold. **Not** the same pipeline as
[`lingbot_va_eef_adapt.md`](lingbot_va_eef_adapt.md).

| | LingBot-**VA** | LingBot-**VLA 2.0** |
|---|---|---|
| Hub base | `lerobot/lingbot_va_*` | [`lerobot/lingbot_vla_v2_base`](https://huggingface.co/lerobot/lingbot_vla_v2_base) |
| Action language | LIBERO 7-D ΔEEF | **55-D slots** (joints and/or EEF quat, …) |
| Pin LeRobot `30da8e68` / 0.6.0 | has `lingbot_va` | **no** `lingbot_vla_v2` |
| 193 EEF derived Hub set | yes | **not** drop-in |

## Chosen Lab 01 path (joint slots)

Prefer **joint-space slots** for SO-101 so we reuse the published 193 joint
dataset without LIBERO FK→ΔEEF remapping:

- `arm.position` ← `observation.state` / `action` dims `[0:5]`
- `effector.position` ← dim `[5:6]`
- cameras: `camera_top` → `camera_top`, `camera_wrist` → `camera_wrist_left`

Slot file: [`configs/lingbot_vla_v2_so101_slots.json`](configs/lingbot_vla_v2_so101_slots.json).

If we later switch to `end.position` (XYZ+quat), FK labels + eval IK would be
required again — different from LIBERO Δ scales.

## Runtime gate

Policy code lives upstream in Hugging Face LeRobot PR
[#3967](https://github.com/huggingface/lerobot/pull/3967). Overnight experiment
uses a **side checkout** (`/tmp/lerobot_vla_v2_src` @ `pr-3967`) and a dedicated
venv — **do not** commit into the pinned `lerobot/` submodule.

### Required side-tree patch: inverse slot offsets

PR #3967 (head `543f4207`) has a bug in
`LingbotVLAV2InverseSlotMappingProcessorStep.action()`. When it advances the
canonical read offset, it skips slots that have no mapping. Our layout is
`arm.position` (14, filled), then `end.position` (14, **unfilled**), then
`effector.position` (2, filled). So raw gripper dim 5 was read from canonical
dim **14** (`end.position[0]`, never supervised) instead of dim **28**. Dim 28
is where the forward slot step writes the gripper and where the loss trains it.

Every V2 rollout before 2026-10-21 therefore sent an unsupervised channel as
the gripper command. That channel is white noise around the dataset mean
(≈0.55): ~50 close crossings per episode, the jaw settles near q≈0.56, and the
cube is never lifted. Arm dims 0–4 were not affected. Apply the patch to the
side checkout before any eval:

```bash
git -C /tmp/lerobot_vla_v2_src apply "$PWD/patches/lerobot-vla-v2-inverse-slot-offset.patch"
```

Upstream status (2026-10-22): #3967 was closed on 2026-10-09 and superseded by
[#4897](https://github.com/huggingface/lerobot/pull/4897) (head `de61f262`). Its
rewritten inverse step advances the offset for every canonical joint, so it does
not have this bug. The patch is only needed on the #3967 head we run. Moving the
side tree to #4897 would drop it, but the processor config format changed there
(`action_spans` / `canonical_joints`), so our checkpoints would need their
processor configs converted first.

Teacher-forced check on the MI210 20K checkpoint (expert frames, default
sampler): 0.5 s before the expert close, gripper MAE drops from 0.55 to 0.04
with the patch. The model had learned the gripper; only the read-back was wrong.
Probe scripts: `.tmp/probe_v2_grasp_open_loop.py`, `.tmp/probe_v2_sampling_variants.py`.

**All closed-loop V2 numbers below that predate the patch are invalid** for
`close` / `lift` / success. `reach_8cm` / `near_4cm` are arm-only and remain
informative.

## Post-patch results (2026-10-21 → 22)

MI210 20K (`outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000`,
local, not on the Hub; copied out of `/tmp` with `training_state/`), patched side
tree, Home20 protocol (`home`, gripper 0.85, 60 s, EGL):

| Sampler (`LAB01_FLOW_NOISE_SCALE` / `LAB01_FLOW_NUM_STEPS`) | Seeds 1000 / 1001 / 1002 | reach_8cm → near_4cm → close → lift_2cm → over_box (/60) |
|---|---|---|
| **Default (noise ×1, 10 steps) — official / fair** | 11 / 8 / 6 = **25/60** | 60 → 59 → 56 → 28 → 27 |
| 0 / 10 (eval-time note; optimistic) | 19 / 20 / 19 = **58/60** | 60 → 60 → 60 → 58 → 58 |
| 0 / 3 | 19 / 18 / 18 = **55/60** | 60 → 60 → 60 → 58 → 55 |
| 0.25 / 3 | 19 / 17 / 18 = **54/60** | 60 → 60 → 60 → 57 → 55 |

Logs: `outputs/eval/lingbot_vla_v2_mi210_unfreeze20k_slotfix_home20_g085_seed100{0,1,2}.log`
(default) and `..._slotfix_noise<scale>_steps<n>_home20_g085_seed100{0,1,2}.log`.
The official Home20 cell is default-sampler **25/60** (same fair bar as other
rows on [`home20.md`](../lab01_pnp_agent/home20.md)). The low-noise rows are an
eval-time setting selected after looking at Home20 (seed-1000 sweep, then
three-seed totals); they are optimistic and not cross-model comparable — other
flow-matching policies on that page were not retuned this way. With noise ×0
the seed no longer drives the sampler, so the three seeds are near-replicates,
the same situation as ACT.

**Fair ranking score:** default sampler **25/60** (no `LAB01_FLOW_*` overrides).
**Annotated sampler note (not ranked):** `LAB01_FLOW_NOISE_SCALE=0
LAB01_FLOW_NUM_STEPS=10` → **58/60**. It has the highest total and the cleanest
failure map among the sweeps. Its two failures are single missed lifts on
different spawns (spawn 4 on seed 1002 after four closes; spawn 5 on seed
1000). Spawn 0 and spawn 1 pass on all three seeds, so the spawn-1
hover-until-timeout stall seen under 0 / 3 (all three seeds) is gone. Under
0 / 3 the failures are spawn 0 (no lift, two seeds) and spawn 1 (stall, three
seeds). Under 0.25 / 3 they spread over spawns 0, 1, 5 (twice) and 7 (twice,
both after lifting the cube). Spawns 0 and 5 sit at
x = 0.335 m, the farthest cubes in the list; spawn 5 is the only one that fails
under all three low-noise samplers. There is no V2 eval `.cmd` in `labs/lab01_pnp/`
yet (Home20 runs go through `.tmp/eval_lingbot_vla_v2_home20_campaign.sh`), so set
the two env vars on the eval command when reproducing the annotated row.

The episode details below are from seed 1000. With the default sampler, eight
of the nine failures close on air (jaw force ≈0 at the first close), and the
ninth never closes. The misses are mixed: too far out (`normal_mm` up to 33), too high
(`dz_mm` up to +21), or off to the side (`tangent_mm` up to 17). Five of the
eleven successes also missed the first close and re-grasped. With zero noise,
the first-close `normal_mm` stays at 11.8–18.9 and `dz_mm` at −10.1 to +0.8.
Only one first close touched nothing, and that episode re-grasped and succeeded.
The one failure lifted the cube on the second close, then hovered at 14 cm
until the timeout.

Teacher-forced arm error explains the gap (EE mm, 48 expert contexts; arm slots
were never affected by the bug):

| Checkpoint / sampler | EE error, whole chunk | Seed spread at close |
|---|---|---|
| 2K, default | 30.5 | 20.7 |
| 10K, default | 19.5 | 14.4 |
| 20K, default | 18.1 | 10.4 |
| 20K, noise ×1, 3 steps | 6.9–9.2 | — |
| 20K, noise ×0, 3 steps | 3.3–7.1 | — |

**Sampler sweep (2026-10-22), seed 1000, same 20K checkpoint**
(`.tmp/v2_sampler_sweep.sh`; tag `mi210_unfreeze20k_slotfix_noise<scale>_steps<n>`):

| Noise × | Steps | Success | lift_2cm | Failed spawns |
|---|---|---|---|---|
| 1 | 10 (default) | 11/20 | 11 | 0 1 3 5 7 8 14 15 17 |
| 1 | 3 | 15/20 | 15 | 0 5 8 10 14 |
| 0.5 | 10 | 15/20 | 16 | 0 4 5 7 8 |
| 0.5 | 3 | 17/20 | 18 | 0 5 7 |
| 0.25 | 3 | **19/20** | 19 | 5 |
| 0 | 1 | 13/20 | 15 | 3 5 6 7 11 14 15 |
| 0 | 3 | **19/20** | 20 | 1 |
| 0 | 10 | **19/20** | 19 | 5 |

Less noise helps monotonically down to ≤0.25. One step (the plain conditional
mean) is worse: 13/20, with drops after lift. Three or more steps are needed. The
spawn-1 stall under (0, 3) does not appear under (0, 10) or (0.25, 3), so it is
a property of that exact deterministic trajectory, not of zero noise in general.
Seeds 1001/1002 for (0.25, 3) and (0, 10) are in the three-seed table above.

At 20K the default sampler also has a mean +14 mm \(z\) bias at the expert close
frame. Its samples close before the descent ends. The error is still
falling with steps, so the learned action distribution is wider than the demos.
Few steps from low noise lands near the conditional mean instead.

## Dataset

Joint demos (reuse):
`alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`

EEF-derived (VA only):
`alexhegit/so101-simstudio-lab01-pnp-agent193-lingbot-eef`

## Campaign notes

Filled in as overnight train/eval progress:

| Item | Status |
|------|--------|
| Side checkout PR #3967 | done (`/tmp/lerobot_vla_v2_src` @ `pr-3967`; `/tmp` patches only) |
| Dedicated venv / smoke import | done (`/tmp/venv-vla-v2-cuda`, torch `2.11.0+cu128`) |
| Smoke train steps | **OK** 2 steps / 4×A30 FSDP |
| Reference finetune | **OK** 5000 steps FSDP → `/tmp/lab01_lingbot_vla_v2_agent193` (ckpts 001000…005000) |
| Fixed-cube (demo pose) | **0/1** — reach/near/close yes, lift 0 (`lingbot_vla_v2_fixed_cube_1ep.log`) |
| Home20 seed 1000 @5K | **0/20** — reach 20/20, near 15/20, close 13/20, lift 0 |
| Home20 seed 1001 @5K | **0/20** — reach 20/20, near 17/20, close 15/20, lift 0 |
| Home20 seed 1002 @5K | **0/20** — reach 20/20, near 16/20, close 14/20, lift 0 |
| Home20 total @5K | **0/20 + 0/20 + 0/20 = 0/60** (close 13+15+14=42/60; lift 0) |

Eval: `PYTHONPATH=/tmp/lerobot_vla_v2_src/src` + `/tmp` `types.py` shim; wrapper
`.tmp/eval_lingbot_vla_v2_home20.sh`. **Do not** drop the policy into pinned `lerobot/`.

**Verdict (superseded):** the 5K "never lifts" result was measured with the broken
gripper read-back (see the side-tree patch above). Train log:
`/tmp/lab01_overnight_logs/v2_train_20261017_210446.log`.

## Improve wave (post-5K → 20K)

Pre-patch Home20 on the MI210 20K checkpoint was **0/60** (close high, lift
0). That looked like an empty-grasp / under-training problem; continuations and
scripted DAgger on top of the 193 demos were tried while the inverse-slot bug
was still unknown. After the patch, the **same** 20K weights on the **same**
193 follow2 set score default-sampler **25/60** (official) — so the lift gap
was gripper read-back, not missing recoveries. All V2 DAgger collect/finetune
scripts, env knobs, and local DAgger trees were removed; do not revive them on
pre-patch rollouts.

Issue [#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19).

## Training preset caveat

The converted checkpoint embeds `optimizer_lr = scheduler_decay_lr = 1e-5`, and
none of our train commands override it. So every V2 run so far trained at a
**constant 1e-5** with global batch 4. The A30 5K run plus the MI210 20K run
saw 100K samples, 0.76 epoch of the 130,835 frames. Upstream recipes in the side checkout are much stronger: the
real-robot recipe uses constant 5e-5 at global batch 3 for 40K steps, and the
RoboTwin recipe uses 1e-4 → 5e-5 at global batch 128 for 50K steps. The README
warns to pass these flags explicitly.
