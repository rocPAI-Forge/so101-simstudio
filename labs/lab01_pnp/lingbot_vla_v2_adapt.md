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
side checkout before any eval or DAgger collect:

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

MI210 20K (`/tmp/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000`),
patched side tree, Home20 protocol (`home`, gripper 0.85, 60 s, EGL):

| Sampler | Seeds 1000 / 1001 / 1002 | reach_8cm → near_4cm → close → lift_2cm → over_box (/60) |
|---|---|---|
| Default (noise ×1, 10 steps) | 11 / 8 / 6 = **25/60** | 60 → 59 → 56 → 28 → 27 |
| `LAB01_FLOW_NOISE_SCALE=0 LAB01_FLOW_NUM_STEPS=3` | 19 / 18 / 18 = **55/60** | 60 → 60 → 60 → 58 → 55 |

Logs: `outputs/eval/lingbot_vla_v2_mi210_unfreeze20k_slotfix_home20_g085_seed100{0,1,2}.log`
and `..._slotfix_noise0_steps3_home20_g085_seed100{0,1,2}.log`. Both rows are in
[`home20.md`](../lab01_pnp_agent/home20.md). With noise ×0 the seed no longer
drives the sampler, so the three seeds are near-replicates, the same situation as
ACT. The zero-noise row is an eval-time setting; the other policies on that page
used their default inference.

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
Seeds 1001/1002 for (0.25, 3) and (0, 10) are running.

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

## Improve wave (post-5K → 20K → DAgger)

Everything in this section was run **without** the inverse-slot patch. The
"empty grasp" diagnosis and the DAgger handovers were driven by the noise
gripper channel. The DAgger expert suffixes are still valid demos; the policy
prefixes were rolled out with the broken gripper.

**Diagnosis (pre-patch):**

- Expert 193: **167/193** show EE \(z\) rise \(>2\,\mathrm{cm}\) after first close — demos contain lift.
- Home20 first close (MI210 20K, n≈59): `normal_mm` p50 ≈ **27 mm** (standoff),
  `max_min_jaw_n` p90 ≈ **0 N** (empty grasp), `dz_mm` p50 ≈ 10 mm (height secondary).
  Report: `.tmp/v2_close_gap_diag/report.json`. Tracking: issue [#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19).

**Tried:**

| Attempt | Result |
|---------|--------|
| `train_expert_only=false` on 4×A30 | **OOM** (~23.5 GB/GPU) |
| Continue 5K→+20K expert-only (4×A30) | Home20 **0/60** (close↑, lift=0) |
| Continue 5K→+20K full unfreeze (4×MI210) | Home20 **0/60** (close≈59/60, lift=0); ~28–32 GB/GPU |
| More BC alone | **not** enough — need close→lift distribution |

**Next (in progress):** scripted full DAgger on MI210 20K failure states.

Close→handover uses **observed** `gripper.pos` (not the chattering V2 command)
plus cumulative near-cube closed ticks (`min_dist`≲6 cm, soft ≤0.35 / hard ≤0.15,
~20 ticks). Expert `_prepare_regrasp` lifts straight up first when EE is low
(V2 empty-grasp basin). Smoke: `failed_close` → expert place → `.tmp/...-v2dagger`.

```bash
# Collect 30 recoveries into .tmp (does not touch Hub follow2)
./labs/lab01_pnp/collect_v2_dagger.cmd
# Smoke one episode:
LAB01_V2_DAGGER_TARGET=1 LAB01_V2_DAGGER_MAX_ATTEMPTS=10 ./labs/lab01_pnp/collect_v2_dagger.cmd

# Finetune from MI210 20K on follow2+dagger root
./labs/lab01_pnp/train_lingbot_vla_v2_dagger.cmd
```


Gate: fixed-cube / seed1000 must show `lift_2cm` before full Home20 ranking.

**DAgger 10K result (gate failed, not ranked):** 30 recoveries (222 episodes /
153,484 frames), expert-only finetune from MI210 20K, 4×MI210, ~6 h. Home20
seed1000 (EGL, `home`, gripper 0.85): **0/20**, funnel `reach_8cm=20 →
near_4cm=8 → close_after_open=8 → lift_2cm=0`. This is worse than 20K
(`near_4cm=19`, `close=19`): the model approaches the cube less often, and it
still never lifts. The cube rises ≤5 mm at most. Weights
`/tmp/lab01_lingbot_vla_v2_mi210_dagger10k/checkpoints/010000`, log
`outputs/eval/lingbot_vla_v2_mi210_dagger10k_home20_g085_seed1000.log`.

**Full unfreeze + DAgger 4× (gate failed, not ranked):** seed1000 **0/20**,
funnel `reach_8cm=16 → near_4cm=4 → close=4 → lift_2cm=0`. It is worse again.
The more DAgger weight the model gets, the worse it approaches. Likely cause:
`--mode full` records V2's own prefix (approach + chattering close) as action
labels, and that prefix is ~51% of DAgger frames. The model is trained to copy
its own failure as well as the expert recovery. Next fix: train only on the
expert suffix (mask frames before `intervention_start_frame`). Recipe used:

```bash
LAB01_V2_DAGGER_TRAIN_EXPERT_ONLY=false \
LAB01_V2_DAGGER_UPWEIGHT_LAST=29 LAB01_V2_DAGGER_UPWEIGHT_REPEAT=4 \
LAB01_V2_DAGGER_TRAIN_OUTPUT=/tmp/lab01_lingbot_vla_v2_mi210_dagger_unfreeze_up4_10k \
  ./labs/lab01_pnp/train_lingbot_vla_v2_dagger.cmd
```

Both DAgger gates above also loaded clobbered eval stats (next section), so
their `near_4cm` drop is confounded too. Clean-stats re-runs with the patch are
running (`.tmp/v2_dagger_regates_cleanstats.sh`).

**Suffix-only + 4× (first post-patch DAgger gate):** trained before the patch
(training never calls the inverse step), gated after it. Same recipe as the
full-unfreeze 4× run, plus
`LAB01_V2_DAGGER_SKIP_PREFIX_JSON=.tmp/datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-v2dagger/dagger_handover.json`,
so DAgger episodes are sampled only from `intervention_start_frame` on. Weights
`/tmp/lab01_lingbot_vla_v2_mi210_dagger_suffix_up4_10k/checkpoints/010000`.
Seed 1000, default sampler: **12/20**, funnel `20 → 20 → 17 → 13 → 12`, against
11/20 for the 20K it started from. Two failures show a mode the 20K does not
have: the jaw closes on the cube (~40 N), then reopens within the close window
and drops it about 2 cm up. The recoveries were collected while V2's gripper was
noise, so their prefix states are not ones the patched policy visits.

**All DAgger checkpoints, patched + clean stats, seed 1000:**

| Checkpoint | Default sampler | Noise ×0, 3 steps |
|---|---|---|
| 20K (start point, no DAgger) | 11/20 | **19/20** |
| DAgger expert-only 10K | 7/20 (close 17, lift 11) | — |
| DAgger full unfreeze 4× | 0/20 (close 4, lift 0) | — |
| DAgger suffix-only 4× | 12/20 | 7/20 (close 16, lift 7) |

Every DAgger finetune is worse than the 20K under the sampler that matters. The
suffix-only run's default-sampler 12/20 is within noise of 11/20, and it falls
from 19/20 to 7/20 at zero noise. This DAgger set is retired. If DAgger is
needed again, re-collect it with the patched policy.

## Dataset integrity incident (2026-10-20 → 21)

`collect_v2_dagger.cmd` used to clone the base with `cp -al` (hardlinks). The
recorder rewrites `meta/`, `data/` and `spawns.jsonl` in place, so DAgger
episodes and stats leaked into the local Hub tree
`datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`: `meta/info.json`
(claimed 223 episodes), `meta/stats.json`, both `data/` and `meta/episodes/`
parquets, and `spawns.jsonl`. Videos were not touched.

- Restored all seven files from the Hub on 2026-10-21. They are byte-identical
  to a fresh Hub download, and `info.json` again says 193 episodes / 130,835
  frames. Clobbered `stats.json` / `info.json` kept in
  `.tmp/follow2_clobbered_backup/`.
- The DAgger clone no longer shares inodes with the base, and the cmd now does
  a full copy.
- Eval patches `observation.state` normalizer stats from the base tree. The two
  DAgger gates (2026-10-21 08:47 and 17:23) therefore ran with state stats off
  by up to 0.085σ (wrist_roll / shoulder_pan). Checkpoint stats were never
  affected; they equal the Hub stats.

## Training preset caveat

The converted checkpoint embeds `optimizer_lr = scheduler_decay_lr = 1e-5`, and
none of our train commands override it. So every V2 run so far trained at a
**constant 1e-5** with global batch 4. The A30 5K run plus the MI210 20K run
saw 100K samples, 0.76 epoch of the 130,835 frames. Upstream recipes in the side checkout are much stronger: the
real-robot recipe uses constant 5e-5 at global batch 3 for 40K steps, and the
RoboTwin recipe uses 1e-4 → 5e-5 at global batch 128 for 50K steps. The README
warns to pass these flags explicitly.
