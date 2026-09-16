# Lab 01 — LingBot-VA extra adaptation (EEF, not joints)

**Status:** specification for delivery. Do **not** treat `train_lingbot_va.cmd` (joints in channels 0–5) as this path. Background and why this is not the VLA-JEPA / FastWAM failure: runbook **§6.4.2**.

This note is the practice contract for a later implementation PR. It does not add converters, IK eval, or a new Hub dataset yet.

## Goal

Fine-tune [`lerobot/lingbot_va_libero_long`](https://huggingface.co/lerobot/lingbot_va_libero_long) on Lab 01 pick-and-place **without changing the teleop collection**. Reuse the existing 50-episode videos and joint trajectories. Change only the **action labels** (and eval command mapping) so they match what LingBot-VA was pretrained to emit: **end-effector pose delta + gripper**, not six `*.pos` joint angles.

Success for this workstream is a closed-loop Lab 01 eval where:

1. The policy’s 7 used channels are interpreted as LIBERO-style EEF + gripper.
2. SimStudio converts those to SO-101 joint `*.pos` (IK) before `send_action`.
3. The stuffed-joint recipe is not cited as LingBot-VA Lab 01 performance.

Classroom demo policy remains ACT 6-D / MolmoAct2 6-D until this adapter actually scores.

## Why extra adaptation (and why not “like FastWAM”)

| | ACT / SmolVLA / Molmo / JEPA / FastWAM | LingBot-VA (`lingbot_va_libero_long`) |
|--|----------------------------------------|----------------------------------------|
| Native action | Joint positions (FastWAM: pad 6 joints into 7-DoF **joint** slots) | 30-d layout; LIBERO **uses channels 0–6** as EEF Δ + gripper (`used_action_channel_ids` default `range(7)`) |
| Lab 01 dataset | `action` = 6× `*.pos` | Same parquet today — **wrong language** for this stack |
| Fair fine-tune | Direct (JEPA/FastWAM still fail for transfer reasons) | Convert joints → EEF labels; train 7 channels; eval IK |
| Current `train_lingbot_va.cmd` | — | Stuffs 6 joints into 0–5; eval sends them as `*.pos`. Unfair. Campaign 600/8000; eval ImportError. |

Cameras stay second-order: `top→image`, `wrist→image2`, drop `front`. That is key-name alignment (top-down ≠ LIBERO agentview; 640×480 vs policy 128×128). Fix **action language first**. Do not start from a new record session.

## Frozen (do not reuse as the delivery recipe)

- `_env.sh` default `LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS='[0,1,2,3,4,5]'`
- `train_lingbot_va.cmd` / `configs/rollout_lingbot_va.yaml` comments: stuffed joints
- Any checkpoint trained that way

Keep those files as a **negative** historical recipe. New knobs/scripts should be clearly named (`*_eef`, new dataset root).

## Delivery steps

### 1. Freeze LIBERO channel semantics from the checkpoint

Before writing converters, dump `lingbot_va_libero_long` `config.json` / `policy_postprocessor.json` and match LeRobot LIBERO env action (7-DoF). Record in this file (or a short appendix) the exact meaning of channels **0–6**, including:

- translation: world vs EE frame; metres vs centimetres; absolute vs Δ per control step
- rotation: axis-angle / Euler / 6D; radians
- gripper: LIBERO often near-binary open/close; SO-101 is a continuous `gripper.pos` (rad)

Do not guess from FastWAM’s 7th **joint** pad. If RoboTwin (`xyz+quaternion+gripper`, `camera_layout=robotwin_tshape`) looks easier, that is a **different** checkpoint (`lingbot_va_robotwin`) and a separate decision — default remains **LIBERO-Long** to stay on the two-cam `image`/`image2` path already wired in Lab 01.

### 2. Convert the dataset (lab-bound copy)

Reuse:

- `observation.images.camera_top` / `camera_wrist` (and timestamps)
- `observation.state` joint positions as FK input
- episode index, FPS (20 Hz), task string

Do **not** overwrite `./datasets/so101-simstudio-lab01-pnp` or the Hub dataset. Write a new root, e.g. `.tmp/so101-simstudio-lab01-pnp-lingbot-eef` for smokes and `./datasets/so101-simstudio-lab01-pnp-lingbot-eef` only if publishing.

Per frame (or per `action_per_frame` stride — confirm vs `frame_chunk_size=4`, `action_per_frame=4`):

1. FK: MuJoCo `gripperframe` (fingertip site used by Lab 01 eval), not `wrist_site`.
2. Δ pose vs previous executed pose (or vs policy convention from step 1).
3. Map gripper joint → channel 6 using the same open/close convention as LIBERO, documented with a numeric example from episode 0.
4. Pack 7 floats into `action`; pad unused 30-d slots with zeros **only if** the trainer still expects `action_dim=30` while `used_action_channel_ids=[0,1,2,3,4,5,6]`. Prefer slicing to 7-d `output_features` the way `patch_lingbot_action_feature_from_used_channels` already does — **with seven ids, not six**.
5. Rebuild `meta/stats.json` including **q01/q99** (LingBot postprocessor is QUANTILES).

Script lives under `labs/lab01_pnp/` (lab-bound). Smoke: `bash -n`, convert 1–2 episodes, print channel ranges vs LIBERO q01/q99 so units are not 100× off.

### 3. Train (7 used channels)

- `./scripts/install-lingbot-va-deps.sh` (never `pip install -e '.[lingbot_va]'`).
- New `_env` block or overrides, e.g. `LAB01_LINGBOT_EEF_DATASET_ROOT`, `LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS='[0,1,2,3,4,5,6]'`, distinct `LAB01_LINGBOT_OUTPUT` (`lab01_pnp_lingbot_va_eef`).
- Same LoRA / `attn_mode=flex` / UMT5-on-CPU defaults as today’s MI300X recipe unless VRAM forces bs4.
- Rename map unchanged (`top→image`, `wrist→image2`).
- Smoke: 20 steps, batch 1. Reference train: time-box like the 10 h campaign, `save_freq` small enough to eval an early ckpt.

### 4. Eval adapter (EEF → joints)

`eval.py` / robot `send_action` must **not** treat the 7 numbers as `shoulder_pan.pos` … `gripper.pos`.

Per control step:

1. Read policy action (7-d, already unnormalized).
2. Compose Δ onto the current EE pose (same convention as labels).
3. IK to 5 arm joints; map gripper channel to `gripper.pos`.
4. Clip to SO-101 joint limits; send `action_mode: position`.

Reuse the MuJoCo Jacobian path already used for velocity teleop where possible. Log IK fail rate. Add `rollout_lingbot_va_eef.yaml` (do not silently reuse stuffed `rollout_lingbot_va.yaml`).

Headless: `LAB01_MUJOCO_GL=egl`, `LAB01_RENDER_WINDOW=false`. Fix the `cube_over_container` import error that killed the stuffed-recipe eval before quoting any rate.

### 5. Measure and quote

| Gate | Pass |
|------|------|
| Smoke closed-loop | ≥1 episode reaches `reach_8cm` (proves IK+Δ sign) |
| Fair n=10 fixed cube | Report k/10 + stage funnel; compare to stuffed recipe only as a control |
| Fair n=20 full-range | Same spawn box as other Lab 01 VLAs |

If success stays ~0% after IK smoke is healthy, then (and only then) treat remaining gap like JEPA: vision OOD / 50-ep capacity — not “forgot to convert actions.”

## Out of scope for the first implementation PR

- New leader recordings
- Publishing an EEF Hub dataset or weights until a fair eval exists
- FastWAM `lambda_video>0` or more JEPA unfreeze
- Switching the whole lab to EEF teleop

## Suggested file touch list (implementation PR)

| Piece | Likely path |
|-------|-------------|
| Converter | `labs/lab01_pnp/convert_joints_to_lingbot_eef.py` |
| Env knobs | `_env.sh` section 5c eef overrides |
| Train wrapper | `train_lingbot_va_eef.cmd` (leave stuffed `train_lingbot_va.cmd` as historical) |
| Eval YAML | `configs/rollout_lingbot_va_eef.yaml` |
| IK / action translate | `src/simstudio/` (eval or robot helper); keep SO-101 `send_action` joint-native |
| Runbook | § LingBot + §6.4 row: replace *pending stuffed* with **eef-adapted** numbers |

## Knobs to add when implementing

```bash
# sketches only — not wired yet
LAB01_LINGBOT_EEF_DATASET_ROOT=./datasets/so101-simstudio-lab01-pnp-lingbot-eef
LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS='[0,1,2,3,4,5,6]'
LAB01_LINGBOT_OUTPUT=./outputs/train/lab01_pnp_lingbot_va_eef
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_lingbot_va_eef.yaml
```
