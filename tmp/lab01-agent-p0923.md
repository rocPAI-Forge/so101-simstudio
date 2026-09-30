# Lab 01 agent-yaw campaign — 2026-09-23

**Who this is for:** any Agent on **DORobot** (`/root/so101-simstudio`) after `.43` went offline.  
**Operator:** user will drop in irregularly; **do not wait**. Resume from §Status and run `tmp/lab01-agent-p0923.cmd`.  
**Dataset:** `so101-simstudio-lab01-pnp-agent-yaw` (100 successful yaw-aware scripted episodes, LeRobot v3, 45654 frames). **Do not mix** with teleop `…-pnp` or fixed-wrist `…-pnp-agent`.

## Status (living — update after every stage)

| Field | Value |
|---|---|
| Last update | 2026-09-27 10:50 CST — short ACT plan in `tmp/lab01-act-yaw-v2-p0927.md` (probe, 80 ep, 40K, home n=20). Do not resume GR00T. |
| Driver | `tmp/lab01-agent-p0923.cmd` **do not run** until GR00T is explicitly requested |
| Marker dir | `tmp/p0923_done/` |
| Current stage | **paused before 4b** — next work is ACT grasp robustness (yaw-v2 plan), not GR00T eval |
| Blockers | none |

## ACT ablation p0926 (ckpt `lab01_pnp_act_agent_yaw/080000`, OSMesa, tick-budgeted 60 s)

Driver `tmp/act-yaw-10h-p0926.cmd`; logs `/mnt/doscratch/outputs/eval/act_yaw_ablation_p0926/`.

| reset_arm | n_action_steps | n | Success | reach_8cm | near_4cm | close | lift_2cm |
|---|---|---|---|---|---|---|---|
| follow (baseline) | 50 | 50 | **9/50 (18%)** | 34/50 | 33/50 | 31/50 | 8/50 |
| home | 50 | 20 | **3/20 (15%)** | 20/20 | 20/20 | 20/20 | 4/20 |
| home | 10 | 20 | 0/20 | 20/20 | 2/20 | 0/20 | 0/20 |
| follow | 10 | 10 | 0/10 | 7/10 | 3/10 | 1/10 | 1/10 |

Reading: `home` start fixes approach (all 20 reach and close) — the follow-mode approach losses were start-pose OOD. Success does not rise because lift-after-close is ~20–26% in both starts. Shorter chunks (10) break the final approach. Bottleneck = grasp robustness of the scripted demos, not start pose, not chunk length. Next: perturbation-tolerance sweep of the yaw expert grasp, then agent-yaw-v2 with jittered starts + regrasp recovery.

Stage markers (touch these files when a stage is done):

```
tmp/p0923_done/1a_act_eval
tmp/p0923_done/1b_act_hf
tmp/p0923_done/2a_smolvla_train
tmp/p0923_done/2b_smolvla_eval
tmp/p0923_done/2c_smolvla_hf
tmp/p0923_done/3a_jepa_train
tmp/p0923_done/3b_jepa_eval
tmp/p0923_done/3c_jepa_hf
tmp/p0923_done/4a_groot_train
tmp/p0923_done/4b_groot_eval
tmp/p0923_done/4c_groot_hf
tmp/p0923_done/5_analysis
```

Fill §Results as numbers land. Do not invent success rates.

---

## Hard rules

1. Work **only on DORobot**. Paths below are DORobot paths.
2. **Never** overwrite teleop Hub repos (`…-pnp-act`, `…-pnp-smolvla`, `…-pnp-vla-jepa`, `…-pnp-groot-n17`). Yaw models get **`-agent-yaw` suffix**.
3. **Never** resume `lab01_pnp_act_agent/080000` (fixed-wrist) or teleop ACT onto yaw data.
4. **Never** mix yaw demos with teleop or fixed-wrist as the train set.
5. Headless eval: `LAB01_MUJOCO_GL=egl` first; OSMesa if EGL dies. Quote **ticks**, GL backend, `reset_arm`, `n_action_steps`. `eval.py` uses 20 Hz × `episode_time_s` ticks (60 s → **1200 ticks**), not wall-clock — OSMesa is slow but valid.
6. GR00T: **N1.7 only**, `new_embodiment`, **keep 10K**. Do not continue 30K / vis-unfreeze (those recipes hurt on teleop).
7. JEPA: use the **teleop keeper recipe** (freeze Qwen, WM off, chunk 30, 6-D, 10K). Do not unfreeze Qwen as the default.
8. Push with `LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co` (official Hub, not the mirror).
9. `hf auth` must already work on DORobot as `alexhegit`. If upload fails on auth, log it in §Status and continue remaining local train/eval.

---

## Paths

| Role | Path |
|---|---|
| Repo | `/root/so101-simstudio` |
| Venv | `/root/so101-simstudio/.venv-rocm` |
| Dataset | `/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw` |
| ACT ckpt (done) | `/mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw/checkpoints/080000/pretrained_model` |
| SmolVLA out | `/mnt/doscratch/outputs/train/lab01_pnp_smolvla_agent_yaw` |
| JEPA out | `/mnt/doscratch/outputs/train/lab01_pnp_vla_jepa_agent_yaw` |
| GR00T out | `/mnt/doscratch/outputs/train/lab01_pnp_groot_n17_agent_yaw` |
| Eval logs | `/mnt/doscratch/outputs/eval/` |
| Train logs | `/mnt/doscratch/outputs/train_*_agent_yaw.log` |
| This plan | `/root/so101-simstudio/tmp/lab01-agent-p0923.md` |

Shared eval env:

```bash
cd /root/so101-simstudio
source .venv-rocm/bin/activate
export LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw
export LAB01_DATASET_ROOT=/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw
export LAB01_MUJOCO_GL=egl          # fallback: osmesa
export LAB01_RENDER_WINDOW=false
unset DISPLAY                       # required for egl/osmesa
```

---

## Stage 1 — ACT 80K (already trained)

Train is **done** (2026-09-22 23:35 UTC): 80K steps, bs=64, 15-D, loss **0.015**, dir `lab01_pnp_act_agent_yaw`. Do not retrain.

### 1a Eval n=50 (headless)

Protocol must match teleop ACT 32/50 and fixed-wrist 9/50: `rollout_act.yaml`, `reset_arm=follow`, sync, `n_action_steps=50`, full-range yaw ±45°, **60 s / 1200 ticks**.

```bash
LAB01_POLICY_PATH=/mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw/checkpoints/080000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=/mnt/doscratch/outputs/eval/act_yaw_n100_080000_n50.log \
  ./labs/lab01_pnp/eval.cmd
```

Record: `Eval complete: k/50`, `Stage funnel`, GL backend, mean ticks. Peek 20K GLFW on .43 was **0/20** (reach 8/20) — 80K is the number that matters.

### 1b Hub

Repo: `alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw`  
**Do not** push onto `…-pnp-act`.

```bash
LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co \
  hf repo create alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw --type model --exist-ok
LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co \
  hf upload alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw \
    /mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw/checkpoints/080000/pretrained_model . \
    --repo-type model --commit-message "ACT 15-D 80K bs64 on agent-yaw 100 ep"
```

Optional: short README stating dataset id, 80K / bs64 / n=50 funnel.

---

## Stage 2 — SmolVLA (teleop recipe)

Teleop reference: **bs=64, 50K**, 8 workers, save 10K, `lerobot/smolvla_base`, rename cameras → camera1/2/3, final loss **0.018**, eval **11/50 (22%)** full-range `home` + **RTC**.

```bash
LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw \
LAB01_DATASET_ROOT=/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw \
LAB01_TRAIN_OUTPUT=/mnt/doscratch/outputs/train/lab01_pnp_smolvla_agent_yaw \
LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_STEPS=50000 \
LAB01_TRAIN_SAVE_FREQ=10000 LAB01_TRAIN_NUM_WORKERS=8 \
  ./labs/lab01_pnp/train.cmd
# then copy ./train.log → /mnt/doscratch/outputs/train_smolvla_agent_yaw.log
```

Eval (same YAML as teleop full-range):

```bash
LAB01_POLICY_PATH=/mnt/doscratch/outputs/train/lab01_pnp_smolvla_agent_yaw/checkpoints/050000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_smolvla.yaml \
LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=/mnt/doscratch/outputs/eval/smolvla_yaw_n100_050000_n50.log \
  ./labs/lab01_pnp/eval.cmd
```

Hub: `alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw`  
`LAB01_SMOLVLA_HF_REPO_ID=…` `LAB01_POLICY_PATH=…/050000/pretrained_model` `./labs/lab01_pnp/push_smolvla_model.cmd` with official endpoint. Card may still mention teleop — overwrite README after upload if needed.

Wall ~8 h train.

---

## Stage 3 — VLA-JEPA (teleop *keeper* recipe, not `_env.sh` defaults)

Teleop keeper: 6-D joints, **freeze Qwen**, **world model off**, chunk / `n_action_steps` **30**, bs16 × **10K**, Hub `…-vla-jepa`. Closed-loop **1/10 (10%)** fixed-pose. Unfreeze-Qwen **hurt** — do not use it here.

`_env.sh` defaults are freeze=false / WM=true / chunk=7 — **override**.

```bash
LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw \
LAB01_DATASET_ROOT=/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw \
LAB01_JEPA_OUTPUT=/mnt/doscratch/outputs/train/lab01_pnp_vla_jepa_agent_yaw \
LAB01_STATE_DIM=6 \
LAB01_JEPA_STEPS=10000 LAB01_JEPA_BATCH_SIZE=16 LAB01_JEPA_SAVE_FREQ=2000 \
LAB01_JEPA_CHUNK_SIZE=30 LAB01_JEPA_N_ACTION_STEPS=30 \
LAB01_JEPA_FREEZE_QWEN=true LAB01_JEPA_ENABLE_WORLD_MODEL=false \
  ./labs/lab01_pnp/train_vla_jepa.cmd
```

Eval n=50 **full-range** (user asked 50; teleop JEPA had no honest n=50 full-range — note that in analysis). Keep demo-mean `home_joints` if using `rollout_vla_jepa_demo_start.yaml`, but set cube random:

```bash
LAB01_STATE_DIM=6 \
LAB01_POLICY_PATH=/mnt/doscratch/outputs/train/lab01_pnp_vla_jepa_agent_yaw/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_vla_jepa_demo_start.yaml \
LAB01_N_ACTION_STEPS=30 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=/mnt/doscratch/outputs/eval/jepa_yaw_n100_010000_n50.log \
  ./labs/lab01_pnp/eval.cmd \
    --robot.reset_cube=random \
    --robot.cube_random_x_range='[0.26,0.34]' \
    --robot.cube_random_y_range='[0.165,0.235]' \
    --robot.cube_random_z=0.0125 \
    --robot.cube_random_yaw_range='[-45.0,45.0]'
```

Hub: `alexhegit/so101-simstudio-lab01-pnp-vla-jepa-agent-yaw`

---

## Stage 4 — GR00T N1.7 (teleop *keeper* recipe)

Teleop keeper: `nvidia/GR00T-N1.7-3B`, `new_embodiment`, 6-D, bs=4, **10K**, relative actions, `tune_llm=false`, `tune_visual=false`. Eval **0/10** but `reach_8cm≈9–10/10`. Cont 30K / vis-20K **worse** — do not repeat.

```bash
./scripts/install-groot-deps.sh   # no-op if already installed; never uv pip install 'lerobot[groot]'
LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw \
LAB01_DATASET_ROOT=/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw \
LAB01_GROOT_OUTPUT=/mnt/doscratch/outputs/train/lab01_pnp_groot_n17_agent_yaw \
LAB01_GROOT_STEPS=10000 LAB01_GROOT_BATCH_SIZE=4 LAB01_GROOT_SAVE_FREQ=2000 \
LAB01_GROOT_NUM_WORKERS=4 \
  ./labs/lab01_pnp/train_groot.cmd
```

Eval n=50 full-range (`rollout_groot.yaml` is RTC + `home`):

```bash
LAB01_POLICY_PATH=/mnt/doscratch/outputs/train/lab01_pnp_groot_n17_agent_yaw/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_groot.yaml \
LAB01_N_ACTION_STEPS=16 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=/mnt/doscratch/outputs/eval/groot_yaw_n100_010000_n50.log \
  ./labs/lab01_pnp/eval.cmd
```

If EGL `EGL_BAD_ALLOC` (seen on MI300X for GR00T before): `LAB01_MUJOCO_GL=osmesa`. Still quote ticks.

Hub: `alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent-yaw`

---

## Stage 5 — Analysis (fill after numbers exist)

Compare **success + train loss + funnel**, not loss alone. Fixed-wrist ACT already showed loss 0.013 and **9/50 (18%)**.

### Teleop baseline (do not overwrite)

| Policy | Train | Loss | Eval protocol | Success | Funnel / notes |
|---|---|---|---|---|---|
| ACT 15-D | 50K bs128 | 0.053 | full, follow, sync, n_action=50 | **32/50 (64%)** | Hub `…-pnp-act` |
| SmolVLA | 50K bs64 | 0.018 | full, home, RTC | **11/50 (22%)** | grasp/yaw fails |
| JEPA 6-D BC | 10K freeze-Qwen WM-off chunk30 | ~0.020 | **fixed** pose n=10 | **1/10 (10%)** | no teleop n=50 full-range |
| GR00T N1.7 | 10K bs4 | (see train log) | full, home, RTC n=10 | **0/10** | reach 9–10/10, lift 0 |

### Agent-yaw (this campaign — fill)

| Policy | Train | Loss | Eval protocol | Success | Funnel |
|---|---|---|---|---|---|
| ACT 15-D 80K | 80K bs64 | **0.015** | full, follow, sync, n_action=50, n=50 | TBD 1a | TBD |
| ACT 15-D 20K peek | same job | — | GLFW n=20 on .43 | **0/20** | reach 8/20, lift 0/20 (underfit peek) |
| SmolVLA 50K | TBD | TBD | full, home, RTC n=50 | TBD | TBD |
| JEPA 6-D BC 10K | TBD | TBD | full-range n=50 (stricter than teleop n=10 fixed) | TBD | TBD |
| GR00T N1.7 10K | TBD | TBD | full, home, RTC n=50 | TBD | TBD |

### Extra rows already on disk (context, not this campaign)

| Policy | Data | Success |
|---|---|---|
| ACT 80K fixed-wrist | `…-pnp-agent` 100 ep, wrist locked | **9/50 (18%)**, reach 50/50, lift 14/50 |

Write 1–2 paragraphs: (1) does yaw-aware ACT beat locked-wrist 18% and close the gap to teleop 64%? (2) do VLAs still fail at grasp while ACT clones the new wrist labels? (3) loss vs success — repeat the 0.013≠success lesson if it shows up again.

Append a row idea for `labs/lab01_pnp/lab01_pnp.md` §6.4 but **do not edit the runbook until n=50 numbers are real**.

---

## How another Agent resumes

```bash
ssh DORobot
cd /root/so101-simstudio
source .venv-rocm/bin/activate
ls tmp/p0923_done
tail -50 /mnt/doscratch/outputs/eval/*.log
pgrep -af 'train_act|lerobot-train|eval.cmd'
# then:
nohup ./tmp/lab01-agent-p0923.cmd >> tmp/lab01-agent-p0923.driver.log 2>&1 &
```

The driver **skips** stages whose marker files exist. It must not start a second train into a non-empty output dir (LeRobot will refuse); if a train died mid-run, either `--resume true` for that family or `rm -rf` the incomplete dir after confirming no process holds it.

Expected wall (sequential, MI300X): ACT eval 0.5–8 h (EGL vs OSMesa) + SmolVLA train ~8 h + eval + JEPA ~few h + GR00T 10K ~several h. Finish analysis in this file when the last eval returns.
