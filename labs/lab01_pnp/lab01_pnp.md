# Lab 01 — Leader Teleop Pick-and-Place Dataset

Build a MuJoCo expert dataset with the real SO-101 leader arm, validate it, inspect trajectories, replay in sim, then train ACT / SmolVLA / MolmoAct2 and evaluate in MuJoCo.

> **Dataset v1 deprecated:** `so101-simstudio-pnp` (old wrist camera + original cube/container layout) is no longer used. Lab 01 now uses **`so101-simstudio-lab01-pnp`** only (new wrist cam, swapped spawn layout).

| Item | Value |
|------|-------|
| Scene | `SO101/scenes/simple_pick/scene.xml` |
| Task | Pick up the cube and place it in the box. |
| Dataset root | `./datasets/so101-simstudio-lab01-pnp` |
| Dataset (Hub) | [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp) (50 leader PnP episodes) |
| SmolVLA (Hub) | [alexhegit/so101-simstudio-lab01-pnp-smolvla](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla) (MI300X bs64 @ 50K) |
| ACT (Hub, 15-D state) | [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act) (MI300X 50K, pos+vel+ee) |
| ACT (Hub, 6-D state) | [alexhegit/so101-simstudio-lab01-pnp-act-state6](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-state6) (MI300X 50K, joint pos; real-robot IL layout) |
| GR00T (Hub, N1.7 10K, negative) | [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17) |
| Episodes | 50 |
| Episode length | 90 s max (save early with `N` / `→`) |
| Reset window | 5 s between episodes |
| Record FPS | 20 Hz (leader serial stability) |
| Cameras | `front`, `top`, `wrist` (640×480, wrist aligned Isaac `gripper_cam`) |
| Teleop | Leader arm, 1:1 joint position (`action_mode: position`) |

Skip recording/training if you only need eval: download the Hub dataset + policy (§7; `alexhegit` examples, or your own HF id), then run `./labs/lab01_pnp/eval.cmd` with `LAB01_POLICY_PATH` pointed at the downloaded weights.

**Script defaults** live in [`labs/lab01_pnp/_env.sh`](_env.sh) (centralized, sectioned:
Shared / Record / Train / Train ACT / Train MolmoAct2 / Eval / Hub). Override via environment:

| Variable | Default | Used by |
|----------|---------|---------|
| `LAB01_DATASET_NAME` | `so101-simstudio-lab01-pnp` | record, train, eval |
| `LAB01_DATASET_HF_USER` | `alexhegit` | record, train, eval |
| `LAB01_DATASET_REPO_ID` | `$LAB01_DATASET_HF_USER/$LAB01_DATASET_NAME` | record, train, eval |
| `LAB01_DATASET_ROOT` | `./datasets/$NAME` | record, train, eval |
| `LAB01_NUM_EPISODES` | `50` | record |
| `LAB01_EPISODE_TIME_S` | `90` | record |
| `LAB01_RESET_TIME_S` | `5` | record |
| `LAB01_RECORD_CONFIG` | `configs/so101_mujoco_pick_leader.yaml` | record |
| `LAB01_TRAIN_OUTPUT` / `LAB01_ACT_OUTPUT` | see `_env.sh` | train / train_act |
| `LAB01_TRAIN_STEPS` / `BATCH_SIZE` / `SAVE_FREQ` / `NUM_WORKERS` | `7500` / `4` / `2000` / `4` | train (set for your GPU; see §5) |
| `LAB01_TRAIN_WARMUP` | `500` | train |
| `LAB01_TRAIN_RESUME_FROM` / `RESUME_STEPS` | `007500` / `20000` | train_smolvla_resume |
| `LAB01_ACT_STEPS` / `BATCH_SIZE` / `SAVE_FREQ` | `10000` / `8` / `10000` | train_act (short default; MI300X reference used 50K / 128). `LAB01_ACT_STATE_DIM=6` for joint-pos-only |
| `LAB01_MOLMO_*` | see `_env.sh` §5 | train_molmoact2 (MI300X defaults: bs32 / 10K) |
| `LAB01_POLICY_PATH` | see `_env.sh` (any `pretrained_model/` path) | eval / push_smolvla_model |
| `LAB01_MUJOCO_GL` | `egl` | eval (`egl` / `glfw` / `osmesa`) |
| `LAB01_RENDER_WINDOW` | `false` | eval (`true` with glfw) |
| `LAB01_EVAL_CONFIG` | `labs/lab01_pnp/configs/rollout_smolvla_demo.yaml` | eval spawn yaml (see §6) |
| `LAB01_EVAL_EPISODES` | `10` | eval |
| `LAB01_EVAL_LOG` | `./outputs/eval/eval_${MUJOCO_GL}.log` | eval |
| `LAB01_N_ACTION_STEPS` | empty (checkpoint default) | eval; set e.g. `50` for ACT |
| `LAB01_ACT_HF_REPO_ID` | `alexhegit/so101-simstudio-lab01-pnp-act` | push_act_model_card |
| `LAB01_ACT_STATE6_HF_REPO_ID` | `alexhegit/so101-simstudio-lab01-pnp-act-state6` | push_act_state6_model |
| `LAB01_SMOLVLA_HF_REPO_ID` | `alexhegit/so101-simstudio-lab01-pnp-smolvla` | push_smolvla_model |
| `LAB01_SMOLVLA_CKPT_STEP` | `050000` | push_smolvla_model (when `LAB01_POLICY_PATH` unset) |

Example: `LAB01_DATASET_NAME=my-run ./labs/lab01_pnp/record.cmd`

---

## Prerequisites

```bash
cd ~/Repo/so101-simstudio
make rocm-sync          # once: .venv-rocm + ROCm torch
source .venv-rocm/bin/activate
```

**Leader arm**

- USB port default: `/dev/ttyACM0` (override with `--teleop.port`)
- User in `dialout` group: `sudo usermod -aG dialout $USER` (re-login)
- Feetech SDK (one-time if missing):

  ```bash
  uv pip install --python .venv-rocm/bin/python 'feetech-servo-sdk>=1.0.0,<2.0.0'
  ```

- First connect runs calibration; cache under `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/`

**Scene notes (already in repo)**

- Oblique `top` / `front` cameras and wrist on fixed jaw (commits on `main`)
- Pick cube color: bright green (`rgba="0.15 0.85 0.20 1"`)

---

## 1. Record dataset

Run from an **existing terminal** (do not double-click the script).

```bash
source .venv-rocm/bin/activate
./labs/lab01_pnp/record.cmd
```

Equivalent manual command:

```bash
python -m simstudio.scripts.record \
  --config configs/so101_mujoco_pick_leader.yaml \
  --view_mode mujoco \
  --dataset.root ./datasets/so101-simstudio-lab01-pnp \
  --dataset.repo_id alexhegit/so101-simstudio-lab01-pnp \
  --dataset.num_episodes 50 \
  --dataset.episode_time_s 90 \
  --dataset.reset_time_s 5 \
  --resume false
```

**GUI modes** (same LeRobot v3.0 output):

| Mode | Flag | When to use |
|------|------|-------------|
| MuJoCo | `--view_mode mujoco` (default in `record.cmd`) | Low-latency teleop |
| Rerun | `./labs/lab01_pnp/record.cmd --view_mode rerun` | Multi-camera preview |

**Recording controls** (evdev, focus-independent):

| Key | Action |
|-----|--------|
| `→` / `N` | Save episode, next |
| `←` / `R` | Discard episode, re-record |
| `ESC` / `Q` | Stop session |

**Episode arm reset (`robot.reset_arm`)** — set in the record YAML (Lab 01 uses
`configs/so101_mujoco_pick_leader.yaml`):

| Value | Behavior at episode start | Use when |
|-------|---------------------------|----------|
| **`follow`** | Do **not** teleport the sim follower; keep its current joint pose | **Real leader** teleop (1:1 joint mapping). Avoids a first-frame yank from home to wherever your hand left the leader. |
| **`home`** | Teleport the sim follower to the fixed home pose | **Keyboard / Joy-Con** (no real leader pose to stay aligned with) |

Lab 01 recording is leader-based → **`reset_arm: follow`** (with `reset_cube: random`).
Cube placement is independent (`reset_cube`: `random` / `fixed` / `none`).
See also AGENTS.md → *Episode reset (sim)*.

**Resume after interruption**

```bash
./labs/lab01_pnp/record.cmd --resume true
```

**Fresh re-record**

```bash
rm -rf ./datasets/so101-simstudio-lab01-pnp   # or: rm -rf "$LAB01_DATASET_ROOT"
./labs/lab01_pnp/record.cmd
```

Log: `test.log` at repo root.

---

## 2. Validate dataset

Automated integrity check (frame count, 20 Hz timestamps, action ranges, videos):

```bash
python -m simstudio.scripts.validate_dataset \
  --root ./datasets/so101-simstudio-lab01-pnp
```

Expect `PASSED` with 50 episodes and LeRobot `codebase_version: v3.0`.

Quick metadata:

```bash
python -c "
import json
info = json.load(open('datasets/so101-simstudio-lab01-pnp/meta/info.json'))
print('task:', info.get('single_task'))
print('episodes:', info['total_episodes'], 'frames:', info['total_frames'], 'fps:', info['fps'])
"
```

---

## 3. Inspect / visualize

### Rerun (cameras + action/state curves)

```bash
source .venv-rocm/bin/activate   # ensures rerun CLI on PATH

python -m simstudio.scripts.dataset_viz \
  --repo-id alexhegit/so101-simstudio-lab01-pnp \
  --root ./datasets/so101-simstudio-lab01-pnp \
  --episode 0
```

Export `.rrd` for smoother playback (avoids live gRPC 1 GiB memory drops):

```bash
python -m simstudio.scripts.dataset_viz \
  --repo-id alexhegit/so101-simstudio-lab01-pnp \
  --root ./datasets/so101-simstudio-lab01-pnp \
  --episode 0 \
  --save --output-dir ./outputs/viz

rerun --memory-limit 4GB ./outputs/viz/alexhegit_so101-simstudio-lab01-pnp_episode_0.rrd
```

### MP4 spot-check (smoothest video preview)

```bash
ffplay -autoexit datasets/so101-simstudio-lab01-pnp/videos/observation.images.camera_front/chunk-000/file-000.mp4
```

**Note:** Rerun live mode can look accelerated or jumpy when uncompressed RGB exceeds the viewer memory limit. That is a visualization artifact, not a dataset defect.

---

## 4. Replay in MuJoCo

Replay recorded joint trajectories in the same sim scene.

**Single episode**

```bash
python -m simstudio.scripts.replay \
  --config configs/so101_mujoco_replay.yaml \
  --dataset.root ./datasets/so101-simstudio-lab01-pnp \
  --dataset.repo_id alexhegit/so101-simstudio-lab01-pnp \
  --dataset.episode 0
```

**All episodes**

```bash
python -m simstudio.scripts.replay_multi \
  --config configs/so101_mujoco_replay_multi.yaml \
  --dataset.root ./datasets/so101-simstudio-lab01-pnp \
  --dataset.repo_id alexhegit/so101-simstudio-lab01-pnp \
  --dataset.episodes all
```

---

## 5. Train SmolVLA

Fine-tune `lerobot/smolvla_base` on the lab dataset. Requires validated data and `.venv-rocm` (`make rocm-sync`).

Set **`LAB01_TRAIN_BATCH_SIZE` / `LAB01_TRAIN_STEPS` / `LAB01_TRAIN_SAVE_FREQ` / `LAB01_TRAIN_NUM_WORKERS`** for your GPU VRAM — there is no `igpu`/`mi300x` profile switch in `_env.sh`. Shared output dir: `./outputs/train/lab01_pnp_smolvla/`. **Eval defaults to the 50K checkpoint** from the MI300X reference run below.

**Reference recipes** (measured; copy the env overrides that fit your machine):

| Recipe | GPU | `batch_size` | `steps` | `save_freq` | `num_workers` | Sample updates | VRAM |
|--------|-----|--------------|---------|-------------|---------------|----------------|------|
| Short / laptop | Strix Halo **8060S** | **4** | **7500** | 2000 | 4 | 30k | ~**2.88 GB** |
| Lab reference | Instinct **MI300X** | **64** | **50000** | 10000 | 8 | 3.2M | ~**26 GB** |

`_env.sh` defaults match the short recipe. Large batches on small GPUs will OOM — lower `LAB01_TRAIN_BATCH_SIZE` (try `1`).

```bash
source .venv-rocm/bin/activate
# Defaults (batch 4 / 7500 steps)
./labs/lab01_pnp/train.cmd

# MI300X-scale example
LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_STEPS=50000 \
  LAB01_TRAIN_SAVE_FREQ=10000 LAB01_TRAIN_NUM_WORKERS=8 \
  ./labs/lab01_pnp/train.cmd
```

**Manual equivalent** (short defaults):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.scheduler_warmup_steps=500 \
  --dataset.repo_id=alexhegit/so101-simstudio-lab01-pnp \
  --dataset.root=./datasets/so101-simstudio-lab01-pnp \
  --dataset.video_backend=pyav \
  --output_dir=./outputs/train/lab01_pnp_smolvla \
  --job_name=lab01_pnp_smolvla \
  --rename_map='{"observation.images.camera_top":"observation.images.camera1","observation.images.camera_front":"observation.images.camera2","observation.images.camera_wrist":"observation.images.camera3"}' \
  --batch_size=4 \
  --steps=7500 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=2000
```

MI300X-scale: same command with `--batch_size=64 --steps=50000 --num_workers=8 --save_freq=10000`.

**Camera rename map** (dataset → `smolvla_base`):

| Dataset key | Policy key |
|-------------|------------|
| `observation.images.camera_top` | `observation.images.camera1` |
| `observation.images.camera_front` | `observation.images.camera2` |
| `observation.images.camera_wrist` | `observation.images.camera3` |

`--policy.empty_cameras=1` matches the base model’s fourth padded camera slot.

To record extra scene cams (e.g. `overview`) and train on a **subset** only
(map just those keys; leave others out of `--rename_map`), see
[SO101/scenes/README.md — Cameras](../../SO101/scenes/README.md#cameras-record-vs-teleop-vs-train).

**Tune via env or CLI overrides:**

```bash
LAB01_TRAIN_STEPS=20000 LAB01_TRAIN_BATCH_SIZE=1 ./labs/lab01_pnp/train.cmd
./labs/lab01_pnp/train.cmd --steps 20000 --batch_size 1
./labs/lab01_pnp/train.cmd --resume true
```

Log: `train.log` at repo root. Checkpoints: `./outputs/train/lab01_pnp_smolvla/checkpoints/`.

**Resume** from a saved checkpoint (needs `pretrained_model/` + `training_state/`):

```bash
# Example: 7500 → 20000
LAB01_TRAIN_RESUME_FROM=007500 LAB01_TRAIN_RESUME_STEPS=20000 \
  ./labs/lab01_pnp/train_smolvla_resume.cmd

# Example: 50000 → 100000 (keep batch/save_freq consistent with the run)
LAB01_TRAIN_RESUME_FROM=050000 LAB01_TRAIN_RESUME_STEPS=100000 \
  LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_SAVE_FREQ=10000 \
  ./labs/lab01_pnp/train_smolvla_resume.cmd
```

**Notes**

- First run downloads HF weights; needs network.
- OOM: lower `LAB01_TRAIN_BATCH_SIZE` (try `1`).
- Closed-loop eval: §6 (`eval.cmd`). Default checkpoint path is the **50K** reference under `checkpoints/050000`.

### Shared training settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base policy | `lerobot/smolvla_base` | Use `--policy.path`, not `pretrained_path` |
| `policy.scheduler_warmup_steps` | 500 | Default in `_env.sh` |
| `policy.empty_cameras` | 1 | Pads fourth camera slot expected by base model |
| `policy.push_to_hub` | false | Avoids HF repo_id requirement |
| `dataset.video_backend` | pyav | Required on ROCm (`torchcodec` fails) |
| Dataset | 50 ep @ 20 Hz | `alexhegit/so101-simstudio-lab01-pnp` |
| Output | `./outputs/train/lab01_pnp_smolvla` | Same path on .43 and DORobot |

### Run 1 — Strix Halo / 8060S (bs4 @ 7.5K)

| Item | Value |
|------|-------|
| Machine | AMD Strix Halo laptop (Ryzen AI MAX+ 395) |
| GPU | AMD Radeon 8060S (iGPU) |
| PyTorch / ROCm | 2.13 / 7.2 (`.venv-rocm`) |
| LeRobot | 0.6.0 (submodule `30da8e68`) |

```bash
source .venv-rocm/bin/activate
./labs/lab01_pnp/train.cmd
# Historical output dir for this run: LAB01_TRAIN_OUTPUT=./outputs/train/lab01_pnp_smolvla_bs4
```

| Run | Platform | batch | steps | wall time | step/s | VRAM | final loss | checkpoint |
|-----|----------|-------|-------|-----------|--------|------|------------|------------|
| 1 | Strix Halo / 8060S | 4 | 7500 | **1h 54m** (2026-08-10 22:55 → 00:49) | ~**1.10** | ~**2.88 GB** | **0.159** (step 7400) | `./outputs/train/lab01_pnp_smolvla_bs4/checkpoints/007500/` |

- **Speed:** ~1.10 step/s (~4 samples/s); `updt_s` ~0.90 s/step.
- **Loss:** plateaued ~**0.15–0.16** from step 6600; last logged **0.159** at 7400.
- **Loss plot:** `labs/lab01_pnp/loss_curve_run1.png` / `loss_curve_run1.csv`.
- **Checkpoints:** `002000`, `004000`, `006000`, `007500`, `last → 007500`.
- Short-run eval only; not the default `eval.cmd` checkpoint.

### Run 2 — MI300X (bs64 @ 50K) — current eval checkpoint

| Item | Value |
|------|-------|
| Machine | DORobot |
| GPU | AMD Instinct MI300X (192 GB HBM) |
| Batch / steps | **64 / 50000** (3.2M sample updates; ~16× Run 1) |
| `num_workers` | 8 |
| `save_freq` | 10000 |

```bash
source .venv-rocm/bin/activate
LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_STEPS=50000 \
  LAB01_TRAIN_SAVE_FREQ=10000 LAB01_TRAIN_NUM_WORKERS=8 \
  ./labs/lab01_pnp/train.cmd
```

| Run | Platform | batch | steps | wall time | step/s | VRAM | final loss | checkpoint |
|-----|----------|-------|-------|-----------|--------|------|------------|------------|
| 2 | DORobot / MI300X | 64 | 50000 | **7h 45m** (2026-08-14) | ~**1.79** | ~**26.1 GB** | **0.018** (step 50K) | `./outputs/train/lab01_pnp_smolvla/checkpoints/050000/` |

- **Throughput:** ~112 samples/s; `updt_s` ~0.54 s/step; ~151 epochs over the 50-ep dataset.
- **Checkpoints:** `010000`–`050000`, `last → 050000`. Same path on DORobot and the local .43 machine.
- **Inference path:** `./outputs/train/lab01_pnp_smolvla/checkpoints/050000/pretrained_model/`
- Compare to ACT on the same host: ACT used batch **128** × 50K (~6.4M updates, loss **0.053**, ~17 h). SmolVLA bs64 is half the sample count, ~26 GB of 192 GB.

### ACT training (Run 1) — v1 dataset only (historical)

> Trained on deprecated `so101-simstudio-pnp` (old wrist cam + layout). Do not use for lab01-pnp eval; retrain on `so101-simstudio-lab01-pnp` after recording.

```bash
source .venv-rocm/bin/activate
# Defaults in _env.sh: LAB01_ACT_STEPS=10000, BATCH_SIZE=8, SAVE_FREQ=10000
./labs/lab01_pnp/train_act.cmd
# MI300X-scale reference (lab01-pnp): override to batch 128 / 50K — see table below
# LAB01_ACT_BATCH_SIZE=128 LAB01_ACT_STEPS=50000 LAB01_ACT_SAVE_FREQ=10000 \
#   ./labs/lab01_pnp/train_act.cmd
```

| Run | Platform | batch | steps | wall time | step/s | VRAM | final loss | checkpoint |
|-----|----------|-------|-------|-----------|--------|------|------------|------------|
| 1 | Strix Halo / 8060S iGPU | 8 | 10000 | **~5h 30m** | ~**0.50** | ~**5.8 GB** | **0.249** (step 10000) | `./outputs/train/lab01_pnp_act/checkpoints/010000/` |
| 2 (resume) | Strix Halo / 8060S iGPU | 8 | 30000 | **~10h 50m** (10k→30k) | ~**0.51** | ~**5.8 GB** | **0.130** (step 30000) | `./outputs/train/lab01_pnp_act/checkpoints/030000/` |

Also saved: Run 1 `005000`; Run 2 `020000`, `030000`, `last → 030000`. Logs: `train_act.log`, `train_act_run2.log`.

**ACT eval (20 episodes, headless, `reset_arm: follow`):** checkpoint `030000` → **7/20 (35%)**. Successful episodes place the cube in the container (~0.30, 0.19–0.21).

### ACT training (lab01-pnp, MI300X 50K) — current 15-D reference

```bash
LAB01_ACT_BATCH_SIZE=128 LAB01_ACT_STEPS=50000 LAB01_ACT_SAVE_FREQ=10000 \
  ./labs/lab01_pnp/train_act.cmd
```

| Run | Platform | batch | steps | state | final loss | checkpoint |
|-----|----------|-------|-------|-------|------------|------------|
| lab01-pnp | DORobot / MI300X | 128 | 50000 | **15** (pos+vel+ee) | **0.053** | `./outputs/train/lab01_pnp_act/checkpoints/050000/` |

**Loss curve** (from `train_act.log`, not eval): `labs/lab01_pnp/loss_curve_act_mi300x_50k.png` / `.csv`.

### ACT 6-D retrain (joint positions) — real-robot IL / sim2real

SimStudio records **15-D** `observation.state` (6 joint `.pos` + 6 `.vel` + 3 EE XYZ). That is a **sim-only** layout. Official real LeRobot SO-101 (`so_follower`) is **6-D joint positions** (+ 6-D joint actions). Training a policy on 15-D sim state means the network expects channels the real arm does not provide, which blocks mixing real IL data and adds a sim2real feature mismatch.

This run trains ACT on the **first 6 dims only** so the policy input matches real-robot IL. Goals:

- Same observation semantics as the real follower (joint `.pos`).
- Easier **sim2real** and later merge with real datasets.
- EE is FK of the joints; vel is largely redundant at `n_obs_steps=1`. Extra dims are not assumed to help closed-loop success.

**Not solved by 6-D alone:** radians vs degrees, gripper scale. Align units before mixing sim and real parquet.

**Do not crop a second dataset.** Videos dominate Lab 01 (~1.17 GB); the extra 9 floats are ~0.73 MB uncompressed (~0.07% of the store). Keep the Hub 15-D dataset; slice at train load (`LAB01_ACT_STATE_DIM=6`). Writes `lab01_pnp_act_state6` so the 15-D 64% checkpoint is not overwritten.

```bash
# DORobot / MI300X — same 50K / batch 128 schedule as the 15-D reference
LAB01_ACT_STATE_DIM=6 \
  LAB01_ACT_BATCH_SIZE=128 LAB01_ACT_STEPS=50000 LAB01_ACT_SAVE_FREQ=10000 \
  LAB01_ACT_OUTPUT=/mnt/doscratch/outputs/train/lab01_pnp_act_state6 \
  ./labs/lab01_pnp/train_act.cmd
```

| Run | Platform | batch | steps | state | final loss | wall | checkpoint |
|-----|----------|-------|-------|-------|------------|------|------------|
| lab01-pnp 6-D | DORobot / MI300X | 128 | 50000 | **6** (joint `.pos`) | **0.054** | ~17 h | `./outputs/train/lab01_pnp_act_state6/checkpoints/050000/` |

**Hub:** [alexhegit/so101-simstudio-lab01-pnp-act-state6](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-state6) (see §7). Local eval:

```bash
LAB01_POLICY_PATH=./outputs/train/lab01_pnp_act_state6/checkpoints/050000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_MUJOCO_GL=egl LAB01_RENDER_WINDOW=false \
LAB01_EVAL_LOG=./outputs/eval/act_state6_egl_fullrange.log \
  ./labs/lab01_pnp/eval.cmd
```

Same YAML as 15-D ACT (`reset_arm: follow`). LeRobot rollout already feeds joint `.pos`. **§6.4 adds a new row**; do not overwrite the 15-D 64% line.

| | 15-D ACT 50K | 6-D ACT 50K |
|--|----------------|-------------|
| Train loss | 0.053 | 0.054 |
| Full-range eval | **32/50 (64%)** | **29/50 (58%)** |

At n=50 the 6-point gap is within sampling noise: **same level** as 15-D. Use 6-D when the goal is real IL / sim2real alignment, not to raise sim success.

Plot 6-D loss with a distinct tag (do not replace the 15-D curve):

```bash
.venv-rocm/bin/python labs/lab01_pnp/plot_act_loss.py \
  --log train_act_state6.log --steps 50000 --tag mi300x_50k_state6
```

**15-D vs 6-D dataset size:** almost all of Lab 01’s store is **videos** (3×480×640 AV1). Parquet is ~2.5 MB. Train with `LAB01_ACT_STATE_DIM=6` instead of duplicating the dataset.

MolmoAct2 / VLA-JEPA 6-D fine-tunes (same slice) are a separate measurement; do not copy this ACT success rate onto those policies.

Regenerate the 15-D curve after a new 15-D ACT run:

```bash
.venv-rocm/bin/python labs/lab01_pnp/plot_act_loss.py \
  --log train_act.log --steps 50000 --tag mi300x_50k
```

**ACT eval:** measured rates and configs are in §6 (do not treat loss alone as a proxy for pick success).

### MolmoAct2 training (lab01-pnp, MI300X) — DORobot

Fine-tune [`lerobot/MolmoAct2-SO100_101-LeRobot`](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) on the Lab 01 dataset. Entry point matches ACT/SmolVLA: knobs in `_env.sh`, run `train_molmoact2.cmd`.

**Defaults (single MI300X 192 GB HBM):** `batch_size=32`, `steps=10000`, `save_freq=2000`, `num_workers=8`, VLM LoRA + trainable action expert, `chunk_size=30`.

| Choice | Lab 01 setting | Why |
|--------|----------------|-----|
| Base | `MolmoAct2-SO100_101-LeRobot` | SO-101 joint pose + LeRobot processor |
| Cameras | `camera_top`→`cam0`, `camera_wrist`→`cam1` | Base is 2-view; keep warm-start layout |
| Joint frame | signs/offsets = identity | Lab01 actions/state are **radians**; SO100 degree offsets must not apply |
| Quantile stats | auto-augment if `meta/stats.json` lacks `q01`/`q99` | MolmoAct2 default norm |

```bash
source .venv-rocm/bin/activate
# Molmo extras only — never: uv pip install 'lerobot[molmoact2]' (CUDA torch)
./scripts/install-molmoact2-deps.sh

# Dataset already under ./datasets/so101-simstudio-lab01-pnp on DORobot
./labs/lab01_pnp/train_molmoact2.cmd
```

| Recipe | GPU | `batch_size` | `steps` | `save_freq` | `num_workers` | Notes |
|--------|-----|--------------|---------|-------------|---------------|-------|
| Lab / DORobot | Instinct **MI300X** | **32** | **10000** | 2000 | 8 | `_env.sh` defaults |
| Smoke | any | 4 | 500 | 500 | 2 | `LAB01_MOLMO_BATCH_SIZE=4 LAB01_MOLMO_STEPS=500 ...` |

```bash
# Smoke
LAB01_MOLMO_BATCH_SIZE=4 LAB01_MOLMO_STEPS=500 LAB01_MOLMO_NUM_WORKERS=2 \
  LAB01_MOLMO_SAVE_FREQ=500 ./labs/lab01_pnp/train_molmoact2.cmd

# Longer MI300X run
LAB01_MOLMO_STEPS=20000 LAB01_MOLMO_BATCH_SIZE=48 \
  ./labs/lab01_pnp/train_molmoact2.cmd
```

Log: `train_molmoact2.log`. Checkpoints: `./outputs/train/lab01_pnp_molmoact2/checkpoints/`.

**Eval:** `rename_map` is `camera_top→cam0`, `camera_wrist→cam1`. Use `n_action_steps=30` and `inference: sync`.

```bash
# Copy the two Molmo YAML files to .43 if that tree predates them, then:
cd /path/to/so101-simstudio
source .venv-rocm/bin/activate
./scripts/install-molmoact2-deps.sh   # once; skip if peft already imports

LAB01_POLICY_PATH=./outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_molmoact2_demo_fixed.yaml \
LAB01_N_ACTION_STEPS=30 \
LAB01_EVAL_EPISODES=10 \
LAB01_EVAL_LOG=./outputs/eval/molmoact2_fixed.log \
  ./labs/lab01_pnp/eval.cmd

LAB01_POLICY_PATH=./outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_molmoact2.yaml \
LAB01_N_ACTION_STEPS=30 \
LAB01_EVAL_EPISODES=20 \
LAB01_EVAL_LOG=./outputs/eval/molmoact2_full.log \
  ./labs/lab01_pnp/eval.cmd
```

**Deps:** `policy.type=molmoact2` lives in the editable `lerobot/` submodule. Install extras with `./scripts/install-molmoact2-deps.sh` (`transformers` 5.4, `peft`, `scipy`). **Do not** `uv pip install 'lerobot[molmoact2]'` — that extra re-resolves torch onto CUDA and `nvidia-*` wheels. If that already happened: `./scripts/install-molmoact2-deps.sh --repair-torch` (or `make rocm-sync`). `peft` is required when `LAB01_MOLMO_ENABLE_LORA_VLM=true`.

### GR00T N1.7 training (lab01-pnp)

Post-train [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) on the Lab 01 **6-D joint** dataset (`embodiment_tag=new_embodiment`). Joint VLA (pad to `max_action_dim`), not LingBot EEF, not **N1.5**. Spec: [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md).

```bash
source .venv-rocm/bin/activate
./scripts/install-groot-deps.sh          # never: uv pip install 'lerobot[groot]'
./labs/lab01_pnp/train_groot.cmd         # 20 steps / bs1 smoke

# Longer (e.g. MI300X); still measure before calling it a demo policy
LAB01_GROOT_STEPS=10000 LAB01_GROOT_BATCH_SIZE=4 LAB01_GROOT_SAVE_FREQ=2000 \
  ./labs/lab01_pnp/train_groot.cmd

LAB01_POLICY_PATH=./outputs/train/lab01_pnp_groot_n17/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_groot.yaml \
LAB01_EVAL_EPISODES=10 LAB01_MUJOCO_GL=egl LAB01_RENDER_WINDOW=false \
  ./labs/lab01_pnp/eval.cmd
```

**Keep `checkpoints/010000`.** Continuation 30K and visual-unfreeze 20K are worse on n=10 (approach collapses). Do not init from `GR00T-N1.7-LIBERO` / `libero_sim`. Keep `tune_llm=false` and `tune_visual=false` on this 50-ep set. N1.7’s tokenizer wants gated [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B); `train_act` / `eval` default to public `Qwen/Qwen3-VL-2B-Instruct` processor files (`LAB01_GROOT_PROCESSOR_MODEL` to override if you have Cosmos access). Comparison: [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md).

### LingBot-VLA 2.0 (lab01-pnp) — ranked LingBot path

Authoritative Lab 01 LingBot path (issue [#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19)):
joint slots on the 193 follow2 demos, side-tree PR #3967 + inverse-slot patch,
MI210 20K → Home20 **25/60** (default sampler). Same finetune→Home20 shape as
GR00T N1.7 absolute joints. Full contract: [`lingbot_vla_v2_adapt.md`](lingbot_vla_v2_adapt.md).
Knobs: `_env.sh` section 5c (`LAB01_V2_*`). Rollout:
`configs/rollout_lingbot_vla_v2.yaml`.

LingBot-VA (`lingbot_va_*`) stuffed-joint / fair-EEF trials are **out of tree** and
**paused** on SO-101 ([issue #9](https://github.com/rocPAI-Forge/so101-simstudio/issues/9)).
Fair EEF Home20 stayed **0/60** (≈19/60 reach_8cm, ≈2/60 close); the derived EEF
dataset was removed. LIBERO-style ΔEEF matches **7-DoF Cartesian** control better
than SO-101’s 5+1 arm (every 6-DoF increment must IK-project onto a lower-DoF
manifold). Prefer openARM-class dual-arm platforms for a future VA line; keep Lab
01 ranked LingBot on VLA 2.0. “Better fit” is action-language alignment, not a
promise that VA will score well without open-loop / controller checks.

### FastWAM training (lab01-pnp, MI300X) — DORobot

Fine-tune [`lerobot/fastwam_base`](https://huggingface.co/lerobot/fastwam_base) on the same dataset. Knobs: `_env.sh` section 5d; run `train_fastwam.cmd`.

The base model concatenates two 224-wide views into `image_size=[224,448]`, with `action_dim=7` and `proprio_dim=8`. Lab 01 mapping:

| Lab 01 | FastWAM |
|--------|---------|
| `camera_top` | `observation.images.image` |
| `camera_wrist` | `observation.images.image2` |
| `camera_front` | unused |
| 6-D joints + 6-D state | `action_dim=6`, `proprio_dim=6` (`LAB01_STATE_DIM=6`) |

Shape-mismatched action/proprio tensors are reinitialized (FastWAM's cross-embodiment loader). Defaults freeze the ~5B Wan **video** expert (`freeze_video_expert=true`, `lambda_video=0`) and train the action expert with gradient checkpointing so batch 32 fits MI300X.

```bash
source .venv-rocm/bin/activate
./scripts/install-fastwam-deps.sh
./labs/lab01_pnp/train_fastwam.cmd
```

```bash
LAB01_POLICY_PATH=./outputs/train/lab01_pnp_fastwam/checkpoints/020000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_fastwam.yaml \
LAB01_N_ACTION_STEPS=32 \
LAB01_EVAL_EPISODES=20 \
LAB01_RENDER_WINDOW=false \
LAB01_MUJOCO_GL=egl \
LAB01_EVAL_LOG=./outputs/eval/fastwam_egl20.log \
  ./labs/lab01_pnp/eval.cmd
```

Resume from `004000` (do **not** `rm` the output dir). Keep bs4; 4K was ~0.75 epoch on the action head.

```bash
LAB01_FASTWAM_KEEP_OUTPUT=1 \
LAB01_FASTWAM_OUTPUT=/mnt/doscratch/outputs/train/lab01_pnp_fastwam \
LAB01_FASTWAM_LOG=/mnt/doscratch/outputs/train_fastwam_resume20k.log \
LAB01_FASTWAM_STEPS=20000 \
LAB01_FASTWAM_BATCH_SIZE=4 \
LAB01_FASTWAM_SAVE_FREQ=2000 \
LAB01_FASTWAM_NUM_WORKERS=4 \
  ./labs/lab01_pnp/train_fastwam.cmd --resume true
```

Resume 4K→20K on MI300X: **4 h 19 min** remaining steps, last log **loss ~0.111**, **~3.8 epochs** (`smpl:80K`, bs4). Checkpoints every 2K under `/mnt/doscratch/outputs/train/lab01_pnp_fastwam/checkpoints/`.

20K closed-loop is still **0%** (keyboard `home` and demo-mean start). Debug dump (`LAB01_DEBUG_ACTIONS=80`) shows the arm is **not** frozen at home (`shoulder_pan` command range ~0.39 rad in the first 80 ticks) but the cube never leaves spawn (`z≈0.015`). Do not unfreeze the 5B video expert as the next cheap lever until a shorter action-head recipe is tried. On MI300X, if EGL `EGL_BAD_ALLOC`, use `LAB01_MUJOCO_GL=osmesa`.

**ROCm extras:** both policies use `./scripts/install-lerobot-extra-deps.sh`, which pins the current `torch==…+rocm` / `torchvision` via `--constraints` and never re-resolves CUDA wheels. If a CUDA leak already happened, add `--repair-torch`.

### xVLA and DM0.5 (193-episode Home20)

These two fine-tunes are not in the pinned LeRobot v0.6.0 submodule. The published runs used LeRobot `24f15b92b556cb861ca65e989462e23505a701bd`. Clone that commit and set `LAB01_LEROBOT_SRC`. Dataset is the 193-episode follow2 set, not the 50-episode teleop root. Recipes and scores: [`../lab01_pnp_agent/home20.md`](../lab01_pnp_agent/home20.md).

```bash
git clone https://github.com/huggingface/lerobot.git /path/to/lerobot
git -C /path/to/lerobot checkout 24f15b92b556cb861ca65e989462e23505a701bd

# xVLA 10K (published on 4× A30). Defaults in _env.sh section 5f.
LAB01_LEROBOT_SRC=/path/to/lerobot \
LAB01_PYTHON=.venv-cuda/bin/python \
  ./labs/lab01_pnp/train_xvla.cmd

# DM0.5 20K (published on 4× MI210).
LAB01_LEROBOT_SRC=/path/to/lerobot \
  ./labs/lab01_pnp/train_dm05.cmd

# Home20, seeds 1000/1001/1002. Pick a new LAB01_HOME20_TAG.
LAB01_LEROBOT_SRC=/path/to/lerobot \
LAB01_POLICY_PATH=./outputs/train/lab01_xvla_agent193_10k/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_xvla.yaml \
LAB01_N_ACTION_STEPS=30 \
LAB01_HOME20_TAG=xvla_repro \
  ./labs/lab01_pnp/eval_home20.cmd
```

`train_xvla.cmd` rewrites a local `config.json` (`action_mode=auto`, 6-D action) and symlinks the `lerobot/xvla-base` weights. `train_dm05.cmd` loads `lerobot/dm05_base` with top and wrist cameras. Both slice on-disk 15-D state to the first 6 joint positions.

---

## 6. Policy eval in MuJoCo (sim2sim)

One entry point: **`./labs/lab01_pnp/eval.cmd`**. Swap `LAB01_POLICY_PATH` + matching
`LAB01_EVAL_CONFIG` for SmolVLA, ACT, or another LeRobot policy; optional
`LAB01_N_ACTION_STEPS` when you need to override the checkpoint.
The frozen 20-pose, three-seed protocol is **`./labs/lab01_pnp/eval_home20.cmd`**
(xVLA and DM0.5 need `LAB01_LEROBOT_SRC`; see the section above).

Closed-loop rollout: load a fine-tuned checkpoint, run inference at 20 Hz in MuJoCo with the same cameras and position-control actions as training.

**Success criterion:** cube center inside the container bounds of the `simple_pick` scene (see `simstudio.common.eval_success`). Episodes that grasp but drop mid-transfer, knock the cube away, or leave it on the container rim count as **fail**.

How the lab moved from teleop demos to agent-yaw data, ACT-follow increments, 6-D joint training, and the 3-seed Home20 score is written up in [`labs/lab01_pnp_agent/`](../lab01_pnp_agent/lab01_pnp_agent.md). That lab holds the agent-generated data notes, the Home20 pose file, the 193-episode publish scripts, coverage figures, and demo clips. This runbook keeps the teleop pipeline and the shared train/eval scripts. Per-run data numbers stay in [`smolvla_v4_diversity.md`](../lab01_pnp_agent/smolvla_v4_diversity.md) and [`data_vs_success.md`](../lab01_pnp_agent/data_vs_success.md). Home20 scores stay in [`home20.md`](../lab01_pnp_agent/home20.md).

### 6.1 What you are measuring

Three axes change the reported success rate. Compare only runs that match on all three, or call out the difference explicitly.

| Axis | Choices | Meaning |
|------|---------|---------|
| **Policy** | SmolVLA 50K (`lab01_pnp_smolvla`) vs ACT 50K (`lab01_pnp_act`) | Different architectures / training recipes on the **same** 50-episode lab dataset |
| **Cube spawn** | Full-range random / home-20 paired / narrowed demo random / fixed pose | Distribution of initial cube `(x, y, yaw)` at episode reset |
| **Inference** | `rtc` vs `sync`; ACT `n_action_steps` | How actions are produced relative to the 20 Hz control loop |

**Honest generalization** uses the recording spawn box, freshly sampled. **Lab01-PnP-Home20** ([`home20.md`](../lab01_pnp_agent/home20.md)) freezes one 20-pose draw from that same box so later runs share initial conditions. It is an eval protocol, not a training dataset. **Demo configs** deliberately shrink that box (or freeze one pose) so a classroom run looks more reliable without retraining. Demo rates, Home20 rates, and a new full-range draw are **not** interchangeable.

### 6.2 Eval configs (spawn)

Lab **eval** YAMLs (demo/fixed spawn, ACT vs SmolVLA rename) live under
**`labs/lab01_pnp/configs/`** — lab-specific measurement setups, not product defaults.
Repo-root `configs/` keeps foundational record/replay/scene assets (including
`configs/so101_mujoco_pick_leader.yaml` used by Lab 01 recording) and generic templates.


Recording / full-range box (leader teleop):  
`x ∈ [0.26, 0.34]`, `y ∈ [0.165, 0.235]`, `yaw ∈ [-45°, 45°]`.

| Config | Policy | `reset_cube` | `reset_arm` | Spawn | Role |
|--------|--------|--------------|------------|-------|------|
| `labs/lab01_pnp/configs/rollout_smolvla.yaml` | SmolVLA | `random` | `home` | Full record box | Honest SmolVLA generalization (new draw each run) |
| policy YAML + `LAB01_EPISODE_INIT=labs/lab01_pnp_agent/configs/smolvla_v4_div50_home20_spawns.json` | Any compatible policy | random, then overwritten | `home` | 20 frozen full-box poses | **Lab01-PnP-Home20** ([`home20.md`](../lab01_pnp_agent/home20.md)) |
| `labs/lab01_pnp/configs/rollout_smolvla_demo.yaml` | SmolVLA | `random` | `home` | `x∈[0.265,0.29]`, `y∈[0.175,0.205]`, `yaw∈[-15°,0°]` | Narrowed random demo (default `eval.cmd`) |
| `labs/lab01_pnp/configs/rollout_smolvla_demo_fixed.yaml` | SmolVLA | `fixed` | `home` | Always `(0.27, 0.20, yaw −8°)` via `cube_positions_demo_fixed.json` | Fixed-pose SmolVLA demo |
| `labs/lab01_pnp/configs/rollout_act.yaml` | ACT | `random` | `follow` | Full record box | Honest ACT benchmark |
| `labs/lab01_pnp/configs/rollout_act_demo_fixed.yaml` | ACT | `fixed` | `follow` | Same pose as SmolVLA fixed demo | Fixed-pose ACT demo |
| `labs/lab01_pnp/configs/rollout_molmoact2.yaml` | MolmoAct2 | `random` | `home` | Full record box | Honest MolmoAct2 benchmark (`top→cam0`, `wrist→cam1`) |
| `labs/lab01_pnp/configs/rollout_lingbot_vla_v2.yaml` | LingBot-VLA 2.0 | Home20 / random | `home` | Joint slots | top + wrist (see slots JSON) |
| `labs/lab01_pnp/configs/rollout_fastwam.yaml` | FastWAM | `random` | keyboard `home` | Full record box | same 2-cam rename as JEPA (`top→image`, `wrist→image2`) |
| `labs/lab01_pnp/configs/rollout_fastwam_demo_start.yaml` | FastWAM | `random` | `home_joints` = demo mean | Full record box | If cube never leaves keyboard-home spawn |
| `labs/lab01_pnp/configs/rollout_molmoact2_demo_fixed.yaml` | MolmoAct2 | `fixed` | `home` | Always `(0.27, 0.20, yaw −8°)` | Fixed-pose MolmoAct2 demo |
| `labs/lab01_pnp/configs/rollout_vla_jepa_demo_fixed.yaml` | VLA-JEPA | `fixed` | keyboard `home` | Same cube; arm at `[0,-0.3,0.6,1.2,…]` | Historical 0/10 protocol |
| `labs/lab01_pnp/configs/rollout_vla_jepa_demo_start.yaml` | VLA-JEPA | `fixed` | `home_joints` = demo mean | Same cube; arm at leader-start mean | Isolates home-OOD vs policy failure |

**`reset_arm` rule of thumb** (same meanings as in §1 recording):

| Teleop / mode | Prefer | Why |
|---------------|--------|-----|
| Real **leader** recording | `follow` | No home teleport → no first-frame yank under 1:1 joint mapping |
| **Keyboard / Joy-Con** recording | `home` | No physical leader; fixed start pose each episode |
| Policy **eval** (no teleop) | Either; document which | `home` = every episode starts from the fixed home pose (easier to reason about). `follow` = leave the arm where the previous episode ended (matches leader-record style; episode starts are path-dependent). |

Lab 01 **keeps both on purpose** in the eval YAMLs above: SmolVLA → `home`, ACT → `follow`. That is an **intentional side-by-side example** of the two reset styles (not an ACT-vs-SmolVLA algorithm requirement). Use it to see how episode-start arm pose changes closed-loop behavior; when quoting §6.4, keep each policy paired with the `reset_arm` in the table (or re-measure after changing it). Do **not** “unify” these files unless you want a single protocol for a new measurement campaign.

#### 6.2.1 Lab01-PnP-Home20

**Lab01-PnP-Home20** freezes 20 cube poses and the home arm start `qpos` `[0.0, -0.3, 0.6, 1.2, 0.0, 0.85]` so later runs share initial conditions. It is an eval protocol, not a training dataset. Pose file, seed rule, the 0.85 score table, and the 193-episode publish scripts: [`labs/lab01_pnp_agent/home20.md`](../lab01_pnp_agent/home20.md). Teleop full-range numbers stay in §6.4.

Fixed pose freezes **both** translation and yaw. Episode-to-episode outcomes can still differ because policies (especially SmolVLA flow sampling) are stochastic and because inference latency / chunking change the closed loop.

**SmolVLA** configs include the training `rename_map` (`camera_top/front/wrist` → `camera1/2/3`). **ACT** configs must **not** use that map — use the `rollout_act*.yaml` files.

### 6.3 Inference backends

| Backend | Config / flag | Behavior | When it matters |
|---------|---------------|----------|-----------------|
| **RTC** | `inference.type: rtc` (SmolVLA rollout defaults) | Async action chunks in a background thread | Needed when policy latency ≫ control period on real robots; in sim, backlog shows as `Indexes diff is not equal to real delay` |
| **Sync** | `inference.type: sync` (ACT rollout default; override with `--inference.type=sync`) | Inline `select_action` on the control thread | Removes RTC queue lag in sim; wall-clock still ~60 s/episode if each episode is capped at 60 s |

Optional **`LAB01_N_ACTION_STEPS`**: when set, passed as `--policy.n_action_steps`. Leave empty to use the value stored in the checkpoint (SmolVLA 50K: 50; ACT 50K train: 100). Lab ACT full-range reference used `50`.

**Headless GPU Hz (sync vs RTC):** `./labs/lab01_pnp/hz_bench.cmd` (`HOST=halo GL=egl` or `HOST=dorobot GL=osmesa`). Fixed cube, 4 episodes, drop ep 1. Compare **replanning latency**, not RTC `get_action` Hz. Tables and RTC FAQ: [`hz_bench/COMPARISON.md`](hz_bench/COMPARISON.md); raw medians: [`hz_bench/RESULTS.md`](hz_bench/RESULTS.md). Logs: `outputs/eval/hz_bench/` (not committed).

### 6.4 Reference measurements (lab01-pnp)

Living comparison table for closed-loop MuJoCo eval on this scene. Append a row when a new policy or checkpoint is measured; do not overwrite prior rows.

**n** is small for demo/fixed runs; treat those as indicative, not precise CI estimates. Full-range rows are the primary generalization numbers. Agent-yaw paired comparisons use the Home20 file in [`home20.md`](../lab01_pnp_agent/home20.md); do not merge those fractions into the rows below. Quote `reset_arm`, inference backend, and `n_action_steps` with the success fraction — do not mix protocols.

SmolVLA full-range is **11/50 (22%)**, not a rounded-only figure.

| Policy | Spawn | `reset_arm` | Inference | Backend | Episodes | Success | Notes |
|--------|-------|-------------|-----------|---------|----------|---------|-------|
| SmolVLA 50K | Full-range random | `home` | RTC | EGL | 50 | **11/50 (22%)** | Most fails never leave spawn; grasp/close unreliable |
| SmolVLA 50K | Demo random (early box `x≤0.30`, yaw ±15°) | `home` | RTC | EGL | 20 | **5/20 (25%)** | Narrowing alone did not fix grasp failures |
| SmolVLA 50K | Fixed `(0.27,0.20,−8°)` | `home` | RTC | GLFW | 10 | **5/10 (50%)** | Same pose; RTC delay warnings throughout |
| SmolVLA 50K | Fixed `(0.27,0.20,−8°)` | `home` | Sync | GLFW | 10 | **3/10 (30%)** | Fewer knock-aways; more mid-transfer drops |
| ACT 50K (15-D state) | Full-range random | `follow` | Sync | EGL | 50 | **32/50 (64%)** | Reference with `LAB01_N_ACTION_STEPS=50`; pos+vel+ee; Hub [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act) |
| ACT 80K (15-D, **scripted-agent** 100 ep) | Full-range random | `follow` | Sync | **GLFW** | 50 | **9/50 (18%)** | `n_action_steps=50`, 30 s / 600 ticks; ckpt `lab01_pnp_act_agent/080000`; Hub data [alexhegit/so101-simstudio-lab01-pnp-agent](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent). Funnel: reach 50/50, close 47/50, lift 14/50. **Do not cite** DORobot OSMesa 0/50 (wall-clock truncation). After-action: [`agent_act_80k.md`](../lab01_pnp_agent/agent_act_80k.md) |
| ACT 50K (6-D state) | Full-range random | `follow` | Sync | EGL | 50 | **29/50 (58%)** | Joint `.pos` only (`LAB01_ACT_STATE_DIM=6`); same protocol as 15-D 64% row; Hub [alexhegit/so101-simstudio-lab01-pnp-act-state6](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-state6) |
| ACT 50K | Full-range random | `follow` | Sync | — | 50 | **23/50 (46%)** | Same checkpoint with `n_action_steps=100` (historical log) |
| ACT 50K | Fixed `(0.27,0.20,−8°)` | `follow` | Sync | GLFW | 10 | **8/10 (80%)** | `n_action_steps=100`, `rollout_act_demo_fixed.yaml` |
| MolmoAct2 10K (15-D state) | Full-range random | `home` | Sync | EGL | 50 | **15/50 (30%)** | `n_action_steps=30`, `rollout_molmoact2.yaml`; CUDA graphs off (gfx1151) |
| MolmoAct2 10K (6-D state) | Full-range random | `home` | Sync | EGL | 50 | **27/50 (54%)** | Joint `.pos` only (`LAB01_STATE_DIM=6`); same protocol as 15-D 30% row; `outputs/train/lab01_pnp_molmoact2_state6/checkpoints/010000/` |
| MolmoAct2 10K | Fixed `(0.27,0.20,−8°)` | `home` | Sync | EGL | 10 | **7/10 (70%)** | `n_action_steps=30`, `rollout_molmoact2_demo_fixed.yaml`; CUDA graphs off (15-D) |
| VLA-JEPA 20K (15-D / 6-D) | Fixed `(0.27,0.20,−8°)` | keyboard `home` | Sync | EGL | 10 | **0/10 (0%)** | `n_action_steps=7`; cube never leaves spawn. 6-D 20K same as 15-D 10K. Keyboard home is ~1.7 rad from leader-demo starts |
| VLA-JEPA 6-D BC 10K | Fixed cube; demo-mean `home_joints` | `home` | Sync | EGL | 10 | **1/10 (10%)** | freeze Qwen, no world model, chunk 30; `rollout_vla_jepa_demo_start.yaml` |
| VLA-JEPA 6-D BC 10K | Fixed cube; demo-mean `home_joints` | `home` | Sync | GLFW | 10 | **0/10 (0%)** | Same checkpoint; `eval_vla_jepa_gui.cmd`; failures at grasp (approach, no commit) |
| VLA-JEPA 6-D BC 10K | Fixed cube; keyboard `home` | `home` | Sync | EGL | 10 | **1/10 (10%)** | same checkpoint; `rollout_vla_jepa_demo_fixed.yaml` |
| VLA-JEPA 6-D BC 20K | demo-mean start **and** keyboard `home` | `home` | Sync | EGL | 10+10 | **0/10 (0%)** | same recipe; train loss ~0.02; worse than 10K on this n=10 |
| VLA-JEPA 6-D BC 10K + act-head 3K | Fixed cube; demo-mean `home_joints` | `home` | Sync | EGL | 10 | **1/10 (10%)** | freeze all but `action_model`; init 010000 BC; `n_action_steps=30`; `outputs/eval/vla_jepa_state6_bc_acthead_003000_demostart.log` |
| VLA-JEPA 6-D BC 10K + act-head 3K | Fixed cube; keyboard `home` | `home` | Sync | EGL | 10 | **0/10 (0%)** | same checkpoint; `rollout_vla_jepa_demo_fixed.yaml`; no gain vs 10K BC 1/10 |
| VLA-JEPA 6-D BC 10K + gripper oracle | Fixed cube; demo-mean `home_joints` | `home` | Sync | EGL | 10 | **0/10 (0%)** | `LAB01_GRIPPER_ORACLE=1` (close when `gripperframe` within 4 cm); close fired every episode, cube never lifted. Log `outputs/eval/vla_jepa_state6_bc_010000_demostart_oracle.log` |
| VLA-JEPA 6-D BC 10K → unfreeze Qwen 10K | Fixed cube; demo-mean `home_joints` | `home` | Sync | OSMesa | 3 | **0/3 (0%)** | Continue from BC 010000; `freeze_qwen=false`, WM off, lr 3e-5, chunk 30. Funnel: `reach_8cm=3/3 → near_4cm=3/3 → close_after_open=0/3`. Log `/mnt/doscratch/outputs/eval/jepa_bc_unfreeze_funnel_matched3.log` |
| VLA-JEPA 6-D BC 10K → unfreeze Qwen 10K | Fixed cube; demo-mean `home_joints` | `home` | Sync | OSMesa | 10 | **0/10 (0%)** | Same ckpt; worse than freeze-Qwen BC 1/10. Funnel: `reach_8cm=10/10 → near_4cm=6/10 → close=0/10`. Log `.../jepa_bc_unfreeze_demostart10.log` |
| VLA-JEPA 6-D BC 10K → unfreeze Qwen 10K | Fixed cube; keyboard `home` | `home` | Sync | OSMesa | 10 | **0/10 (0%)** | Same ckpt; `near_4cm=0/10`. Log `.../jepa_bc_unfreeze_fixed10.log`. Intervention ledger: §6.4.2 |
| LingBot-VLA 2.0 joint-slot 20K | Lab01-PnP-Home20 | `home` | Sync | EGL | 60 | **25/60** | Inverse-slot patch required. Ledger: [`lingbot_vla_v2_adapt.md`](lingbot_vla_v2_adapt.md). |
| FastWAM 4K 6-D (freeze video expert, bs4) | Full-range random | keyboard `home` | Sync | OSMesa | 20 | **0/20 (0%)** | MI300X; `n_action_steps=32`, `rollout_fastwam.yaml`; cube never leaves spawn (`z≈0.015`). Policy infer **33.5 Hz** (2631 calls). EGL `EGL_BAD_ALLOC` — OSMesa. Log `/mnt/doscratch/outputs/eval/fastwam_osmesa20.log`; ckpt `.../004000/` |
| FastWAM 20K 6-D (resume 4K, freeze video, bs4) | Full-range random | keyboard `home` | Sync | OSMesa | 10 | **0/10 (0%)** | Same YAML / `n_action_steps=32`; cube still `z≈0.015`. Infer **35.8 Hz** (1207 calls). Log `/mnt/doscratch/outputs/eval/fastwam_osmesa10_20k.log`; ckpt `.../020000/` |
| FastWAM 20K 6-D | Full-range random | demo-mean `home_joints` | Sync | OSMesa | 10 | **0/10 (0%)** | `rollout_fastwam_demo_start.yaml`, `n_action_steps=10`; cube still `z≈0.015`. Infer **18.1 Hz** (1133 calls; shorter chunks). Log `.../fastwam_osmesa10_20k_demo.log` |
| FastWAM pad7 unfreeze-4 8K | Full-range random | keyboard `home` | Sync | **GLFW (GUI)** | 5 | **0/5 (0%)** | `.43` iGPU visual confirmation of the OSMesa 0/10; cube `z≈0.0149` all 5 episodes. Infer **13.9 Hz** (2665 calls). Arm sweeps wide but not toward the cube. Log `outputs/eval/fastwam_pad7_unfreeze4_glfw5.log`. Closed Lab 01 case: §6.4.2 |
| GR00T N1.7-3B 10K (`new_embodiment`, 6-D joints, bs4) | Full-range random | `home` | RTC | OSMesa | 10 | **0/10 (0%)** | MI300X; `rollout_groot.yaml`. Funnel `reach_8cm=10/10 → near_4cm=6/10 → close=1/10 → lift_2cm=0/10`. Cube `z≈0.015`. EGL `EGL_BAD_ALLOC` — OSMesa. Log `/mnt/doscratch/outputs/eval/groot_full_010000_osmesa.log`. **Keeper** `outputs/train/lab01_pnp_groot_n17/checkpoints/010000/` (DORobot `/mnt/doscratch/outputs/train/lab01_pnp_groot_n17/checkpoints/010000/`). **Not** N1.5. Spec: [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md). |
| GR00T N1.7-3B 10K | Full-range random | `home` | RTC | **EGL** | 10 | **0/10 (0%)** | **8060S / halo.** Same keeper 010000. Funnel `reach_8cm=9/10 → near_4cm=7/10 → close=5/10 → lift=0/10`. Real chunk **2.53 Hz** (~395 ms); control compute 56.8 Hz. Log `outputs/eval/groot_halo_full_010000.log`. |
| GR00T N1.7-3B 10K | Fixed demo cube | `home` | RTC | EGL | 10 | **0/10 (0%)** | Same ckpt; `rollout_groot_demo_fixed.yaml`. Funnel `reach_8cm=9/10 → near_4cm=6/10 → close=2/10 → lift=0/10`. Chunk **2.51 Hz**. Log `outputs/eval/groot_halo_fixed_010000.log`. |
| GR00T N1.7-3B 10K → cont 30K (fresh opt from 010000) | Full-range random | `home` | RTC | OSMesa | 10 | **0/10 (0%)** | **Do not keep.** Approach collapsed (`reach_8cm=2/10`). Chunk ~5.8 Hz. Ckpt `/mnt/doscratch/outputs/train/lab01_pnp_groot_n17_cont30k/checkpoints/030000/`. Log `.../groot_n10_cont30k_full.log`. |
| GR00T N1.7-3B 10K → cont 30K | Fixed demo cube | `home` | RTC | OSMesa | 10 | **0/10 (0%)** | **Do not keep.** `reach_8cm=1/10`. Log `.../groot_n10_cont30k_fixed.log`. |
| GR00T N1.7-3B cont30K → vis-unfreeze 20K | Full-range random | `home` | RTC | OSMesa | 10 | **0/10 (0%)** | **Do not keep.** `tune_visual=true` on 50-ep set; `reach_8cm=1/10`. Ckpt `.../lab01_pnp_groot_n17_vis20k/checkpoints/020000/`. Log `.../groot_n10_vis20k_full.log`. |
| GR00T N1.7-3B vis-unfreeze 20K | Fixed demo cube | `home` | RTC | OSMesa | 10 | **0/10 (0%)** | **Do not keep.** `reach_8cm=0/10` (worse than 10K). Log `.../groot_n10_vis20k_fixed.log`. |

**Reading the table**

- `reset_arm` matches the intentional §6.2 contrast (SmolVLA and MolmoAct2 eval YAMLs → `home`, ACT → `follow`). Do not mix rows when claiming a single protocol.
- On this 50-episode dataset, under the measured setups: **ACT full-range (15-D 64% / 6-D 58%, same level at n=50) > MolmoAct2 10K 6-D (54%) > MolmoAct2 10K 15-D (30%) > SmolVLA 50K full-range (22%)**. Quote `state=15` vs `state=6` when citing ACT or MolmoAct2; do not collapse them into one number.
- **Scripted-agent ACT 80K (100 ep) is 9/50 (18%)** on the same full-range ACT YAML (GLFW, 30 s/ep). Do not cite it as beating teleop 64%. Wrist is locked in the demos; see [`agent_act_80k.md`](../lab01_pnp_agent/agent_act_80k.md).
- 6-D ACT matches real SO-101 joint-pos IL. The 6-point gap vs 15-D is within n=50 noise; choose 6-D for sim2real alignment, not for a sim-success gain.
- MolmoAct2 6-D vs 15-D is a **different** story than ACT. The SO100/101 base discretizes proprio into a **language state string** pretrained on **6 joint tokens**. Lab 01 15-D fine-tune fitted quantile stats on 15 channels (pos+vel+EE) and eval expanded rollout to match; 6-D keeps the pretrained layout. Train recipe was otherwise identical (bs32 × 10K, LoRA). The 24-point full-range gap is unlikely n=50 noise (two-proportion z≈2.5); failure mode is still mostly missed grasps at spawn, just fewer of them. Do not copy this Molmo gap onto ACT.
- Fixed spawn raises ACT (80%) and MolmoAct2 (70%) more than SmolVLA (≈30–50%). SmolVLA is not classroom-stable even at a known-good pose.
- For SmolVLA, spawn tightening helped less than expected: failure analysis pointed to **grasp instability** and **yaw polarity** (positive yaw often knocks the cube) more than XY coverage alone.
- RTC lag on iGPU (≈8–9 control frames) changes *how* SmolVLA fails; switching to sync did not turn fixed-pose eval into a high success rate.
- MolmoAct2 10K eval requires loading the fine-tune processor JSON (not rebuilding from the SO100/SO101 degree `norm_tag`) and feeding the 15-dim dataset state the normalizer was fitted on. Apply `patches/lerobot-molmoact2-eval.patch` to the local `lerobot/` checkout (`git -C lerobot apply ../patches/lerobot-molmoact2-eval.patch`); do not commit that edit inside the submodule.
- FastWAM and VLA-JEPA on this 50-ep set are **closed joint-space cross-embodiment transfer-failure cases**, not classroom policies. Classroom pick-and-place still uses ACT 6-D or MolmoAct2 6-D. Ranked LingBot on Lab 01 is **LingBot-VLA 2.0** ([#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19)); LingBot-VA Lab 01 trials are closed ([#9](https://github.com/rocPAI-Forge/so101-simstudio/issues/9)). Family split: **§6.4.2**. Also: [JEPA #7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7), [FastWAM #8](https://github.com/rocPAI-Forge/so101-simstudio/issues/8), [GR00T #6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6).
- **FastWAM** fails *before approach* (`reach_8cm=0`; cube never leaves spawn). Best Lab 01 recipe (LIBERO pad7 + unfreeze last 4 video MoT layers) still **0/10** with open-loop RMSE **0.136 rad** (~3.8× ACT). Lab 01 recipes kept `lambda_video=0` (not the paper's video co-training). Full intervention ledger: **§6.4.2**. Mechanism numbers: **§6.4.1**. Issue: [#8](https://github.com/rocPAI-Forge/so101-simstudio/issues/8).
- **VLA-JEPA** fails *at / after grasp*. Best measured run is freeze-Qwen BC 10K (**1/10** fixed-pose EGL). Unfreezing Qwen from that checkpoint for 10K steps made closed-loop **worse** (**0/10**; approach without close). Hub artifact remains the BC 10K weights: [alexhegit/so101-simstudio-lab01-pnp-vla-jepa](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-vla-jepa). Do not iterate more JEPA/FastWAM long trains on this dataset — see **§6.4.2**. Issue: [#7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7).
- **GR00T N1.7** on the same 50-ep joint set: **keep 10K only** (`checkpoints/010000`). Closed-loop **0/10** but approach works (`reach_8cm≈9–10/10`); grasp/lift does not. Continuing 30K from 10K (cosine already at lr=0) and unfreezing vision 20K both **hurt** approach. Real chunk Hz: **~2.5** on 8060S EGL, **~5.5–6** on MI300X (OSMesa control compute ~2 Hz is software render). Relative actions need RTC. To raise k/n, add more successful grasp/place demos — do not stack more steps on 010000 with the failed resume recipe. Ledger: [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md). Issue: [#6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6).

#### 6.4.1 Protocol-aligned causal audit (2026-09-15)

This audit separates reference success rates from mechanism probes. Full-range rows below are still not one perfectly controlled benchmark: ACT uses `reset_arm: follow`, SmolVLA/MolmoAct2 use `home`, and each policy keeps its native camera/state/action interface. Wilson intervals show sampling uncertainty; architecture rankings must not be inferred when intervals and protocols overlap.

| Policy / recipe | Cameras | State / action norm | Chunk | Training sample updates | Reused / frozen path | Full-range result (95% Wilson CI) |
|---|---:|---|---:|---:|---|---|
| ACT 6-D 50K | 3 | 6 / MEAN_STD | 100 | 6.40M | ResNet-18; Lab 01 policy trained end-to-end | 29/50 = 58% (44–71%) |
| ACT 15-D 50K | 3 | 15 / MEAN_STD | 100 | 6.40M | same recipe | 32/50 = 64% (50–76%) |
| SmolVLA 50K | 3 | 6 / MEAN_STD | 50 | 3.20M | SmolVLM base | 11/50 = 22% (13–35%) |
| MolmoAct2 6-D 10K | 2 | 6 / QUANTILES | 30 | 0.32M | SO100/101 6-joint base, VLM LoRA + action expert | 27/50 = 54% (40–67%) |
| MolmoAct2 15-D 10K | 2 | 15 / QUANTILES | 30 | 0.32M | same, but 15 tokens break the pretrained 6-joint state layout | 15/50 = 30% (19–44%) |
| VLA-JEPA BC 10K | 2 | 6 / MIN_MAX | 30 | 0.16M | LIBERO base; Qwen frozen; state/action projections reinitialized; WM off | fixed only: 1/10 = 10% (2–40%) |
| VLA-JEPA BC→unfreeze Qwen 10K | 2 | 6 / MIN_MAX | 30 | +0.16M | Continue from BC 010000; Qwen trainable; WM off; lr 3e-5 | fixed only: 0/10 = 0% (0–28%); worse than BC |
| FastWAM pad7 + unfreeze-4 | 2 | 6→8 / 6→7, MEAN_STD | 32 | 0.448M | LIBERO action path; Wan video expert frozen except last 4 MoT layers in final 8K | 0/10 = 0% (0–28%) |

Matched state ablations support embodiment alignment, not a generic “less state is better” rule: ACT 6-D vs 15-D differs by only 6 points with heavily overlapping intervals, while MolmoAct2 gains 24 points because 6-D restores the SO100/101 base model's pretrained six-token proprio layout. ACT and MolmoAct2 6-D are statistically indistinguishable at n=50 (58% vs 54%); SmolVLA is clearly less reliable despite 10× MolmoAct2's fine-tuning sample updates.

**Dataset audit.** The dataset has 50 episodes / 21,218 frames; episode length is 267–1,057 frames (median 406.5). Episode-start arm pose is not fixed (largest joint std 0.180 rad), which invalidates the old “only cube pixels vary” interpretation of first-frame cross-episode probes. Gripper supervision is also not binary: 50.3% of frames are `<0.15` rad, 25.0% are `>0.75`, and 24.7% are in transition. Median close ramp is 15 frames / 0.75 s; replacing it with two modes has **0.209 rad RMSE**. The current JEPA BC recipe correctly sets `binarize_gripper_action=false`; binarization is therefore a hypothetical ablation, not its failure cause.

**Unified in-sample open-loop diagnostic.** All checkpoints trained on all 50 episodes, so no honest held-out action test exists without retraining. The probe samples four episodes at start / 25% / 50% / 75%, uses each model's native horizon, raw-radian actions and deterministic seed, and compares with holding the current pose. Horizon differences mean absolute RMSE is diagnostic rather than a fair score.

| Policy | Horizon | RMSE | Hold RMSE | Improvement over hold | RMSE after cross-episode image shuffle | Start RMSE | Pred/demo travel |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACT 6-D | 100 | **0.040** | 0.579 | **93%** | 0.410 | 0.047 | 1.01× |
| SmolVLA | 50 | 0.160 | 0.354 | 55% | 0.255 | 0.251 | 0.93× |
| MolmoAct2 6-D | 30 | 0.113 | 0.204 | 44% | 0.216 | 0.103 | 1.01× |
| VLA-JEPA BC 10K | 30 | 0.187 | 0.204 | **8%** | **0.172 (improves)** | 0.245 | 1.00× |
| VLA-JEPA WM 20K | 7 | 0.080 | 0.084 | **4%** | 0.204 | 0.094 | 1.27× |
| VLA-JEPA unfreeze Qwen 10K | 30 | **0.204** | 0.230 | **11%** | 0.204 (≈unchanged) | 0.273 | 1.12× |
| FastWAM unfreeze-4 | 32 | 0.127 | 0.215 | 41% | 0.192 | 0.193 | 0.99× |

ACT is the only checkpoint that strongly beats a hold baseline at every phase and whose error explodes when images are mismatched. JEPA BC barely beats hold and shuffled images improve RMSE, direct evidence that its visual conditioning is often mapped in the wrong direction. Unfreezing Qwen from BC 10K made RMSE **worse** (0.187 → 0.204) and start-phase RMSE (0.273) still beats hold the wrong way (hold 0.159 at start); shuffle leaves RMSE flat — features moved, but not into a better demo-aligned map. JEPA WM's 0.080 must not be compared naively with 30–100-step rows: over its 7-step horizon hold already scores 0.084. It uses images (shuffle hurts), but the next intervention shows that the encoded variation is barely about cube XY.

**Controlled cube grounding and camera ablation.** `probe_visual_grounding.py` renders a 3×3 cube grid with arm, 6-D proprio and background fixed. A counterfactual hides only the cube geometry and disables the block-targeted light so no cube position leaks through illumination. At 640×480 the green cube is large enough to see (equivalent width: top 31 px, front 44 px, wrist mean 55 px), so poor sensitivity is not a sub-pixel problem.

| Policy | Cube-driven action spread | Pairwise cube-XY/action-distance ρ | Top-only spread | Wrist-only spread | First-chunk FK min cube distance | Fixed-image proprio-driven spread |
|---|---:|---:|---:|---:|---:|---:|
| ACT 6-D | 0.131 | **0.481** | 0.064 | 0.104 | **4.7 cm** (3/9 within 4 cm) | **0.038** |
| SmolVLA | 0.231 | 0.322 | 0.101 | 0.015 | 9.8 cm | 0.145 |
| MolmoAct2 6-D | 0.060 | **0.553** | 0.063 (ρ=0.717) | 0.187 (ρ=0.163) | 8.6 cm | 0.259 |
| VLA-JEPA BC 10K | 0.028 | 0.275 | 0.038 | 0.057 | 11.6 cm | 0.189 |
| VLA-JEPA WM 20K | **0.006** | 0.163 | 0.006 | 0.005 | **20.7 cm** | 0.106 |
| VLA-JEPA unfreeze Qwen 10K | — | — | — | — | — | — |
| FastWAM unfreeze-4 | 0.123 | 0.397 | 0.187 (ρ≈0) | 0.158 (ρ≈0.02) | **12.8 cm** | **0.284** |

The no-cube control has effectively zero spread, proving that the reported variation is caused by cube pixels. ACT's closest FK point tracks target X/Y with correlations 0.946/0.972. FastWAM sees cube displacement at a similar amplitude, but its closest-point X correlation is only 0.216 (Y 0.673), no grid point comes within 4 cm, and the proprio intervention drives 2.3× more action spread than moving the cube. Its defect is therefore not a disconnected camera; it is inaccurate cross-embodiment coordinate/action grounding. JEPA WM is a different failure: the auxiliary objective makes the final checkpoint **4.4× less cube-sensitive than BC** and its first chunk remains about 21 cm away. **Controlled grounding for the unfreeze-Qwen checkpoint was not obtained on MI300X** (`EGL_BAD_ALLOC`, then OSMesa segfault in `probe_visual_grounding.py`); open-loop + closed-loop funnel below still close the case.

**Matched closed-loop funnel (exploratory n=3).** All policies below use the same fixed cube `(0.27, 0.20, −8°)`, demo-mean arm home, sync inference, 30 executed actions and 35 s. This is a mechanism probe, not a replacement for full-range n=50.

| Policy | Reach ≤8 cm | Near ≤4 cm | Close near cube | Lift ≥2 cm | Over box | Place | Final |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACT 6-D | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| SmolVLA | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| MolmoAct2 6-D | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** |
| VLA-JEPA BC 10K | 3/3 | 3/3 | 3/3 | 3/3 | **0/3** | 0/3 | **0/3** |
| VLA-JEPA unfreeze Qwen 10K | 3/3 | 3/3 | **0/3** | 0/3 | 0/3 | 0/3 | **0/3** |
| FastWAM unfreeze-4 | **0/3** | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | **0/3** |

JEPA BC's matched failure is localized after grasp: all three runs reach, close and lift, then drop before reaching the box (`dropped_after_lift=3/3`). Unfreezing Qwen **regresses** that funnel: still reaches / gets near, but **`close_after_open=0/3`** (demo-mean 10-ep: `near_4cm=6/10`, close still 0). FastWAM fails before approach (`reach_8cm=0/3`; cube never moves). SmolVLA's 3/3 versus its earlier fixed sync 3/10 from keyboard home demonstrates that arm-start protocol is a large confound; n=3 does not establish a new success rate.

**What “world model” means in these actual recipes.**

- The [Fast-WAM paper](https://arxiv.org/abs/2603.16666) concludes that video co-training, not test-time imagination, provides most of the WAM gain. The Lab 01 recipe sets `lambda_video=0` and freezes the video expert for 20K, then unfreezes only four MoT layers for 8K while the video loss remains zero. It therefore removes the paper's claimed causal ingredient. Inference intentionally uses only first-frame video tokens, as the paper specifies.
- VLA-JEPA BC 10K has `enable_world_model=false`. WM 20K adds a 0.1-weight auxiliary prediction of frozen V-JEPA future latents; the predictor is not called by `predict_action`, and can affect actions only indirectly through shared Qwen weights. The [official paper](https://arxiv.org/abs/2602.10098) first trains on 220K human videos and 76K robot trajectories and fine-tunes with about 2K LIBERO demos; 50 SO-101 demonstrations with reinitialized action/state projections are not an equivalent learning regime.
- The falsifiable result is stronger than naming: WM 20K has lower cube sensitivity and no better closed-loop result than BC. FastWAM has adequate pixel sensitivity but wrong robot-space grounding. Unfreezing JEPA's Qwen also failed the gate and **regressed** closed-loop vs freeze-Qwen BC.
- **Gate status (closed):** no Lab 01 JEPA/FastWAM checkpoint passes start RMSE `<0.08`, cube/action correlation `>0.5`, and useful first-chunk approach. Subsequent long interventions (pad7, unfreeze-4 MoT, unfreeze Qwen) did not pass either. **This lab does not open further large world-model training runs on the 50-ep set** for classroom success; see the intervention ledger in **§6.4.2**. Paper-level video co-training (`lambda_video>0`) remains an out-of-scope experiment on larger / same-embodiment data, not a Lab 01 success path.

Reproduce the durable JSON bundle and offline probes:

```bash
source .venv-rocm/bin/activate
python labs/lab01_pnp/analyze_model_gap.py
python labs/lab01_pnp/probe_open_loop.py \
  --checkpoint outputs/train/lab01_pnp_act_state6/checkpoints/050000/pretrained_model \
  --episodes 4 --output .tmp/openloop_act6.json
python labs/lab01_pnp/probe_visual_grounding.py \
  --checkpoint outputs/train/lab01_pnp_act_state6/checkpoints/050000/pretrained_model \
  --output .tmp/grounding_act6.json
```

`eval.py` now emits per-episode funnel maps and a final `Stage funnel` line for every sim eval. Matched logs are `outputs/eval/funnel_matched_{act6,smolvla,molmo6,jepa_bc10k,fastwam}.log`. Machine-readable aggregate: `.tmp/model_gap_analysis.json`.

**Adding a new policy:** measure full-range (`n=50`) and optionally fixed-pose (`n=10`) with the matching `labs/lab01_pnp/configs/rollout_*.yaml`. Add one row per (policy, spawn, `reset_arm`, inference) tuple. Keep Success as `k/n (p%)`.

#### 6.4.2 Same 50-ep set: joint-space transfer failures vs EEF policies (2026-09-16)

**Verdict.** On the same 50 expert episodes, ACT 6-D (~58%) and MolmoAct2 6-D (~54%) show the task is learnable as **joint-position IL**. VLA-JEPA (0–10%) and FastWAM (0%) are **joint-space LIBERO→SO-101 transfer failures**, not evidence that Lab 01 pick-and-place is too hard. LingBot-VA EEF trials on Lab 01 are closed ([#9](https://github.com/rocPAI-Forge/so101-simstudio/issues/9)); ranked LingBot is VLA 2.0 joint slots ([#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19)). Classroom demos stay on ACT 6-D / MolmoAct2 6-D. Mechanism probes: §6.4.1; JEPA/FastWAM interventions below; action-language split at the end of this section. Also: [JEPA #7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7), [FastWAM #8](https://github.com/rocPAI-Forge/so101-simstudio/issues/8), [GR00T #6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6).

##### VLA-JEPA intervention ledger

| Recipe | Closed-loop (fixed cube) | Funnel / symptom | Hypothesis tested → outcome |
|--------|--------------------------|------------------|-----------------------------|
| WM 20K (chunk 7, Qwen trainable, WM weight 0.1) | 0/10 | Cube never leaves spawn | “World-model co-training helps” → **no**; cube sensitivity **worse** than BC |
| Freeze-Qwen BC 10K (chunk 30, WM off) | **1/10** (best JEPA) | Approach; grasp often fails; matched n=3: lift then drop before box | “Simplify to frozen VLM + action head” → **partial** only |
| Same BC 20K | 0/10 | Worse than 10K | “More steps on frozen features” → **overfit / regress** |
| Action-head-only 3K from BC 10K | 1/10 / 0/10 | No gain | “Bottleneck is action head” → **no** |
| Shorter `n_action_steps` 10 / 5 | worse | Open later or absent | “Replan too late” → **no** |
| Gripper threshold snap | no recovery | Close never issued | “Ambiguous gripper modes” → **no** (binarize already off) |
| Gripper proximity oracle | 0/10 | Close every ep; never lift; ~4 cm Y miss | “Only the jaws hesitate” → **no**; approach not centered |
| **Unfreeze Qwen 10K from BC 010000** (lr 3e-5, WM off, bs16) | **0/10** demo-mean and keyboard-home | Matched: `near_4cm` OK, **`close_after_open=0`**; open-loop RMSE 0.187→**0.204** | “Frozen LIBERO vision is the bottleneck; thaw Qwen” → **regress** |

**Why unfreeze regressed.** Freeze-Qwen BC fitted the action head to a **fixed** feature space (LIBERO-ish, weakly cube-sensitive) and produced a fragile average approach that sometimes closed. Thawing ~2.3B Qwen parameters on 50 demos (~7.5 epochs) **moved the features** while the action head’s calibration stayed in the old space. Train loss fell to ~0.011 without improving demo-aligned open-loop; the weakest channel (close) disappeared first. This is representation drift / catastrophic interference on a tiny dataset — not “unfreeze failed to run.”

Hub publish stays at freeze-Qwen BC 10K. Unfreeze weights are **not** a Hub artifact. Follow-up: [issue #7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7).

##### FastWAM intervention ledger

| Recipe | Closed-loop | Funnel / open-loop | Hypothesis → outcome |
|--------|-------------|--------------------|----------------------|
| `fastwam_base` 4K/20K, freeze video, `lambda_video=0`, reinit 6-D action path | 0/20, 0/10 | Cube `z≈0.015`; RMSE ~0.23 rad | “Train action DiT on frozen Wan” → underfit; base **has no action pathway tensors** |
| Demo-mean start + shorter chunks | 0/10 | Still never leave spawn | “Keyboard home OOD” → **not sufficient** |
| LIBERO pad7 20K (inherit 7-D/8-D path, pad 6 joints) | 0/10 | RMSE 0.198 rad | “Inherit action pathway” → better open-loop, still 0% |
| Unfreeze last 4 video MoT + 8K | 0/10 (also 0/5 GLFW) | RMSE **0.136** rad; matched `reach_8cm=0/3`; FK min ~12.8 cm | “Frozen video features bottleneck” → open-loop helps, **closed-loop still 0%** |

Lab 01 FastWAM recipes **never enabled video co-training** (`lambda_video=0`). That is a documented gap vs the [Fast-WAM paper](https://arxiv.org/abs/2603.16666), not a remaining “cheap lever” for Lab 01 success: JEPA already showed that adding the marketed world-model ingredient on this 50-ep SO-101 set does not produce a classroom policy, and FastWAM still fails **before** the grasp stage JEPA sometimes reaches. Follow-up: [issue #8](https://github.com/rocPAI-Forge/so101-simstudio/issues/8).

##### Contrast and stop boundary

| | VLA-JEPA (best / last) | FastWAM (best) |
|--|------------------------|----------------|
| Stage of death | At / after grasp (BC); after unfreeze: approach without close | Before approach |
| Open-loop vs ACT | ~5× worse RMSE; barely beats hold | ~3.8× worse after unfreeze-4 |
| Pass gate RMSE&lt;0.08 / r&gt;0.5? | No | No |

**Stop.** Do not open further Lab 01 long trains for JEPA (full unfreeze, more BC steps, WM weight sweeps) or FastWAM (`lambda_video>0`, more MoT layers, more pad7 steps) aimed at classroom success on this dataset. If a future experiment wants to test paper-level video co-training, do it on larger or same-embodiment data and keep it out of the classroom path.

##### Same dataset, two failure families

Lab 01 teleop is **joint-position IL** (`action` = 6× `*.pos`). Policies that emit the same language can train and eval without a converter:

| Family | Lab 01 examples | Fine-tune on this dataset | Typical Lab 01 outcome |
|--------|-----------------|---------------------------|------------------------|
| **Joint-space** | ACT, SmolVLA, MolmoAct2 (6-D), VLA-JEPA, FastWAM (pad 6→7 joints), **GR00T N1.7** (`new_embodiment`, 10K keeper) | Direct: `action` = robot command (GR00T zero-pads to `max_action_dim`) | ACT/Molmo work. JEPA/FastWAM still fail — **not** because they output EEF, but because of LIBERO→SO-101 vision/action-head transfer on 50 episodes (reinit/pad, frozen or drifted features, `lambda_video=0` / unused WM). **GR00T 10K: 0/10, approach yes / grasp no.** Extra 30K + vis-unfreeze 20K worse. Ledger: [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md). |
| **End-effector (paused on Lab 01)** | LingBot-VA (`lingbot_va_*`) | Official LIBERO is ΔEEF; fair EEF Home20 **0/60** (sparse reach / IK); adapter + derived dataset removed. | Better action-language fit on **7-DoF Cartesian** arms (e.g. openARM dual-arm) than SO-101 5+1. Ranked LingBot here: **VLA 2.0** ([`lingbot_vla_v2_adapt.md`](lingbot_vla_v2_adapt.md)). |

Cameras: JEPA/FastWAM/LingBot all rename `top→image`, `wrist→image2` and drop `front`. That is key-name alignment only (top-down ≠ LIBERO agentview). For joint models it was a **second-order** issue (FastWAM still responded to cube pixels). For LingBot it remains OOD, but the **first-order** block is action language.

**Hub.** Published: dataset, SmolVLA, ACT 15-D/6-D, MolmoAct2, VLA-JEPA BC 10K (negative, [issue #7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7)), **GR00T N1.7 10K** (negative, [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17); [issue #6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6)). **Not** on Hub: FastWAM ([issue #8](https://github.com/rocPAI-Forge/so101-simstudio/issues/8)), Molmo 6-D as a separate repo, JEPA unfreeze, GR00T 30K / vis-20K. LingBot-VLA 2.0 20K is on Hub ([#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19)).

**Follow-up (not this analysis).** Ranked LingBot: [`lingbot_vla_v2_adapt.md`](lingbot_vla_v2_adapt.md) — [issue #19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19). LingBot-VA Lab 01 path closed — [issue #9](https://github.com/rocPAI-Forge/so101-simstudio/issues/9). GR00T N1.7: [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md) — [issue #6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6). Do not load GR00T N1.5 on this LeRobot pin.

##### Log index (durable)

| Artifact | Path |
|----------|------|
| JEPA unfreeze train | `/mnt/doscratch/outputs/train/lab01_pnp_vla_jepa_state6_bc_unfreeze/` (010000) |
| JEPA unfreeze open-loop | `/mnt/doscratch/outputs/eval/jepa_bc_unfreeze_open_loop.json` |
| JEPA unfreeze closed-loop | `.../jepa_bc_unfreeze_{funnel_matched3,demostart10,fixed10}.log` |
| Matched funnel (causal audit) | `outputs/eval/funnel_matched_{act6,smolvla,molmo6,jepa_bc10k,fastwam}.log` |
| Aggregate JSON | `.tmp/model_gap_analysis.json` (regenerate via `analyze_model_gap.py`) |

### 6.5 How to run

```bash
source .venv-rocm/bin/activate

# Default: SmolVLA demo spawn + LAB01_POLICY_PATH (50K)
./labs/lab01_pnp/eval.cmd

# SmolVLA full-range
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_smolvla.yaml \
  LAB01_EVAL_EPISODES=50 ./labs/lab01_pnp/eval.cmd

# SmolVLA fixed-pose demo (GUI)
LAB01_MUJOCO_GL=glfw LAB01_RENDER_WINDOW=true LAB01_EVAL_EPISODES=10 \
  LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_smolvla_demo_fixed.yaml \
  ./labs/lab01_pnp/eval.cmd

./labs/lab01_pnp/eval.cmd --inference.type=sync   # compare to RTC

# ACT full-range (same eval.cmd — swap path + yaml + n_action_steps)
LAB01_POLICY_PATH=./outputs/train/lab01_pnp_act/checkpoints/last/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=./outputs/eval/act_egl.log \
  ./labs/lab01_pnp/eval.cmd

# ACT fixed-pose demo (GUI)
LAB01_POLICY_PATH=./outputs/train/lab01_pnp_act/checkpoints/last/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act_demo_fixed.yaml \
LAB01_MUJOCO_GL=glfw LAB01_RENDER_WINDOW=true \
LAB01_EVAL_EPISODES=10 LAB01_N_ACTION_STEPS=100 \
LAB01_EVAL_LOG=./outputs/eval/act_glfw_fixed.log \
  ./labs/lab01_pnp/eval.cmd

# ACT 6-D (joint pos; real IL layout) — same yaml as 15-D ACT
LAB01_POLICY_PATH=./outputs/train/lab01_pnp_act_state6/checkpoints/050000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=./outputs/eval/act_state6_egl.log \
  ./labs/lab01_pnp/eval.cmd

# MolmoAct2 10K — cameras top→cam0, wrist→cam1; chunk 30; sync
LAB01_POLICY_PATH=./outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_molmoact2_demo_fixed.yaml \
LAB01_N_ACTION_STEPS=30 LAB01_EVAL_EPISODES=10 \
LAB01_EVAL_LOG=./outputs/eval/molmoact2_fixed.log \
  ./labs/lab01_pnp/eval.cmd

LAB01_POLICY_PATH=./outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_molmoact2.yaml \
LAB01_N_ACTION_STEPS=30 LAB01_EVAL_EPISODES=20 \
LAB01_EVAL_LOG=./outputs/eval/molmoact2_full.log \
  ./labs/lab01_pnp/eval.cmd

# VLA-JEPA GLFW (transfer-failure case). Default: 6-D BC 10K, demo-mean start, 3 eps.
./labs/lab01_pnp/eval_vla_jepa_gui.cmd
LAB01_EVAL_EPISODES=10 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd
# LAB01_JEPA_EVAL_VARIANT=acthead|bc20k|wm|15d  LAB01_JEPA_EVAL_START=home

# Log predicted vs current joint commands for the first N control ticks (any policy).
LAB01_DEBUG_ACTIONS=400 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd

# Snap the gripper command to open/closed at eval time (diagnostic; off by default).
# Threshold 0.45 rad is the midpoint of the demo closed/open modes; latch arms only
# after the policy has opened once. Quote snap on/off next to any success rate.
LAB01_GRIPPER_SNAP=1 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd

# Proximity oracle: close when gripperframe is within 4 cm of the cube, open over the box.
# Ablates the gripper channel; do not quote as a policy success rate.
LAB01_GRIPPER_ORACLE=1 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd

# Optional flow-matching sampler overrides (diagnostic). Official V2 Home20 uses defaults.
# LAB01_FLOW_NOISE_SCALE=… LAB01_FLOW_NUM_STEPS=… ./labs/lab01_pnp/eval.cmd
```


| Override | Example |
|----------|---------|
| Policy path | `LAB01_POLICY_PATH=./outputs/train/.../pretrained_model` |
| Episode count | `LAB01_EVAL_EPISODES=50` |
| GL / window | `LAB01_MUJOCO_GL=egl\|glfw` + `LAB01_RENDER_WINDOW=false\|true` |
| Spawn / policy yaml | `LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_*.yaml` |
| Chunk exec (ACT) | `LAB01_N_ACTION_STEPS=50` or `100` (omit for checkpoint default) |
| Log path | `LAB01_EVAL_LOG=./outputs/eval/my_run.log` |
| Inference engine | `./labs/lab01_pnp/eval.cmd --inference.type=sync` |
| Rerun viz | `./labs/lab01_pnp/eval.cmd --display_data true` |

Log default: `./outputs/eval/eval_${MUJOCO_GL}.log` (override with `LAB01_EVAL_LOG`).

### 6.6 Open questions — improving success (try yourself)

There is no single prescribed fix in this lab. The rates above are a baseline on **50 expert episodes**. Useful experiments (data, training, or eval) include:

1. **Data quality / coverage** — More episodes; denser sampling of far-x / positive-yaw; cleaner grasps (close only after contact, fewer pushes). Re-validate and retrain before claiming gains.
2. **Train longer or change recipe** — Resume SmolVLA past 50K; change batch / LR / warmup; compare ACT vs SmolVLA sample-update budgets on the same host.
3. **Inference knobs in sim** — RTC vs sync; ACT `n_action_steps` (50 vs 100 vs train default); confirm wall-clock delay does not dominate on your GPU.
4. **Spawn curriculum vs honest eval** — Use demo/fixed configs for demos; always report **full-range** numbers when claiming generalization. Do not tune the eval box alone and call it a better policy.
5. **Failure taxonomy** — Log initial spawn vs final cube pose; separate “never grasped”, “dropped in transfer”, and “near-miss on rim”. Different failure modes suggest different next experiments.
6. **Policy choice for demos** — Classroom pick-and-place: ACT 6-D (or MolmoAct2 6-D). Keep SmolVLA if the goal is VLA fine-tuning. VLA-JEPA and FastWAM on this 50-ep set are **closed joint-space transfer-failure cases** (§6.4 / §6.4.2; [#7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7), [#8](https://github.com/rocPAI-Forge/so101-simstudio/issues/8)). Ranked LingBot: VLA 2.0 ([#19](https://github.com/rocPAI-Forge/so101-simstudio/issues/19)). GR00T N1.7 10K is the **kept** Lab 01 GR00T ckpt (**0/10**, approach yes / grasp no); not a classroom policy — [`groot_n1_7_adapt.md`](groot_n1_7_adapt.md), [#6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6).

When you change any axis, re-run with a fixed episode count and the same `MUJOCO_GL` / inference settings, and record config path + checkpoint step next to the success fraction.

### Historical notes (v1 / short ACT runs)

Older ACT runs on deprecated data or shorter schedules (10K/30K) are not the lab01-pnp reference. Prefer the MI300X **50K** checkpoints in the §6.4 table (15-D and 6-D).

---

## 7. Hub artifacts (download + publish)

Reference Hub assets use the **`alexhegit`** account (example below). Switch to **your own
Hugging Face user/org** for personal upload/download by overriding the env vars in `_env.sh`
(or on the command line):

| Purpose | Variable | Example |
|---------|----------|---------|
| Dataset user | `LAB01_DATASET_HF_USER` | `your-hf-id` |
| Dataset repo | `LAB01_DATASET_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp` |
| SmolVLA model repo | `LAB01_SMOLVLA_HF_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp-smolvla` |
| ACT model repo (15-D) | `LAB01_ACT_HF_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp-act` |
| ACT model repo (6-D) | `LAB01_ACT_STATE6_HF_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp-act-state6` |
| VLA-JEPA model repo | `LAB01_JEPA_HF_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp-vla-jepa` |
| GR00T N1.7 model repo | `LAB01_GROOT_HF_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp-groot-n17` |

Published Lab 01 reference assets (`alexhegit`):

| Asset | Hub repo |
|-------|----------|
| Dataset | [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp) |
| SmolVLA policy | [alexhegit/so101-simstudio-lab01-pnp-smolvla](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla) |
| ACT policy (15-D state) | [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act) |
| ACT policy (6-D joint pos) | [alexhegit/so101-simstudio-lab01-pnp-act-state6](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-state6) |
| VLA-JEPA (6-D BC 10K; transfer failure) | [alexhegit/so101-simstudio-lab01-pnp-vla-jepa](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-vla-jepa) ([issue #7](https://github.com/rocPAI-Forge/so101-simstudio/issues/7)) |
| GR00T N1.7 (10K; negative result) | [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17) |
| MolmoAct2 | [alexhegit/so101-simstudio-lab01-pnp-molmoact2](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-molmoact2) |
| FastWAM / LingBot-VLA 2.0 | FastWAM: **not published** (0% closed-loop). VLA 2.0: Home20 **25/60** — [lingbot-vla-v2-agent193-20k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k); [`lingbot_vla_v2_adapt.md`](lingbot_vla_v2_adapt.md). |

Commands in §7.1–7.2 keep `alexhegit/...` as copy-paste examples; replace the namespace with yours when working under your own Hub account.

### 7.1 Download for train / eval (no local record required)

Needs network and `hf` CLI (`huggingface_hub`; available after `make rocm-sync` / Hub tooling). Mirror users: set `HF_ENDPOINT` as usual. Examples use `alexhegit/...`; substitute `your-hf-id/...` if downloading your own uploads.

**Dataset** → local root used by train / eval stats / optional `--resume` recording:

```bash
source .venv-rocm/bin/activate
mkdir -p ./datasets
hf download alexhegit/so101-simstudio-lab01-pnp \
  --repo-type dataset \
  --local-dir ./datasets/so101-simstudio-lab01-pnp
```

Or in Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alexhegit/so101-simstudio-lab01-pnp",
    repo_type="dataset",
    local_dir="./datasets/so101-simstudio-lab01-pnp",
)
```

**Policies** → any directory you pass as `LAB01_POLICY_PATH` (LeRobot `pretrained_model/` layout):

```bash
# SmolVLA
hf download alexhegit/so101-simstudio-lab01-pnp-smolvla \
  --local-dir ./outputs/hub/lab01_pnp_smolvla

# ACT 15-D (pos+vel+ee)
hf download alexhegit/so101-simstudio-lab01-pnp-act \
  --local-dir ./outputs/hub/lab01_pnp_act

# ACT 6-D (joint pos; real-robot IL layout)
hf download alexhegit/so101-simstudio-lab01-pnp-act-state6 \
  --local-dir ./outputs/hub/lab01_pnp_act_state6

# VLA-JEPA 6-D BC 10K (transfer-failure artifact; not a demo policy)
hf download alexhegit/so101-simstudio-lab01-pnp-vla-jepa \
  --local-dir ./outputs/hub/lab01_pnp_vla_jepa
```

**Eval with downloaded weights** (same `eval.cmd`; see §6):

```bash
# SmolVLA demo spawn
LAB01_POLICY_PATH=./outputs/hub/lab01_pnp_smolvla \
  ./labs/lab01_pnp/eval.cmd

# ACT 15-D full-range
LAB01_POLICY_PATH=./outputs/hub/lab01_pnp_act \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=./outputs/eval/act_hub.log \
  ./labs/lab01_pnp/eval.cmd

# ACT 6-D full-range (same yaml)
LAB01_POLICY_PATH=./outputs/hub/lab01_pnp_act_state6 \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=./outputs/eval/act_state6_hub.log \
  ./labs/lab01_pnp/eval.cmd
```

`LAB01_DATASET_ROOT` should point at the downloaded dataset so eval can patch normalizer stats (default `./datasets/so101-simstudio-lab01-pnp` already matches).

**Extend / stack more demonstrations:** after downloading the Hub dataset into `LAB01_DATASET_ROOT`, keep recording with resume so new episodes append to the same LeRobot v3.0 store:

```bash
./labs/lab01_pnp/record.cmd --resume true
```

Then retrain (`train.cmd` / `train_act.cmd`) on the enlarged local root, or keep using the Hub policies for eval-only workflows. For a separate personal dataset, set `LAB01_DATASET_NAME` / `LAB01_DATASET_REPO_ID` to a new id and record from scratch.

### 7.2 Publish

After `hf auth login`, push with the defaults (`alexhegit/...`) or your own ids:

```bash
# Your own Hub user (example)
export LAB01_DATASET_HF_USER=your-hf-id
export LAB01_SMOLVLA_HF_REPO_ID=your-hf-id/so101-simstudio-lab01-pnp-smolvla
export LAB01_ACT_HF_REPO_ID=your-hf-id/so101-simstudio-lab01-pnp-act
export LAB01_ACT_STATE6_HF_REPO_ID=your-hf-id/so101-simstudio-lab01-pnp-act-state6
export LAB01_JEPA_HF_REPO_ID=your-hf-id/so101-simstudio-lab01-pnp-vla-jepa

# Dataset
./labs/lab01_pnp/push_dataset.cmd

# SmolVLA weights + card (MI300X bs64 @ 50K)
./labs/lab01_pnp/push_smolvla_model.cmd

# ACT 15-D model card (weights already on Hub)
./labs/lab01_pnp/push_act_model_card.cmd

# ACT 6-D weights + card
LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp/push_act_state6_model.cmd

# VLA-JEPA 6-D BC 10K weights + card (transfer-failure artifact)
LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp/push_vla_jepa_model.cmd

# GR00T N1.7 10K weights + card (negative result; NVIDIA Open Model License)
LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp/push_groot_model.cmd
```

Cards: `hf_dataset_card.md`, `hf_model_card_smolvla.md`, `hf_model_card_act.md`, `hf_model_card_act_state6.md`, `hf_model_card_vla_jepa.md`, `hf_model_card_groot.md`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `feetech-servo-sdk` missing | `uv pip install --python .venv-rocm/bin/python 'feetech-servo-sdk>=1.0.0,<2.0.0'` |
| `/dev/ttyACM0` permission denied | Add user to `dialout`, re-login |
| Terminal closes immediately | Run `./labs/lab01_pnp/record.cmd` inside an open terminal, not by double-clicking |
| Rerun viewer not found | `source .venv-rocm/bin/activate` or use `dataset_viz --save` |
| Rerun playback jumpy | Use `--save` + `rerun --memory-limit 4GB`, or `ffplay` on dataset mp4 |
| Used `smoke-leader-teleop` for Rerun | Rerun only works with **record**, not `teleoperate` |
| SmolVLA training OOM | Lower `LAB01_TRAIN_BATCH_SIZE` (try `1`); do not copy large-batch recipes onto small GPUs |
| `rename_map` / camera mismatch | Use `camera_top/front/wrist` keys (see §5 table) |
| Eval window / EGL | `LAB01_MUJOCO_GL=glfw` or `egl` / `osmesa` |
| Sim arm yanks on first frame of each episode (leader record) | Use `reset_arm: follow` (Lab 01 default). `home` is for keyboard/Joy-Con — see §1 |

---

## Lab files

| File | Purpose |
|------|---------|
| `_env.sh` | All Lab 01 env defaults (Shared / Record / Train / Eval / Hub sections) |
| `record.cmd` | Leader pick-and-place recording |
| `train.cmd` | SmolVLA fine-tune |
| `train_smolvla_resume.cmd` | Resume SmolVLA from a checkpoint |
| `train_act.cmd` | ACT imitation learning (`LAB01_ACT_STATE_DIM=6` for joint-pos-only) |
| `train_molmoact2.cmd` | MolmoAct2 fine-tune (MI300X / DORobot defaults) |
| `groot_n1_7_adapt.md` | GR00T N1.7 Lab 01 notes (`new_embodiment`, 6-D joints) |
| `train_groot.cmd` | GR00T N1.7 fine-tune (smoke default 20 steps / bs1) |
| `train_vla_jepa.cmd` | VLA-JEPA fine-tune from LIBERO (transfer-failure recipes in `_env.sh`) |
| `lingbot_vla_v2_adapt.md` | LingBot-VLA 2.0 authoritative path (patch + 193→20K → Home20 25/60) |
| `setup_lingbot_vla_v2_side_tree.cmd` | Fetch PR #3967 side tree + apply inverse-slot patch |
| `eval_lingbot_vla_v2_home20.cmd` | Home20 seeds 1000–1002 for the ranked V2 checkpoint |
| `push_lingbot_vla_v2_agent193_model.cmd` | Upload / refresh Hub 20K weights + card |
| `train_fastwam.cmd` | FastWAM fine-tune from `fastwam_base` (MI300X; freeze video expert) |
| `train_xvla.cmd` | xVLA 10K on the 193-episode set (`LAB01_LEROBOT_SRC`; Home20 47/60) |
| `train_dm05.cmd` | DM0.5 20K on the 193-episode set (`LAB01_LEROBOT_SRC`; Home20 37/60) |
| `eval_home20.cmd` | Home20 seeds 1000/1001/1002 (`rollout_xvla.yaml` / `rollout_dm05.yaml`) |
| `campaign_lingbot_fastwam_10h.cmd` | Time-boxed LingBot-VA + FastWAM train (bs4) then 20-ep EGL eval |
| `configs/` | Lab-local **eval** YAMLs + `cube_positions_demo_fixed.json` |
| `eval.cmd` | Policy-agnostic sim2sim eval (`LAB01_POLICY_PATH` + `LAB01_EVAL_CONFIG`) |
| `hz_bench.cmd` | Headless closed-loop Hz sweep across keepers (wraps `eval.cmd`) |
| `hz_bench/` | `COMPARISON.md` (sync/RTC tables + FAQ), `RESULTS.md`, `summarize.py` |
| `audit_teleop_quality.py` | Phase 0 teleop parquet quality proxies (path / jerk / duration) |
| `auto_collect_poc.cmd` | GLFW PoC: scripted pick-place on .43 (`LAB01_POC_HEADLESS=1` for OSMesa) |
| `collect_yaw.cmd` | One-click yaw-aware dataset: `auto` create-50 / expand-100, `init`, `expand [N]` |
| `auto_collect_yaw.cmd` | Low-level collector (smoke unless `LAB01_YAW_RECORD=1`); used by `collect_yaw.cmd` |
| `estimate_cube_pose.py` / `dataset_stats_viz.py` | Recover cube XY/yaw from the top camera; write `../lab01_pnp_agent/viz/coverage_*.png` |
| `collect_policy_recovery.py` / `collect_policy_takeover.py` | Policy-rollout recorders for the agent-yaw line (ACT follow, DAgger, takeover) |
| `train_upweight_episodes.py` | `lerobot-train` wrapper that repeats the last N episodes' frames |
| `probe_open_loop.py` / `probe_release_open_loop.py` | Open-loop checks of a policy on recorded expert frames |
| `eval_vla_jepa_gui.cmd` | GLFW wrapper for local VLA-JEPA checkpoints |
| `plot_act_loss.py` | Parse `train_act.log` → ACT loss CSV/PNG |
| `push_dataset.cmd` / `hf_dataset_card.md` | Validate + upload dataset to Hub |
| `push_smolvla_model.cmd` / `hf_model_card_smolvla.md` | Upload SmolVLA 50K weights + Hub README |
| `push_act_model_card.cmd` / `hf_model_card_act.md` | Upload ACT 15-D Hub README |
| `push_act_state6_model.cmd` / `hf_model_card_act_state6.md` | Upload ACT 6-D weights + Hub README |
| `push_vla_jepa_model.cmd` / `hf_model_card_vla_jepa.md` | Upload VLA-JEPA 6-D BC 10K + Hub README (failure case) |
| `push_groot_model.cmd` / `hf_model_card_groot.md` | Upload GR00T N1.7 10K + Hub README (negative result) |
| `loss_curve_run1.*` | SmolVLA Run 1 training loss (reference) |
| `loss_curve_act_mi300x_50k.*` | ACT 50K training loss (reference) |
| `lab01_pnp.md` | This runbook |

Agent-generated trajectories, the Home20 pose file, and the 193-episode Hub publish scripts are under [`labs/lab01_pnp_agent/`](../lab01_pnp_agent/lab01_pnp_agent.md). Collectors (`auto_collect_yaw.py`, `collect_policy_*.py`) stay here. Train and eval entry points stay here; the agent lab's `_env.sh` sources this lab's `_env.sh` and adds only Home20 / 193 Hub ids. Old `push_*agent193*` / `push_smolvla_follow2` paths in this directory exec the scripts next door.

Local scratch (not committed): project-root `.tmp/`.

**Config layout (principle):** `labs/lab01_pnp/` holds only **lab-bound** artifacts
(runbook, `*.cmd`, `_env.sh`, Hub cards, training curves, Hz-bench notes, and eval
spawn/demo YAMLs under `labs/lab01_pnp/configs/`). Foundational product features stay at repo root —
e.g. `configs/so101_mujoco_pick_leader.yaml` (leader PnP record),
`configs/scenes/simple_pick/`, generic keyboard/joycon/leader templates — even when
Lab 01 wraps them via `LAB01_*`. Shared lab conventions for future labs:
[labs/README.md](../README.md).
