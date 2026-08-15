# Lab 01 — Leader Teleop Pick-and-Place Dataset

Build a MuJoCo expert dataset with the real SO-101 leader arm, validate it, inspect trajectories, replay in sim, then train ACT/SmolVLA and evaluate in MuJoCo.

> **Dataset v1 deprecated:** `so101-simstudio-pnp` (old wrist camera + original cube/container layout) is no longer used. Lab 01 now uses **`so101-simstudio-lab01-pnp`** only (new wrist cam, swapped spawn layout).

| Item | Value |
|------|-------|
| Scene | `SO101/scenes/simple_pick/scene.xml` |
| Task | Pick up the cube and place it in the box. |
| Dataset root | `./datasets/so101-simstudio-lab01-pnp` |
| Dataset (Hub) | [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp) (50 leader PnP episodes) |
| SmolVLA (Hub) | [alexhegit/so101-simstudio-lab01-pnp-smolvla](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla) (MI300X bs64 @ 50K) |
| ACT (Hub) | [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act) (MI300X 50K) |
| Episodes | 50 |
| Episode length | 90 s max (save early with `N` / `→`) |
| Reset window | 5 s between episodes |
| Record FPS | 20 Hz (leader serial stability) |
| Cameras | `front`, `top`, `wrist` (640×480, wrist aligned Isaac `gripper_cam`) |
| Teleop | Leader arm, 1:1 joint position (`action_mode: position`) |

Skip recording/training if you only need eval: download the Hub dataset + policy (§7; `alexhegit` examples, or your own HF id), then run `./labs/lab01_pnp/eval.cmd` with `LAB01_POLICY_PATH` pointed at the downloaded weights.

**Script defaults** live in [`labs/lab01_pnp/_env.sh`](_env.sh) (centralized, sectioned:
Shared / Record / Train / Train ACT / Eval / Hub). Override via environment:

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
| `LAB01_ACT_STEPS` / `BATCH_SIZE` / `SAVE_FREQ` | `10000` / `8` / `10000` | train_act (short default; MI300X reference used 50K / 128) |
| `LAB01_POLICY_PATH` | see `_env.sh` (any `pretrained_model/` path) | eval / push_smolvla_model |
| `LAB01_MUJOCO_GL` | `egl` | eval (`egl` / `glfw` / `osmesa`) |
| `LAB01_RENDER_WINDOW` | `false` | eval (`true` with glfw) |
| `LAB01_EVAL_CONFIG` | `labs/lab01_pnp/configs/rollout_smolvla_demo.yaml` | eval spawn yaml (see §6) |
| `LAB01_EVAL_EPISODES` | `10` | eval |
| `LAB01_EVAL_LOG` | `./outputs/eval/eval_${MUJOCO_GL}.log` | eval |
| `LAB01_N_ACTION_STEPS` | empty (checkpoint default) | eval; set e.g. `50` for ACT |
| `LAB01_ACT_HF_REPO_ID` | `alexhegit/so101-simstudio-lab01-pnp-act` | push_act_model_card |
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

### ACT training (lab01-pnp, MI300X 50K) — current reference

```bash
LAB01_ACT_BATCH_SIZE=128 LAB01_ACT_STEPS=50000 LAB01_ACT_SAVE_FREQ=10000 \
  ./labs/lab01_pnp/train_act.cmd
```

| Run | Platform | batch | steps | final loss | checkpoint |
|-----|----------|-------|-------|------------|------------|
| lab01-pnp | DORobot / MI300X | 128 | 50000 | **0.053** | `./outputs/train/lab01_pnp_act/checkpoints/050000/` |

**Loss curve** (from `train_act.log`, not eval): `labs/lab01_pnp/loss_curve_act_mi300x_50k.png` / `.csv`.

Regenerate after a new ACT run:

```bash
.venv-rocm/bin/python labs/lab01_pnp/plot_act_loss.py \
  --log train_act.log --steps 50000 --tag mi300x_50k
```

**ACT eval:** measured rates and configs are in §6 (do not treat loss alone as a proxy for pick success).

---

## 6. Policy eval in MuJoCo (sim2sim)

One entry point: **`./labs/lab01_pnp/eval.cmd`**. Swap `LAB01_POLICY_PATH` + matching
`LAB01_EVAL_CONFIG` for SmolVLA, ACT, or another LeRobot policy; optional
`LAB01_N_ACTION_STEPS` when you need to override the checkpoint.

Closed-loop rollout: load a fine-tuned checkpoint, run inference at 20 Hz in MuJoCo with the same cameras and position-control actions as training.

**Success criterion:** cube center inside the container bounds of the `simple_pick` scene (see `simstudio.common.eval_success`). Episodes that grasp but drop mid-transfer, knock the cube away, or leave it on the container rim count as **fail**.

### 6.1 What you are measuring

Three axes change the reported success rate. Compare only runs that match on all three, or call out the difference explicitly.

| Axis | Choices | Meaning |
|------|---------|---------|
| **Policy** | SmolVLA 50K (`lab01_pnp_smolvla`) vs ACT 50K (`lab01_pnp_act`) | Different architectures / training recipes on the **same** 50-episode lab dataset |
| **Cube spawn** | Full-range random / narrowed demo random / fixed pose | Distribution of initial cube `(x, y, yaw)` at episode reset |
| **Inference** | `rtc` vs `sync`; ACT `n_action_steps` | How actions are produced relative to the 20 Hz control loop |

**Honest generalization** uses the recording spawn box. **Demo configs** deliberately shrink that box (or freeze one pose) so a classroom run looks more reliable without retraining. Demo rates are **not** interchangeable with full-range rates.

### 6.2 Eval configs (spawn)

Lab **eval** YAMLs (demo/fixed spawn, ACT vs SmolVLA rename) live under
**`labs/lab01_pnp/configs/`** — lab-specific measurement setups, not product defaults.
Repo-root `configs/` keeps foundational record/replay/scene assets (including
`configs/so101_mujoco_pick_leader.yaml` used by Lab 01 recording) and generic templates.


Recording / full-range box (leader teleop):  
`x ∈ [0.26, 0.34]`, `y ∈ [0.165, 0.235]`, `yaw ∈ [-45°, 45°]`.

| Config | Policy | `reset_cube` | `reset_arm` | Spawn | Role |
|--------|--------|--------------|------------|-------|------|
| `labs/lab01_pnp/configs/rollout_smolvla.yaml` | SmolVLA | `random` | `home` | Full record box | Honest SmolVLA benchmark |
| `labs/lab01_pnp/configs/rollout_smolvla_demo.yaml` | SmolVLA | `random` | `home` | `x∈[0.265,0.29]`, `y∈[0.175,0.205]`, `yaw∈[-15°,0°]` | Narrowed random demo (default `eval.cmd`) |
| `labs/lab01_pnp/configs/rollout_smolvla_demo_fixed.yaml` | SmolVLA | `fixed` | `home` | Always `(0.27, 0.20, yaw −8°)` via `cube_positions_demo_fixed.json` | Fixed-pose SmolVLA demo |
| `labs/lab01_pnp/configs/rollout_act.yaml` | ACT | `random` | `follow` | Full record box | Honest ACT benchmark |
| `labs/lab01_pnp/configs/rollout_act_demo_fixed.yaml` | ACT | `fixed` | `follow` | Same pose as SmolVLA fixed demo | Fixed-pose ACT demo |

**`reset_arm` rule of thumb** (same meanings as in §1 recording):

| Teleop / mode | Prefer | Why |
|---------------|--------|-----|
| Real **leader** recording | `follow` | No home teleport → no first-frame yank under 1:1 joint mapping |
| **Keyboard / Joy-Con** recording | `home` | No physical leader; fixed start pose each episode |
| Policy **eval** (no teleop) | Either; document which | `home` = every episode starts from the fixed home pose (easier to reason about). `follow` = leave the arm where the previous episode ended (matches leader-record style; episode starts are path-dependent). |

Lab 01 **keeps both on purpose** in the eval YAMLs above: SmolVLA → `home`, ACT → `follow`. That is an **intentional side-by-side example** of the two reset styles (not an ACT-vs-SmolVLA algorithm requirement). Use it to see how episode-start arm pose changes closed-loop behavior; when quoting §6.4, keep each policy paired with the `reset_arm` in the table (or re-measure after changing it). Do **not** “unify” these files unless you want a single protocol for a new measurement campaign.

Fixed pose freezes **both** translation and yaw. Episode-to-episode outcomes can still differ because policies (especially SmolVLA flow sampling) are stochastic and because inference latency / chunking change the closed loop.

**SmolVLA** configs include the training `rename_map` (`camera_top/front/wrist` → `camera1/2/3`). **ACT** configs must **not** use that map — use the `rollout_act*.yaml` files.

### 6.3 Inference backends

| Backend | Config / flag | Behavior | When it matters |
|---------|---------------|----------|-----------------|
| **RTC** | `inference.type: rtc` (SmolVLA rollout defaults) | Async action chunks in a background thread | Needed when policy latency ≫ control period on real robots; in sim, backlog shows as `Indexes diff is not equal to real delay` |
| **Sync** | `inference.type: sync` (ACT rollout default; override with `--inference.type=sync`) | Inline `select_action` on the control thread | Removes RTC queue lag in sim; wall-clock still ~60 s/episode if each episode is capped at 60 s |

Optional **`LAB01_N_ACTION_STEPS`**: when set, passed as `--policy.n_action_steps`. Leave empty to use the value stored in the checkpoint (SmolVLA 50K: 50; ACT 50K train: 100). Lab ACT full-range reference used `50`.


### 6.4 Reference measurements (lab01-pnp, MI300X 50K checkpoints)

Numbers below are closed-loop MuJoCo evals on the lab01-pnp scene. **n** is small for demo/fixed runs; treat those as indicative, not precise CI estimates. Full-range rows are the primary generalization numbers.

| Policy | Spawn | `reset_arm` | Inference | Backend | Episodes | Success | Notes |
|--------|-------|-------------|-----------|---------|----------|---------|-------|
| SmolVLA 50K | Full-range random | `home` | RTC | EGL | 50 | **11/50 (22%)** | Most fails never leave spawn; grasp/close unreliable |
| SmolVLA 50K | Demo random (early box `x≤0.30`, yaw ±15°) | `home` | RTC | EGL | 20 | **5/20 (25%)** | Narrowing alone did not fix grasp failures |
| SmolVLA 50K | Fixed `(0.27,0.20,−8°)` | `home` | RTC | GLFW | 10 | **5/10 (50%)** | Same pose; RTC delay warnings throughout |
| SmolVLA 50K | Fixed `(0.27,0.20,−8°)` | `home` | Sync | GLFW | 10 | **3/10 (30%)** | Fewer “knock away”; more mid-transfer drops |
| ACT 50K | Full-range random | `follow` | Sync | EGL | 50 | **32/50 (64%)** | Reference with `LAB01_N_ACTION_STEPS=50` |
| ACT 50K | Full-range random | `follow` | Sync | — | 50 | **23/50 (46%)** | Same checkpoint with `n_action_steps=100` (historical log) |
| ACT 50K | Fixed `(0.27,0.20,−8°)` | `follow` | Sync | GLFW | 10 | **8/10 (80%)** | `n_action_steps=100`, `rollout_act_demo_fixed.yaml` |

**Reading the table**

- `reset_arm` matches the intentional §6.2 contrast (SmolVLA eval YAMLs → `home`, ACT → `follow`). Do not mix rows when claiming a single protocol.
- On this 50-episode dataset, **ACT full-range ≫ SmolVLA full-range** under the measured setups.
- **Demo / fixed spawn raises ACT demo reliability** (80% @ fixed) but does **not** make SmolVLA classroom-stable (≈30–50% even at a known-good pose).
- For SmolVLA, spawn tightening helped less than expected: failure analysis pointed to **grasp instability** and **yaw polarity** (positive yaw often knocks the cube) more than XY coverage alone.
- RTC lag on iGPU (≈8–9 control frames) changes *how* SmolVLA fails; switching to sync did not turn fixed-pose eval into a high success rate.

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
6. **Policy choice for demos** — If the goal is a reliable classroom pick-and-place, ACT + fixed (or narrowed) spawn is currently stronger on this dataset; keep SmolVLA if the learning goal is VLA fine-tuning and analysis of its failure modes.

When you change any axis, re-run with a fixed episode count and the same `MUJOCO_GL` / inference settings, and record config path + checkpoint step next to the success fraction.

### Historical notes (v1 / short ACT runs)

Older ACT runs on deprecated data or shorter schedules (10K/30K) are not the lab01-pnp reference. Prefer the MI300X **50K** checkpoints and the §6.4 table above.

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
| ACT model repo | `LAB01_ACT_HF_REPO_ID` | `your-hf-id/so101-simstudio-lab01-pnp-act` |

Published Lab 01 reference assets (`alexhegit`):

| Asset | Hub repo |
|-------|----------|
| Dataset | [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp) |
| SmolVLA policy | [alexhegit/so101-simstudio-lab01-pnp-smolvla](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla) |
| ACT policy | [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act) |

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

# ACT
hf download alexhegit/so101-simstudio-lab01-pnp-act \
  --local-dir ./outputs/hub/lab01_pnp_act
```

**Eval with downloaded weights** (same `eval.cmd`; see §6):

```bash
# SmolVLA demo spawn
LAB01_POLICY_PATH=./outputs/hub/lab01_pnp_smolvla \
  ./labs/lab01_pnp/eval.cmd

# ACT full-range
LAB01_POLICY_PATH=./outputs/hub/lab01_pnp_act \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
LAB01_EVAL_LOG=./outputs/eval/act_hub.log \
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

# Dataset
./labs/lab01_pnp/push_dataset.cmd

# SmolVLA weights + card (MI300X bs64 @ 50K)
./labs/lab01_pnp/push_smolvla_model.cmd

# ACT model card
./labs/lab01_pnp/push_act_model_card.cmd
```

Cards: `hf_dataset_card.md`, `hf_model_card_smolvla.md`, `hf_model_card_act.md`.

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
| `train_act.cmd` | ACT imitation learning |
| `configs/` | Lab-local **eval** YAMLs + `cube_positions_demo_fixed.json` |
| `eval.cmd` | Policy-agnostic sim2sim eval (`LAB01_POLICY_PATH` + `LAB01_EVAL_CONFIG`) |
| `plot_act_loss.py` | Parse `train_act.log` → ACT loss CSV/PNG |
| `push_dataset.cmd` / `hf_dataset_card.md` | Validate + upload dataset to Hub |
| `push_smolvla_model.cmd` / `hf_model_card_smolvla.md` | Upload SmolVLA 50K weights + Hub README |
| `push_act_model_card.cmd` / `hf_model_card_act.md` | Upload ACT Hub README |
| `loss_curve_run1.*` | SmolVLA Run 1 training loss (reference) |
| `loss_curve_act_mi300x_50k.*` | ACT 50K training loss (reference) |
| `lab01_pnp.md` | This runbook |

Local scratch (not committed): project-root `.tmp/`.

**Config layout (principle):** `labs/lab01_pnp/` holds only **lab-bound** artifacts
(runbook, `*.cmd`, `_env.sh`, Hub cards, training curves, and eval spawn/demo YAMLs
under `labs/lab01_pnp/configs/`). Foundational product features stay at repo root —
e.g. `configs/so101_mujoco_pick_leader.yaml` (leader PnP record),
`configs/scenes/simple_pick/`, generic keyboard/joycon/leader templates — even when
Lab 01 wraps them via `LAB01_*`. Shared lab conventions for future labs:
[labs/README.md](../README.md).
