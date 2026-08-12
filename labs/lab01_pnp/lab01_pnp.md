# Lab 01 — Leader Teleop Pick-and-Place Dataset

Build a MuJoCo expert dataset with the real SO-101 leader arm, validate it, inspect trajectories, replay in sim, then (planned) train SmolVLA and evaluate in MuJoCo.

| Item | Value |
|------|-------|
| Scene | `SO101/scenes/simple_pick/scene.xml` |
| Task | Pick up the cube and place it in the box. |
| Dataset root | `./datasets/so101-simstudio-pnp` |
| Dataset repo_id | `alexhegit/so101-simstudio-pnp` |
| Episodes | 30 |
| Episode length | 60 s max (save early with `N` / `→`) |
| Reset window | 5 s between episodes |
| Record FPS | 20 Hz (leader serial stability) |
| Cameras | `front`, `top`, `wrist` (640×480) |
| Teleop | Leader arm, 1:1 joint position (`action_mode: position`) |

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
  --dataset.root ./datasets/so101-simstudio-pnp \
  --dataset.repo_id alexhegit/so101-simstudio-pnp \
  --dataset.num_episodes 30 \
  --dataset.episode_time_s 60 \
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

**Resume after interruption**

```bash
./labs/lab01_pnp/record.cmd --resume true
```

**Fresh re-record**

```bash
rm -rf ./datasets/so101-simstudio-pnp
./labs/lab01_pnp/record.cmd
```

Log: `test.log` at repo root.

---

## 2. Validate dataset

Automated integrity check (frame count, 20 Hz timestamps, action ranges, videos):

```bash
python -m simstudio.scripts.validate_dataset \
  --root ./datasets/so101-simstudio-pnp
```

Expect `PASSED` with 30 episodes and LeRobot `codebase_version: v3.0`.

Quick metadata:

```bash
python -c "
import json
info = json.load(open('datasets/so101-simstudio-pnp/meta/info.json'))
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
  --repo-id alexhegit/so101-simstudio-pnp \
  --root ./datasets/so101-simstudio-pnp \
  --episode 0
```

Export `.rrd` for smoother playback (avoids live gRPC 1 GiB memory drops):

```bash
python -m simstudio.scripts.dataset_viz \
  --repo-id alexhegit/so101-simstudio-pnp \
  --root ./datasets/so101-simstudio-pnp \
  --episode 0 \
  --save --output-dir ./outputs/viz

rerun --memory-limit 4GB ./outputs/viz/alexhegit_so101-simstudio-pnp_episode_0.rrd
```

### MP4 spot-check (smoothest video preview)

```bash
ffplay -autoexit datasets/so101-simstudio-pnp/videos/observation.images.camera_front/chunk-000/file-000.mp4
```

**Note:** Rerun live mode can look accelerated or jumpy when uncompressed RGB exceeds the viewer memory limit. That is a visualization artifact, not a dataset defect.

---

## 4. Replay in MuJoCo

Replay recorded joint trajectories in the same sim scene.

**Single episode**

```bash
python -m simstudio.scripts.replay \
  --config configs/so101_mujoco_replay.yaml \
  --dataset.root ./datasets/so101-simstudio-pnp \
  --dataset.repo_id alexhegit/so101-simstudio-pnp \
  --dataset.episode 0
```

**All episodes**

```bash
python -m simstudio.scripts.replay_multi \
  --config configs/so101_mujoco_replay_multi.yaml \
  --dataset.root ./datasets/so101-simstudio-pnp \
  --dataset.repo_id alexhegit/so101-simstudio-pnp \
  --dataset.episodes all
```

---

## 5. Train SmolVLA

Fine-tune `lerobot/smolvla_base` on the lab dataset. Requires validated data and `.venv-rocm` (`make rocm-sync`).

**One command:**

```bash
source .venv-rocm/bin/activate
./labs/lab01_pnp/train.cmd
```

**Manual equivalent** (same defaults as `train.cmd`):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --policy.scheduler_warmup_steps=500 \
  --dataset.repo_id=alexhegit/so101-simstudio-pnp \
  --dataset.root=./datasets/so101-simstudio-pnp \
  --dataset.video_backend=pyav \
  --output_dir=./outputs/train/lab01_pnp_smolvla \
  --job_name=lab01_pnp_smolvla \
  --rename_map='{"observation.images.camera_top":"observation.images.camera1","observation.images.camera_front":"observation.images.camera2","observation.images.camera_wrist":"observation.images.camera3"}' \
  --batch_size=4 \
  --steps=7500 \
  --save_checkpoint=true \
  --save_freq=2000
```

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

Log: `train.log` at repo root. Checkpoints: `./outputs/train/lab01_pnp_smolvla/checkpoints/` (override output dir with `LAB01_TRAIN_OUTPUT=...`).

**Notes for 30-episode dataset**

- Default **`batch_size=4`, `steps=7500`** (~30k sample updates); see reference run below for wall time on Strix Halo.
- OOM: `--batch_size 1` or `LAB01_TRAIN_BATCH_SIZE=1`.
- First run downloads HF weights; needs network.
- MuJoCo closed-loop eval is not wired yet (see §6).

### Reference platform & software (Run 1)

Measured on the machine used for the first full lab training run:

| Item | Value |
|------|-------|
| Machine | AMD Strix Halo Laptop (Ryzen AI MAX+ 395) |
| GPU | AMD Radeon 8060S (iGPU) |
| PyTorch | 2.13 |
| ROCm | 7.2 (`.venv-rocm` via `make rocm-sync`) |
| LeRobot | 0.6.0 (pinned submodule commit `30da8e68`) |
| Video decode | `pyav` (`torchcodec` fails on ROCm) |

### Default training settings (`train.cmd`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base policy | `lerobot/smolvla_base` | Use `--policy.path`, not `pretrained_path` |
| `batch_size` | **4** | ~2.88 GB VRAM on Run 1; reduce to 1 if OOM |
| `steps` | **7500** | ~30k sample updates (7500 × batch 4) |
| `policy.scheduler_warmup_steps` | 500 | |
| `policy.empty_cameras` | 1 | Pads fourth camera slot expected by base model |
| `policy.push_to_hub` | false | Avoids HF repo_id requirement |
| `dataset.video_backend` | pyav | Required on ROCm |
| `save_freq` | 2000 | Checkpoints at 2k / 4k / 6k / final |
| Dataset | 30 ep, 14265 frames @ 20 Hz | `alexhegit/so101-simstudio-pnp` |
| Output (Run 1) | `./outputs/train/lab01_pnp_smolvla_bs4` | Set via `LAB01_TRAIN_OUTPUT=...` |

Reproduce Run 1:

```bash
source .venv-rocm/bin/activate
LAB01_TRAIN_OUTPUT=./outputs/train/lab01_pnp_smolvla_bs4 ./labs/lab01_pnp/train.cmd
```

### Training run log

| Run | Platform | batch | steps | wall time | step/s | VRAM | final loss | checkpoint |
|-----|----------|-------|-------|-----------|--------|------|------------|------------|
| 1 | Strix Halo / 8060S iGPU | 4 | 7500 | **1h 54m** (2026-08-10 22:55 → 00:49) | ~**1.10** | ~**2.88 GB** | **0.159** (step 7400) | `./outputs/train/lab01_pnp_smolvla_bs4/checkpoints/007500/` |

**Run 1 details**

- **Speed:** ~1.10 step/s sustained (~4 samples/s with batch 4); `updt_s` ~0.90 s/step in logs.
- **Loss curve:** dropped through warmup, then plateaued ~**0.15–0.16** from step 6600 onward; last logged value **0.159** at step 7400 (logs every 200 steps).
- **Loss plot:** `labs/lab01_pnp/loss_curve_run1.png` (raw points: `loss_curve_run1.csv`, parsed from `train.log`).
- **Checkpoints saved:** `002000`, `004000`, `006000`, `007500`, plus `last → 007500`.
- **Inference path:** `./outputs/train/lab01_pnp_smolvla_bs4/checkpoints/007500/pretrained_model/`

### ACT training (Run 1)

```bash
source .venv-rocm/bin/activate
./labs/lab01_pnp/train_act.cmd
```

| Run | Platform | batch | steps | wall time | step/s | VRAM | final loss | checkpoint |
|-----|----------|-------|-------|-----------|--------|------|------------|------------|
| 1 | Strix Halo / 8060S iGPU | 8 | 10000 | **~5h 30m** | ~**0.50** | ~**5.8 GB** | **0.249** (step 10000) | `./outputs/train/lab01_pnp_act/checkpoints/010000/` |
| 2 (resume) | Strix Halo / 8060S iGPU | 8 | 30000 | **~10h 50m** (10k→30k) | ~**0.51** | ~**5.8 GB** | **0.130** (step 30000) | `./outputs/train/lab01_pnp_act/checkpoints/030000/` |

Also saved: Run 1 `005000`; Run 2 `020000`, `030000`, `last → 030000`. Logs: `train_act.log`, `train_act_run2.log`.

**ACT eval (20 episodes, headless, `reset_arm: follow`):** checkpoint `030000` → **7/20 (35%)**. Successful episodes place the cube in the container (~0.30, 0.19–0.21).

---

## 6. Policy eval in MuJoCo (sim2sim)

Closed-loop rollout: load a fine-tuned checkpoint, run inference at 20 Hz in MuJoCo with the same three cameras and position-control actions as training.

### Quick eval (success rate)

```bash
source .venv-rocm/bin/activate
./labs/lab01_pnp/eval.cmd
```

Defaults: Run 1 checkpoint, **10 episodes × 60 s**, random cube reset, RTC inference (SmolVLA on iGPU).

Manual equivalent:

```bash
python -m simstudio.scripts.eval \
  --config configs/so101_mujoco_rollout.yaml \
  --policy.path=./outputs/train/lab01_pnp_smolvla_bs4/checkpoints/007500/pretrained_model \
  --eval.num_episodes=10
```

Success criterion: cube center inside container bounds in `simple_pick` scene (see `simstudio.common.eval_success`).

### Single continuous rollout (no success tally)

```bash
python -m simstudio.scripts.rollout \
  --config configs/so101_mujoco_rollout.yaml \
  --strategy.type=base \
  --policy.path=./outputs/train/lab01_pnp_smolvla_bs4/checkpoints/007500/pretrained_model \
  --duration=60
```

### Options

| Override | Example |
|----------|---------|
| Checkpoint | `LAB01_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp/eval.cmd` |
| Episode count | `LAB01_EVAL_EPISODES=5 ./labs/lab01_pnp/eval.cmd` |
| Sync inference (debug) | `./labs/lab01_pnp/eval.cmd --inference.type sync` |
| Rerun viz | `./labs/lab01_pnp/eval.cmd --display_data true` |

**Camera rename map** (same as training §5): `camera_top/front/wrist` → `camera1/2/3`.

If inference is slower than 20 Hz, keep `--inference.type=rtc` (default in `so101_mujoco_rollout.yaml`).

### ACT eval

```bash
./labs/lab01_pnp/eval_act.cmd   # 20 episodes, headless, sync inference
```

Uses `configs/so101_mujoco_rollout_act.yaml` (no camera rename_map; dataset camera names directly).

| Policy | Episodes | Success |
|--------|----------|---------|
| SmolVLA 7500-step (GUI) | 10 | 0/10 (0%) — arm reaches cube, grasp fails |
| ACT 10000-step (headless) | 20 | **0/20 (0%)** |
| ACT 10000-step (`reset_arm: follow`, headless) | 10 | **0/10 (0%)** — reset 对齐后仍无成功 |
| ACT 30000-step (`reset_arm: follow`, headless) | 20 | **7/20 (35%)** — default `n_action_steps=100` |
| ACT 30000-step inference sweep (headless, 20 ep each) | | |
| → `n_action_steps=5` | 20 | 0/20 (0%) |
| → `n_action_steps=10` | 20 | 1/20 (5%) |
| → `n_action_steps=20` | 20 | 4/20 (20%) |
| → `n_action_steps=50` | 20 | 5/20 (25%) |
| → `n_action_steps=1` + `temporal_ensemble_coeff=0.01` | 20 | 4/20 (20%) |

Sweep logs: `labs/lab01_pnp/sweep_logs/` (2026-08-12). **Default `n_action_steps=100` remains best** — shorter replan intervals hurt (model trained for 100-step open-loop chunks).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `feetech-servo-sdk` missing | `uv pip install --python .venv-rocm/bin/python 'feetech-servo-sdk>=1.0.0,<2.0.0'` |
| `/dev/ttyACM0` permission denied | Add user to `dialout`, re-login |
| Terminal closes immediately | Run `./labs/lab01_pnp/record.cmd` inside an open terminal, not by double-clicking |
| Rerun viewer not found | `source .venv-rocm/bin/activate` or use `dataset_viz --save` |
| Rerun playback jumpy | Use `--save` + `rerun --memory-limit 4GB`, or `ffplay` on dataset mp4 |
| Used `smoke-leader-teleop` for Rerun | Rerun only works with **record**, not `teleoperate` |
| SmolVLA training OOM | `./labs/lab01_pnp/train.cmd --batch_size 1` |
| `rename_map` / camera mismatch | Use `camera_top/front/wrist` keys (see §5 table) |

---

## Lab files

| File | Purpose |
|------|---------|
| `record.cmd` | One-command leader pick-and-place recording for this lab |
| `train.cmd` | One-command SmolVLA fine-tune from `smolvla_base` |
| `train_act.cmd` | One-command ACT imitation learning on pnp dataset |
| `eval.cmd` | One-command sim2sim SmolVLA eval (multi-episode success rate) |
| `eval_act.cmd` | One-command sim2sim ACT eval (20 episodes, headless) |
| `lab01_pnp.md` | This runbook |

Related repo configs: `configs/so101_mujoco_pick_leader.yaml`
