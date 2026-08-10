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

## 5. Train SmolVLA — planned

Not run yet for this lab dataset. Planned workflow on ROCm:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=alexhegit/so101-simstudio-pnp \
  --dataset.root=./datasets/so101-simstudio-pnp \
  --output_dir=./outputs/train/lab01_pnp_smolvla \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.front":"observation.images.camera2","observation.images.wrist":"observation.images.camera3"}' \
  --job_name=lab01_pnp_smolvla
```

Tune `batch_size`, `steps`, and `eval_freq` for available VRAM. Requires `.venv-rocm` with `lerobot[smolvla]` (installed by `make rocm-sync`).

---

## 6. Policy eval in MuJoCo — planned

Closed-loop rollout in sim is **not implemented yet** (see `ROADMAP.md`: policy rollout in MuJoCo). Planned steps:

1. Load checkpoint from `./outputs/train/lab01_pnp_smolvla`
2. Run inference at 20 Hz against `so101_mujoco` with the same three cameras
3. Measure success rate on random cube resets (`reset_cube: random`)

Placeholder (API may change when rollout wrapper lands):

```bash
# TODO: replace with simstudio rollout script once available
lerobot-eval \
  --policy.path=./outputs/train/lab01_pnp_smolvla/checkpoints/.../pretrained_model \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.front":"observation.images.camera2","observation.images.wrist":"observation.images.camera3"}'
```

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

---

## Lab files

| File | Purpose |
|------|---------|
| `record.cmd` | One-command leader pick-and-place recording for this lab |
| `lab01_pnp.md` | This runbook |

Related repo configs: `configs/so101_mujoco_pick_leader.yaml`
