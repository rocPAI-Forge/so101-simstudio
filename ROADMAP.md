# ROADMAP

## v0.0.1 ✅ (`release-v0.0.1`)

- [x] MuJoCo SO-101 scene
- [x] Keyboard teleop (Z/X for Z axis)
- [x] Recording controls (Left/R, Right/N, ESC/Q)
- [x] LeRobot dataset recording
- [x] Single / multi-episode replay
- [x] ROCm environment setup
- [x] Headless smoke test
- [x] Zero submodule modifications
- [x] Plugin registration wrappers

## v0.0.2 ✅ (`release-v0.0.2`)

### Hardware-in-the-loop

- [x] Leader arm teleop (Feetech STS3215, direct position)
- [x] Leader calibration (LeRobot SOLeader)

### Engineering

- [x] Dual control paradigms (velocity / position)
- [x] Auto-scaling (normalized → MuJoCo radians)
- [x] `teleoperate` script
- [x] `llm-wiki` knowledge base

## v0.0.3 ✅ (merged into v0.1.0; no separate tag)

- [x] Post-record validation
- [x] Joy-Con teleop (velocity paradigm, left/right)

## v0.1.0 ✅ (`release-v0.1.0`)

- [x] Rename to **so101-simstudio** (Python package: `simstudio`)
- [x] Keyboard, Joy-Con, leader arm teleop
- [x] Record / replay / validate pipeline

## v0.1.1 ✅ (`release-v0.1.1`)

### Recording display

- [x] `--view_mode mujoco | rerun` on record script
- [x] Both modes write LeRobot dataset v3.0
- [x] Rerun streaming patch (live camera updates)

### Input

- [x] Evdev focus-independent keyboard (Rerun / Wayland)
- [x] Leader / Joy-Con evdev recording controls (N/R/Q)
- [x] Recording control debounce (prevents key bursts from skipping episodes)

### Leader arm

- [x] Stable teleop/record sync; safe gripper mapping
- [x] `reset_arm: home|follow`, `reset_cube: fixed|random|none`

### Engineering

- [x] Smoke scripts → `scripts/smoke/` + `make smoke-*`
- [x] Tests: `test_record_view_mode`, `test_keyboard_teleop`, `test_episode_reset`

## v0.1.2 ✅ (`release-v0.1.2`)

### Joy-Con teleop overhaul

- [x] Direct HID stick/button reading (decoupled from joycon-robotics position integration)
- [x] Cylindrical control: stick forward/back = reach, left/right = base swing
- [x] True `shoulder_pan` anchor for radial direction (fixes inconsistent extend vs swing)
- [x] IMU rotation deadzone / clamp / settle window
- [x] Gripper toggle mode (`gripper_toggle: true`, press ZR/ZL to latch)
- [x] One-handed recording: A/Y/+ (right), d-pad/Minus (left); coexists with keyboard evdev

### Project hygiene

- [x] Remove redundant root `test_*.sh` delegators
- [x] Add `scripts/quicktest/*.cmd` for fixed collaboration runs
- [x] Documentation sync (English)
- [x] English user-facing prints and logs
- [x] joycon-robotics: pin upstream submodule; apply `patches/joycon-robotics.patch` via `make joycon-sync` (no in-submodule project commits)
- [x] GitHub repository `alexhegit/so101-simstudio`

## v0.1.3 ✅ (`release-v0.1.3`)

### Lab 01 — pick-and-place train / eval

- [x] Leader PnP lab runbook (`labs/lab01_pnp/`) with centralized `_env.sh`
- [x] Record → validate → ACT / SmolVLA train → MuJoCo sim2sim eval scripts
- [x] Lab-local eval YAMLs under `labs/lab01_pnp/configs/` (demo / fixed / full-range)
- [x] Single policy-agnostic `eval.cmd` (SmolVLA + ACT via `LAB01_POLICY_PATH` / `LAB01_EVAL_CONFIG`)
- [x] Document `reset_arm: home|follow` (keyboard/Joy-Con vs leader; intentional eval contrast)
- [x] Hub reference assets: dataset + ACT + SmolVLA (`alexhegit/so101-simstudio-lab01-pnp*`)

### Scene / cameras

- [x] Wrist camera aligned with Sim-to-Real Isaac `gripper_cam`
- [x] Swapped cube spawn vs container layout for Lab 01 dataset
- [x] Lab dataset id `so101-simstudio-lab01-pnp` (deprecate v1 `so101-simstudio-pnp`)

### Platform / docs

- [x] ROCm-only install guidance (no CUDA/macOS path in this release)
- [x] Joy-Con init guards (invalid device / missing serial)
- [x] Headless MuJoCo camera backend notes (`llm-wiki`)

## v0.1.4 ✅ (`release-v0.1.4`)

### Agent dataset (193 episodes)

- [x] Scripted collectors sample cube XY and yaw inside the record box; the wrist follows the cube yaw
- [x] Adopted set: 100 one-shot grasps + 10 pre-close alignments + 10 local regrasps + 20 DAgger + 53 ACT-follow successes = **193 episodes / 130,835 frames**
- [x] ACT 6-D trained on the first 140 only (Home20 **53/60**). Its successful `follow` rollouts are the last 53 episodes, used to finetune other policies
- [x] Hub dataset `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`. The v0.1.3 50-episode teleop set stays the human reference

### Finetune trials

- [x] Camera map recorded per policy (top+wrist or front). The dataset still stores the front camera
- [x] 6-D absolute joint targets for sim2real-style policies. GR00T relative actions stayed near 0/60; absolute joints are the Home20 frame
- [x] Frozen backbone, action-expert-only, LoRA, and full updates. pi05 full unfreeze **34/60** (expert-only was 12/60). MolmoAct2 LoRA **39/60**; the later full update fell to 23/60 and was not adopted
- [x] xVLA full 10K with three cameras and a 20-D pretrained head adapted to 6-D absolute SO-101 joints: Home20 **47/60**. The finetune is local, not uploaded
- [x] Uploaded checkpoints linked from [home20.md](labs/lab01_pnp_agent/home20.md), including DM0.5 20K at **37/60** (the historical Hub id still ends in `10k`)

### Home20

- [x] One protocol: 20 frozen poses, `reset_arm: home`, gripper start 0.85, seeds 1000 / 1001 / 1002
- [x] Home20 totals compare recipes on that protocol. They are not a ranking of model families
- [x] Success is `final_success`. Report each seed /20 and the total /60
- [x] Full-range redraws, the single demo pose, and gripper −0.1 runs stay off that column
- [x] `LAB01_EPISODE_INIT` applies the frozen list over the random cube and the YAML home

### Other

- [x] Agent line lives in `labs/lab01_pnp_agent/`; train/eval entry points stay in `labs/lab01_pnp/`
- [x] Lab 01 entry points for MolmoAct2, pi05, GR00T N1.7, VLA-JEPA, FastWAM
- [x] Instinct MI210 / MI300 / MI300X eval notes: `MUJOCO_GL=osmesa`
- [x] MuJoCo GLFW window no longer vsync-blocks the control loop
- [x] Directories named `tmp/` stay untracked
- [x] Pinned LeRobot remains v0.6.0. DM0.5 and xVLA are not in that pin

## Evolution

```
v0.0.x: so101-mujoco-teleop
        ↓ v0.1.0
        ↓ v0.1.1 (unified view_mode + evdev)
        ↓ v0.1.2 (`release-v0.1.2`) Joy-Con cylindrical + one-handed recording + joycon patch workflow
        ↓ v0.1.3 (`release-v0.1.3`) Lab 01 PnP: train ACT/SmolVLA + MuJoCo sim2sim eval + Hub assets
        ↓ v0.1.4 (`release-v0.1.4`) 193-episode agent set, per-model finetune trials, Home20
Current: so101-simstudio @ v0.1.4
        Expert trajectories + Lab 01 closed-loop train/eval (ROCm)
        Future: domain randomization, broader automated pipeline, real follower, CUDA/macOS
```

## Future work

### Platform support

- [ ] **NVIDIA CUDA (Linux)** — not supported yet (unverified); needs documented install path, PyTorch CUDA wheels, smoke tests
- [ ] **macOS / Apple Silicon (MPS)** — not supported yet (unverified); needs documented install path, GLFW / input quirks, smoke tests

**Current release:** only **Ubuntu 24.04 + AMD ROCm** (`make rocm-sync` → `.venv-rocm`) is supported for simulation, recording, validation, and VLA training.

### Automated data pipeline

- [ ] Dataset annotation (VLM language labels)
- [x] Behavior cloning training — Lab 01 ACT / SmolVLA, plus xVLA, MolmoAct2, pi05, GR00T, DM0.5 and the negative JEPA / FastWAM transfers
- [x] Policy rollout in MuJoCo — Lab 01 sim2sim eval (`eval.cmd` + lab configs)
- [x] Agent-generated Lab 01 trajectories and the Home20 benchmark (`labs/lab01_pnp_agent/`)
- [ ] Broader multi-task / multi-scene automated pipeline beyond Lab 01

### Real hardware

- [ ] Real SO-101 follower driver (stub → production)
