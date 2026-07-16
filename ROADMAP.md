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

### Leader arm

- [x] Stable teleop/record sync; safe gripper mapping
- [x] `reset_arm: home|follow`, `reset_cube: fixed|random|none`

### Input (recording)

- [x] Recording control debounce (prevents key bursts from skipping episodes)

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

## Evolution

```
v0.0.x: so101-mujoco-teleop
        ↓ v0.1.0
        ↓ v0.1.1 (unified view_mode + evdev)
        ↓ v0.1.2 (`release-v0.1.2`) Joy-Con cylindrical + one-handed recording + joycon patch workflow
Current: so101-simstudio @ v0.1.2
        Expert trajectory generation (teleop, scenes, record, replay, validate)
        Future: domain randomization, policy rollout, real follower
```

## Future work

### Automated data pipeline

- [ ] Dataset annotation (VLM language labels)
- [ ] Behavior cloning training
- [ ] Policy rollout in MuJoCo

### Real hardware

- [ ] Real SO-101 follower driver (stub → production)
