# Quick Start Guide

Get SO-101 SimStudio running and recording demonstrations.

## Requirements

- Python 3.12+
- ROCm GPU (recommended) or CUDA
- MuJoCo 3.x

## Install

```bash
git clone --recursive https://github.com/alexhegit/so101-simstudio.git
cd so101-simstudio
make rocm-sync
source .venv-rocm/bin/activate
```

If submodules are missing after clone:

```bash
git submodule update --init --recursive
```

**Joy-Con (optional):** Submodule `third_party/joycon-robotics` is pinned to upstream box2ai-robotics; project-specific fixes ship as `patches/joycon-robotics.patch` and are applied **locally at install time** (not committed inside the submodule):

```bash
make joycon-sync
```

This editable-installs joycon-robotics into the venv and runs `git apply patches/joycon-robotics.patch` (serial compatibility + English connect messages). Re-run after resetting the submodule or if Joy-Con connect fails. Pair the controller over Bluetooth first.

## Smoke check

```bash
# Preview MuJoCo scene
python -m mujoco.viewer --mjcf=SO101/scenes/simple_pick/scene.xml

# Pytest + optional ROCm record smoke
make rocm-smoke-record
make smoke-keyboard-record VIEW_MODE=mujoco EPISODES=1
```

**Manual tests:** `make smoke-*` or `scripts/smoke/`. **Fixed collaboration runs:** `scripts/quicktest/*.cmd` (output `test.log`).

## Recording controls (all teleops)

Focus-independent **keyboard** controls (evdev) work in every record session:

| Key | Action |
|-----|--------|
| Right arrow / N | Save episode, go to next |
| Left arrow / R | Cancel episode, re-record |
| ESC / Q | Stop entire session |

Startup log should show: `SO101 recording controls via evdev, focus-independent`.

If evdev is unavailable, add your user to the `input` group and re-login: `sudo usermod -aG input $USER`.

---

## 1. Keyboard teleop

**Live control (no recording):**

```bash
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_keyboard_teleop.yaml
# or: make smoke-keyboard-teleop
```

**Record:**

```bash
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml \
    --view_mode mujoco   # or rerun
```

Both view modes write LeRobot dataset v3.0.

**Movement (hold keys):**

| Key | Action |
|-----|--------|
| W / S | +Y / −Y |
| A / D | −X / +X |
| Z / X | +Z / −Z |
| I / K | Wrist flex up / down |
| [ / ] | Wrist roll left / right |
| O / C | Gripper open / close |

Robot uses **world-frame** velocity (`horizontal_control_mode: world`, default).

---

## 2. Joy-Con teleop

Requires `make joycon-sync` (see **Install** above) so the submodule patch is applied and the package is installed in your venv.

```bash
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_joycon_teleop.yaml \
    --view_mode mujoco
```

**Record (right Joy-Con):**

```bash
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_joycon.yaml \
    --view_mode mujoco
# left: configs/so101_mujoco_joycon_left.yaml
```

**Movement (right Joy-Con)** — cylindrical arm-centric control (`horizontal_control_mode: cylindrical`):

| Input | Action |
|-------|--------|
| Stick forward / back | Reach out / retract (radial) |
| Stick left / right | Base swing (shoulder_pan arc) |
| R | Move up (+Z) |
| Stick press | Move down (−Z) |
| Tilt controller | Wrist flex (roll) / wrist roll (yaw) |
| ZR | Toggle gripper close/open (`gripper_toggle: true`, default) |

**One-handed recording (right Joy-Con)** — in addition to keyboard:

| Button | Action |
|--------|--------|
| A | Save episode, next |
| Y | Cancel episode, re-record |
| + (Plus) | Stop session |

**Left Joy-Con** (no A/Y/+): d-pad **Down** = save & next, **Up** = re-record, **Minus** = stop.

Flip a direction in YAML: `invert_x` (reach), `invert_y` (swing), `invert_z` (up/down). Hold-to-close gripper: `gripper_toggle: false`.

---

## 3. Leader arm teleop

Real SO-101 leader arm (Feetech STS3215) drives the sim follower with 1:1 joint positions.

**USB port:** default `/dev/ttyACM0`; override with `--teleop.port /dev/ttyACM1`.

**Calibration (first run):** Move each joint (including gripper) slowly through its full range when prompted. Cache: `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/None.json`. Press Enter to reuse, or `c` to recalibrate.

**Live control:**

```bash
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_leader_teleop.yaml
# or: make smoke-leader-teleop
```

**Record:**

```bash
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_leader.yaml \
    --view_mode mujoco
```

**Notes:**
- Default `record_fps: 20` (matches `dataset.fps`) for stable real-time sync with serial reads.
- Gripper maps to a safe near-closed value (`home_gripper: -0.1`), not the hard limit.
- Recording uses keyboard only (no Joy-Con-style buttons on leader).

---

## Multi-episode reset (sim)

When `robot.reset_mode: auto` (default), each new episode resets arm and cube independently:

| Field | Values | Meaning |
|-------|--------|---------|
| `reset_arm` | `home` | Teleport arm to fixed home (keyboard default) |
| | `follow` | Keep arm pose; teleop takes over next frame (leader default) |
| `reset_cube` | `fixed` | Predefined pose from `cube_positions.json` |
| | `random` | Sample in graspable bounds (leader default) |
| | `none` | Leave cube unchanged |

**Why leader uses `follow`:** The passive leader stays where you left it. Teleporting the sim to home while your hand is elsewhere causes a large first-frame jump when position mapping resumes.

Random cube bounds (leader config):

```yaml
robot:
  reset_cube: random
  cube_random_x_range: [0.03, 0.11]
  cube_random_y_range: [0.03, 0.10]
  cube_random_z: 0.0125
  cube_random_yaw_range: [0.0, 0.0]
```

---

## Config files

| File | Purpose |
|------|---------|
| `so101_mujoco_keyboard.yaml` | Keyboard record |
| `so101_mujoco_keyboard_teleop.yaml` | Keyboard live teleop |
| `so101_mujoco_joycon.yaml` | Joy-Con record (right) |
| `so101_mujoco_joycon_left.yaml` | Joy-Con record (left) |
| `so101_mujoco_joycon_teleop.yaml` | Joy-Con live teleop |
| `so101_mujoco_leader.yaml` | Leader record |
| `so101_mujoco_leader_teleop.yaml` | Leader live teleop |

**CLI overrides:**

```bash
--dataset.num_episodes 5
--dataset.root ./my-datasets
--resume true
--teleop.side left
```

---

## Dataset tools

```bash
# Visualize
.venv-rocm/bin/python -m simstudio.scripts.dataset_viz \
    --repo-id alexhegit/so101_mujoco_keyboard_test \
    --root ./datasets/keyboard-test --episode 0

# Validate
.venv-rocm/bin/python -m simstudio.scripts.validate_dataset \
    --root ./datasets/keyboard-test

# Replay
.venv-rocm/bin/python -m simstudio.scripts.replay \
    --config configs/so101_mujoco_replay.yaml
```

---

## FAQ

**MuJoCo window not visible?** Set `render_window: true` in robot config; on Ubuntu 24.04 / GNOME do not disable GLFW focus hints.

**Rerun feels laggy?** Expected — Rerun shows pre-action observations. Use `--view_mode mujoco` for low-latency teleop.

**Joy-Con won't connect?** Run `make joycon-sync`, pair via Bluetooth, check `/dev/hidraw*` permissions.

**Record loop slower than target FPS?** Lower `record_fps` and `dataset.fps` together. Leader defaults to 20 Hz for this reason.

**Leader gripper desync?** Recalibrate fully. Ensure FPS is achievable; watch for `running slower than target FPS` warnings.

**Upload to HuggingFace:**

```yaml
dataset:
  push_to_hub: true
  repo_id: your-username/your-dataset
```

## More

- [Architecture](DESIGN.md)
- [Roadmap](ROADMAP.md)
- [AGENTS.md](AGENTS.md)
