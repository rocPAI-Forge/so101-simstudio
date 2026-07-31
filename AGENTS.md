# AGENTS.md

SO-101 simulation studio: expert trajectory generation with MuJoCo and LeRobot.

## Project Layout

- **`lerobot/`**: HuggingFace LeRobot submodule pinned to a known-good upstream commit. The submodule is kept unmodified; all SO-101-specific registration and runtime behavior live in project wrapper scripts under `src/simstudio/scripts/`.
- **`src/simstudio/`**: Project-specific robot and teleoperator implementations.
  - `robots/so101_mujoco/`: MuJoCo simulation robot.
  - `robots/so101_real_follower/`: Real SO-101 follower arm (stub).
  - `teleoperators/so101_keyboard/`: Keyboard teleop.
  - `teleoperators/so101_joycon/`: Nintendo Switch Joy-Con teleop (supports left/right).
  - `teleoperators/so101_leader/`: Real SO-101 leader arm used as teleoperator input.
  - `common/`: Shared constants and action mapping utilities.
- **`src/lerobot_robot_so101_mujoco/`** / **`src/lerobot_teleoperator_so101_keyboard/`**: Namespace plugin packages. They re-export the implementations but live inside the same project distribution, so LeRobot's distribution-based discovery does **not** pick them up automatically. Project wrapper scripts import the config classes explicitly to register them before delegating to LeRobot.
- **`SO101/`**: MuJoCo robot assets and scene definitions.
  - `scenes/<scene_id>/scene.xml`: per-scene MJCF (default: `simple_pick`).
- **`configs/`**: Recording configs; scene layout data under `configs/scenes/<scene_id>/`.
- **`scripts/`**: Environment setup scripts (ROCm, Joy-Con patch install, etc.).
- **`third_party/joycon-robotics/`**: Upstream [box2ai-robotics/joycon-robotics](https://github.com/box2ai-robotics/joycon-robotics) git submodule (pinned to upstream `master`). **Do not commit project-specific edits inside the submodule.** SO-101 fixes (serial compatibility, English connect messages) live in `patches/joycon-robotics.patch` and are applied locally by `make joycon-sync` / `scripts/setup-joycon.sh`.

## Setup

**Supported platform:** Ubuntu 24.04 + AMD GPU + system ROCm 7.x, Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/).

**Not supported yet (unverified; planned):** NVIDIA CUDA on Linux and macOS (including Apple Silicon / MPS). Do **not** use `uv sync` for this project — it installs CUDA PyTorch and is not a supported install path. See [ROADMAP.md](ROADMAP.md).

```bash
git clone --recursive https://github.com/rocPAI-Forge/so101-simstudio.git
cd so101-simstudio
make rocm-sync
source .venv-rocm/bin/activate
```

`make rocm-sync` runs `scripts/setup-rocm.sh`, which:

1. Creates `.venv-rocm` (Python 3.12).
2. Installs **ROCm PyTorch first** (`uv pip install --torch-backend rocm7.2 torch torchvision`).
3. Installs remaining deps under torch version constraints (MuJoCo, LeRobot, SmolVLA training deps, etc.).
4. Editable-installs this project and the `lerobot` submodule.
5. Verifies `torch.version.hip` is present.

First run may take 1–3 hours (ROCm torch wheel is ~6 GB). Re-run `make rocm-sync` to refresh deps; do **not** run bare `uv sync` inside `.venv-rocm`.

**Verify after install:**

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('HIP:', getattr(torch.version, 'hip', None))
print('GPU:', torch.cuda.is_available(), end='')
if torch.cuda.is_available():
    print(' ', torch.cuda.get_device_name(0))
else:
    print()
"
```

Expected: version contains `+rocm`, HIP is not `None`, GPU is `True` on a working ROCm machine.

**Critical**: `lerobot/` and `third_party/joycon-robotics/` are git submodules. If missing, run:

```bash
git submodule update --init --recursive
```

**Joy-Con (optional):** The joycon-robotics submodule stays at the pinned upstream commit; project changes are **not** committed inside it. After `make rocm-sync`, install Joy-Con support and apply the local patch:

```bash
make joycon-sync
```

This runs `uv pip install -e third_party/joycon-robotics` and `git apply patches/joycon-robotics.patch` (serial compatibility + English connect messages). Re-run after a fresh submodule checkout or if Joy-Con connect fails. Pair the controller over Bluetooth before recording.

**Known-good LeRobot commit**: `30da8e68` (`v0.6.0` tag).

- This project is validated against that upstream LeRobot release.
- Do not assume the latest LeRobot main branch is compatible.
- Upgrade the submodule only in a dedicated branch, then re-run record/replay smoke tests before merging.

## Commands

| Task | Command |
|------|---------|
| Lint | `make rocm-lint` |
| Format | `make rocm-format` |
| Test | `make rocm-test` |
| MuJoCo preview | `.venv-rocm/bin/python -m mujoco.viewer --mjcf=SO101/scenes/simple_pick/scene.xml` |
| Smoke: keyboard record | `make smoke-keyboard-record VIEW_MODE=mujoco` |
| Smoke: keyboard replay | `make smoke-keyboard-replay` |
| Smoke: keyboard teleop | `make smoke-keyboard-teleop` |
| Smoke: Joy-Con record | `make smoke-joycon-record SIDE=right VIEW_MODE=mujoco` |
| Smoke: leader record | `make smoke-leader-record VIEW_MODE=mujoco` |
| Record (keyboard) | `.venv-rocm/bin/python -m simstudio.scripts.record --config configs/so101_mujoco_keyboard.yaml` |
| Record (Joy-Con right) | `.venv-rocm/bin/python -m simstudio.scripts.record --config configs/so101_mujoco_joycon.yaml` |
| Record (Joy-Con left) | `.venv-rocm/bin/python -m simstudio.scripts.record --config configs/so101_mujoco_joycon_left.yaml` |
| Quick-test record | `./scripts/quicktest/keyboard.cmd` / `joycon.cmd` / `leader.cmd` |
| Teleoperate (leader arm) | `.venv-rocm/bin/python -m simstudio.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml` |
| Record (leader arm) | `.venv-rocm/bin/python -m simstudio.scripts.record --config configs/so101_mujoco_leader.yaml` |
| Short functional test | `.venv-rocm/bin/python -m simstudio.scripts.record --config configs/so101_mujoco_keyboard_test.yaml` |
| Replay one episode | `.venv-rocm/bin/python -m simstudio.scripts.replay --config configs/so101_mujoco_replay.yaml` |
| Replay all episodes | `.venv-rocm/bin/python -m simstudio.scripts.replay_multi --config configs/so101_mujoco_replay_multi.yaml` |
| Dataset visualization | `.venv-rocm/bin/python -m simstudio.scripts.dataset_viz --repo-id <repo_id> --root <root> --episode 0` |
| Dataset validation | `.venv-rocm/bin/python -m simstudio.scripts.validate_dataset --root <dataset_root>` |
| VLA training (SmolVLA) | `.venv-rocm/bin/lerobot-train ...` (see [QUICKSTART.md](QUICKSTART.md)) |
| Joy-Con setup | `make joycon-sync` |
| ROCm setup | `make rocm-sync` |

## Key Conventions

- **Python**: 3.12+ required by latest LeRobot.
- **LeRobot integration**: Robots and teleoperators are packaged as third-party plugins (`lerobot_robot_*` / `lerobot_teleoperator_*`) so the submodule stays unmodified. Because they share the project distribution, project wrapper scripts import their config classes explicitly before calling LeRobot entry points.
- **Action semantics**: Two control paradigms supported:
  - **Velocity** (keyboard, Joy-Con): `{vx, vy, vz, wrist_flex_rate, yaw_rate, gripper_delta}` — MuJoCo robot uses Jacobian IK internally.
  - **Position** (leader arm): `{joint.pos: float}` — direct joint position mapping, 1:1 from leader to follower.
  - MuJoCo robot's `send_action` auto-detects format via key names.
- **Joy-Con velocity mapping**: Stick is read directly from HID (not joycon-robotics position integration). With `robot.horizontal_control_mode: cylindrical` (Joy-Con configs): stick forward/back → reach in/out, left/right → base swing; radial uses the true `shoulder_pan` world anchor. Gripper: `gripper_toggle: true` (press ZR/ZL to latch) or `false` (hold to close). One-handed recording via `enable_button_recording` + `next_episode_button` / `restart_episode_button` / `stop_button` (right: A/Y/+; left: d-pad Down/Up, Minus).
- **Window visibility**: The MuJoCo recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, do **not** set `FOCUSED=FALSE` or `FOCUS_ON_SHOW=FALSE`, or the window may be invisible. Set `render_window: false` in the robot config to disable the GLFW window for headless/CI runs.
- **Teleop view_mode**: Use `--view_mode rerun` for LeRobot Rerun camera feeds, or `--view_mode mujoco` (default) for the MuJoCo GLFW window. Both modes record the same LeRobot v3.0 dataset. Recording controls are identical for keyboard, Joy-Con and leader in both modes and run through the project's focus-independent evdev backend (no terminal focus needed): Left/R cancel & rerecord, Right/N save & next, ESC/Q stop. The startup log prints `SO101 recording controls via evdev, focus-independent`; if evdev is unavailable (user not in the `input` group) it falls back to LeRobot's terminal listener, which needs the terminal focused.
- **Episode reset (sim)**: `robot.reset_mode: auto` (default) resets before each recorded episode; `manual` leaves state unchanged (real-hardware style). Under `auto`, `robot.reset_arm` (`home` teleport vs `follow` = stay put, for the passive leader arm) and `robot.reset_cube` (`fixed` predefined / `random` within `cube_random_*` bounds / `none`) control arm and cube independently. Keyboard/replay default to `home`+`fixed`; leader configs default to `follow`+`random`. `dataset.reset_time_s` still allows a short teleop window between episodes.
- **Rendering performance**: Camera rendering is done with MuJoCo's offscreen renderer. On CPU-only machines the record loop will be slower than 30 Hz; it still records and saves episodes, but a GPU is strongly recommended for real teleoperation.
- **Smoke scripts**: Interactive manual tests live in `scripts/smoke/` (`make smoke-*`). Fixed collaboration quick-tests live in `scripts/quicktest/` (`*.cmd`, tee `test.log`). Pytest unit tests stay in `tests/`.

## ROCm

- **Only supported Python environment:** `.venv-rocm`, created by `scripts/setup-rocm.sh` (`make rocm-sync`).
- Installs **ROCm PyTorch first** (`--torch-backend rocm7.2`), then pins torch while installing MuJoCo, LeRobot, SmolVLA deps (`transformers`, `accelerate`, …), and dev tools.
- **Do not** run `uv sync` in `.venv-rocm` — it can replace ROCm torch with a CUDA build.
- Commands: `make rocm-sync`, `make rocm-lint`, `make rocm-test`, `make rocm-format`, `make rocm-smoke-record`.
- Details: [llm-wiki/gpu-compute/rocm-setup.md](llm-wiki/gpu-compute/rocm-setup.md).

## Reusable Knowledge

Project-agnostic development knowledge lives in `llm-wiki/`:

| Topic | Path |
|-------|------|
| PyPI mirrors | `llm-wiki/python-dev/mirrors.md` |
| uv constraints | `llm-wiki/python-dev/uv-constraints.md` |
| ROCm setup | `llm-wiki/gpu-compute/rocm-setup.md` |
| PyTorch backends | `llm-wiki/gpu-compute/torch-backends.md` |
| Input listeners | `llm-wiki/platform-quirks/input-listeners.md` |
| GLFW window | `llm-wiki/platform-quirks/glfw-window.md` |
| MuJoCo | `llm-wiki/robot-sim/mujoco-setup.md` |
| LeRobot plugins | `llm-wiki/robot-sim/lerobot-plugin.md` |
| Third-party licenses | `THIRD_PARTY_NOTICES.md`, `NOTICE`, `SO101/ATTRIBUTION.md` |
| Acknowledgements | `ACKNOWLEDGEMENTS.md` |
