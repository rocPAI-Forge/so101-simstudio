# AGENTS.md

SO-101 MuJoCo teleoperation and behavior-cloning dataset collection, built on HuggingFace LeRobot.

## Project Layout

- **`lerobot/`**: HuggingFace LeRobot submodule (latest upstream). One import line was added to `lerobot/src/lerobot/scripts/lerobot_record.py` to register the project's third-party robot/teleop classes; see "Plugin registration" below.
- **`src/so101_mujoco_teleop/`**: Project-specific robot and teleoperator implementations.
  - `robots/so101_mujoco/`: MuJoCo simulation robot.
  - `robots/so101_real_follower/`: Real SO-101 follower arm (stub).
  - `teleoperators/so101_keyboard/`: Keyboard teleop.
  - `teleoperators/so101_joycon/`: Nintendo Switch Joy-Con teleop (stub).
  - `teleoperators/so101_leader/`: Real SO-101 leader arm used as teleoperator input (stub).
  - `common/`: Shared constants and action mapping utilities.
- **`src/lerobot_robot_so101_mujoco/`** / **`src/lerobot_teleoperator_so101_keyboard/`**: Namespace plugin packages. They re-export the implementations but live inside the same project distribution, so LeRobot's distribution-based discovery does **not** pick them up automatically. Explicit imports in `lerobot_record.py` handle registration.
- **`SO101/`**: MuJoCo scene and robot assets.
- **`configs/`**: Recording configs.
- **`scripts/`**: Environment setup scripts (ROCm, etc.).

## Setup

```bash
git clone --recursive https://github.com/alexhegit/so101-mujoco-teleop.git
cd so101-mujoco-teleop
uv sync
source .venv/bin/activate
```

**Critical**: `lerobot/` is a git submodule. If missing, run:

```bash
git submodule update --init --recursive
```

If downloads are slow, a PyPI mirror can be used via `UV_INDEX_URL`, e.g.:

```bash
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
```

## Commands

| Task | Command |
|------|---------|
| Lint | `make lint` |
| Format | `make format` |
| Test | `make test` |
| MuJoCo preview | `python -m mujoco.viewer --mjcf=SO101/pick_scene.xml` |
| Record (keyboard) | `python lerobot/src/lerobot/scripts/lerobot_record.py --config configs/so101_mujoco_keyboard.yaml` |
| Short functional test | `python lerobot/src/lerobot/scripts/lerobot_record.py --config configs/so101_mujoco_keyboard_test.yaml` |
| ROCm setup | `make rocm-sync` |

## Key Conventions

- **Python**: 3.12+ required by latest LeRobot.
- **LeRobot integration**: Robots and teleoperators are packaged as third-party plugins (`lerobot_robot_*` / `lerobot_teleoperator_*`) so the submodule stays unmodified. Because they share the project distribution, `lerobot/src/lerobot/scripts/lerobot_record.py` imports them explicitly to register their config classes.
- **Action semantics**: All teleoperators output a normalized velocity dict `{vx, vy, vz, wrist_flex_rate, yaw_rate, gripper_delta}`. The MuJoCo robot's `send_action` converts velocities into joint position targets internally; saved dataset actions are the velocity commands.
- **Window visibility**: The MuJoCo recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, do **not** set `FOCUSED=FALSE` or `FOCUS_ON_SHOW=FALSE`, or the window may be invisible. Set `render_window: false` in the robot config to disable the GLFW window for headless/CI runs.
- **Rendering performance**: Camera rendering is done with MuJoCo's offscreen renderer. On CPU-only machines the record loop will be slower than 30 Hz; it still records and saves episodes, but a GPU is strongly recommended for real teleoperation.

## ROCm

- Uses `.venv-rocm` created by `scripts/setup-rocm.sh`.
- Installs torch with `--torch-backend rocm7.2` after project deps to avoid pulling CUDA torch.
- Commands: `make rocm-lint`, `make rocm-test`, `make rocm-format`.
