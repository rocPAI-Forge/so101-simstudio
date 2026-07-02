# AGENTS.md

SO-101 MuJoCo teleoperation and behavior-cloning dataset collection, built on HuggingFace LeRobot.

## Project Layout

- **`lerobot/`**: HuggingFace LeRobot submodule (latest upstream).
- **`src/so101_mujoco_teleop/`**: Project-specific robot and teleoperator implementations, registered as LeRobot third-party plugins.
  - `robots/so101_mujoco/`: MuJoCo simulation robot.
  - `robots/so101_leader/`: Real SO-101 leader arm (planned).
  - `teleoperators/so101_keyboard/`: Keyboard teleop.
  - `teleoperators/so101_joycon/`: Nintendo Switch Joy-Con teleop (planned).
  - `common/`: Shared constants and action mapping utilities.
- **`SO101/`**: MuJoCo scene and robot assets.
- **`configs/`**: Recording and training configs.
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

## Commands

| Task | Command |
|------|---------|
| Lint | `make lint` |
| Format | `make format` |
| Test | `make test` |
| MuJoCo preview | `python -m mujoco.viewer --mjcf=SO101/pick_scene.xml` |
| Record (keyboard) | `python lerobot/src/lerobot/scripts/lerobot_record.py --config configs/so101_mujoco_keyboard.yaml` |
| ROCm setup | `make rocm-sync` |

## Key Conventions

- **Python**: 3.12+ required by latest LeRobot.
- **LeRobot integration**: Robots and teleoperators are packaged as third-party plugins (`lerobot_robot_*` / `lerobot_teleoperator_*`) so the submodule stays unmodified.
- **Action semantics**: All teleoperators output a normalized velocity dict `{vx, vy, vz, wrist_flex_rate, yaw_rate, gripper_delta}`. Robots or the action-mapping layer convert this to their native action format.
- **Window visibility**: The MuJoCo recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, do **not** set `FOCUSED=FALSE` or `FOCUS_ON_SHOW=FALSE`, or the window may be invisible.

## ROCm

- Uses `.venv-rocm` created by `scripts/setup-rocm.sh`.
- Installs torch with `--torch-backend rocm7.2` after project deps to avoid pulling CUDA torch.
- Commands: `make rocm-lint`, `make rocm-test`, `make rocm-format`.
