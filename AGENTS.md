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

**Known-good LeRobot commit**: `c746ca2d`.

- This project is validated against that upstream LeRobot commit.
- Do not assume the latest LeRobot release tag is compatible.
- Upgrade the submodule only in a dedicated branch, then re-run record/replay smoke tests before merging.

## Commands

| Task | Command |
|------|---------|
| Lint | `make lint` |
| Format | `make format` |
| Test | `make test` |
| MuJoCo preview | `python -m mujoco.viewer --mjcf=SO101/pick_scene.xml` |
| Record (keyboard) | `uv run python -m simstudio.scripts.record --config configs/so101_mujoco_keyboard.yaml` |
| Record (Joy-Con right) | `uv run python -m simstudio.scripts.record --config configs/so101_mujoco_joycon.yaml` |
| Record (Joy-Con left) | `uv run python -m simstudio.scripts.record --config configs/so101_mujoco_joycon_left.yaml` |
| Teleoperate (leader arm) | `uv run python -m simstudio.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml` |
| Record (leader arm) | `uv run python -m simstudio.scripts.record --config configs/so101_mujoco_leader.yaml` |
| Short functional test | `uv run python -m simstudio.scripts.record --config configs/so101_mujoco_keyboard_test.yaml` |
| Replay one episode | `uv run python -m simstudio.scripts.replay --config configs/so101_mujoco_replay.yaml` |
| Replay all episodes | `uv run python -m simstudio.scripts.replay_multi --config configs/so101_mujoco_replay_multi.yaml` |
| Dataset visualization | `uv run python -m simstudio.scripts.dataset_viz --repo-id <repo_id> --root <root> --episode 0` |
| Dataset validation | `uv run python -m simstudio.scripts.validate_dataset --root <dataset_root>` |
| Joy-Con setup | `make joycon-sync` |
| ROCm setup | `make rocm-sync` |

## Key Conventions

- **Python**: 3.12+ required by latest LeRobot.
- **LeRobot integration**: Robots and teleoperators are packaged as third-party plugins (`lerobot_robot_*` / `lerobot_teleoperator_*`) so the submodule stays unmodified. Because they share the project distribution, project wrapper scripts import their config classes explicitly before calling LeRobot entry points.
- **Action semantics**: Two control paradigms supported:
  - **Velocity** (keyboard): `{vx, vy, vz, wrist_flex_rate, yaw_rate, gripper_delta}` — MuJoCo robot uses Jacobian IK to convert to joint positions internally.
  - **Position** (leader arm): `{joint.pos: float}` — direct joint position mapping,1:1 from leader to follower.
  - MuJoCo robot's `send_action` auto-detects format via key names.
- **Window visibility**: The MuJoCo recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, do **not** set `FOCUSED=FALSE` or `FOCUS_ON_SHOW=FALSE`, or the window may be invisible. Set `render_window: false` in the robot config to disable the GLFW window for headless/CI runs.
- **Teleop view_mode**: Use `--view_mode rerun` (default) to show camera feeds in Rerun, `--view_mode mujoco` for MuJoCo window only, or `--view_mode both` for both.
- **Rendering performance**: Camera rendering is done with MuJoCo's offscreen renderer. On CPU-only machines the record loop will be slower than 30 Hz; it still records and saves episodes, but a GPU is strongly recommended for real teleoperation.

## ROCm

- Uses `.venv-rocm` created by `scripts/setup-rocm.sh`.
- Installs torch with `--torch-backend rocm7.2` after project deps to avoid pulling CUDA torch.
- Commands: `make rocm-lint`, `make rocm-test`, `make rocm-format`.

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
