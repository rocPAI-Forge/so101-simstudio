# SO-101 MuJoCo Teleop

SO-101 robotic arm teleoperation and dataset collection in MuJoCo simulation, built on top of HuggingFace LeRobot.

Supports keyboard teleoperation today, with stubs reserved for Joy-Con, Xbox gamepad, and real SO-101 leader arm teleoperation.

## Quick Start

```bash
git clone --recursive https://github.com/alexhegit/so101-mujoco-teleop.git
cd so101-mujoco-teleop

# CUDA / CPU environment
uv sync
source .venv/bin/activate

# Launch MuJoCo scene preview
python -m mujoco.viewer --mjcf=SO101/pick_scene.xml

# Record demonstrations with keyboard
uv run python -m so101_mujoco_teleop.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml
```

## Project Status

| Component | Status |
|-----------|--------|
| SO-101 MuJoCo robot (`so101_mujoco`) | Working |
| Keyboard teleop (`so101_keyboard`) | Working |
| Joy-Con teleop (`so101_joycon`) | Stub |
| SO-101 leader arm robot (`so101_leader`) | Stub |
| Training with train/eval splits | Planned |

## Notes

- Python 3.12+ is required.
- `lerobot/` is a git submodule; if missing run `git submodule update --init --recursive`.
- The project keeps `lerobot/` unmodified at runtime. SO-101 plugin registration and keyboard-listener integration are handled by project wrapper scripts under `src/so101_mujoco_teleop/scripts/`.
- Known-good upstream LeRobot commit: `c746ca2d`.
- Do not assume the latest LeRobot release tag is compatible with this project; upgrade the submodule only after validating record and replay flows.
- The live recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, `FOCUSED=FALSE` and `FOCUS_ON_SHOW=FALSE` are **not** set, so the window remains visible.
- Rendering performance depends heavily on GPU availability. On CPU-only machines the loop will run slower than 30 Hz and emit a warning, but it will still record frames and save episodes.

## Commands

```bash
# Record
uv run python -m so101_mujoco_teleop.scripts.record --config configs/so101_mujoco_keyboard.yaml

# Replay one episode
uv run python -m so101_mujoco_teleop.scripts.replay --config configs/so101_mujoco_replay.yaml

# Replay all episodes sequentially
uv run python -m so101_mujoco_teleop.scripts.replay_multi --config configs/so101_mujoco_replay_multi.yaml

# Build / repair the ROCm environment
make rocm-sync

# Short headless ROCm smoke test
make rocm-smoke-record
```

## ROCm Notes

- `.venv-rocm` is a local virtualenv created by `scripts/setup-rocm.sh` / `make rocm-sync`; it is not a repo artifact and should not be committed.
- The ROCm setup installs `torch` / `torchvision` with `--torch-backend rocm7.2` before installing the rest of the stack, to avoid accidentally resolving Linux CUDA wheels.
- `make rocm-smoke-record` runs a short headless MuJoCo recording smoke test using `configs/so101_mujoco_keyboard_smoke.yaml`.
- The ROCm setup intentionally pins a compatible PyAV range (`av>=15,<16`) and a stable `placo` / `pin` / `cmeel-*` combination to avoid runtime ABI issues.

## ROCm Environment

```bash
make rocm-sync      # or: ./scripts/setup-rocm.sh
source .venv-rocm/bin/activate
```

## Architecture

See [DESIGN.md](DESIGN.md) for full architecture and migration plan.

## License

Apache-2.0
