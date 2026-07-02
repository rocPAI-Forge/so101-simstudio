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
python lerobot/src/lerobot/scripts/lerobot_record.py \
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

- Python 3.12+ is required by the latest LeRobot submodule.
- `lerobot/` is a git submodule; if missing run `git submodule update --init --recursive`.
- The SO-101 MuJoCo robot and keyboard teleop are registered with LeRobot through explicit imports in `lerobot/src/lerobot/scripts/lerobot_record.py` (the distribution-based third-party plugin discovery only sees top-level distribution names, so our namespace subpackages are imported explicitly there).
- The live recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, `FOCUSED=FALSE` and `FOCUS_ON_SHOW=FALSE` are **not** set, so the window remains visible.
- Rendering performance depends heavily on GPU availability. On CPU-only machines the loop will run slower than 30 Hz and emit a warning, but it will still record frames and save episodes.

## ROCm Environment

```bash
make rocm-sync      # or: ./scripts/setup-rocm.sh
source .venv-rocm/bin/activate
```

## Architecture

See [DESIGN.md](DESIGN.md) for full architecture and migration plan.

## License

Apache-2.0
