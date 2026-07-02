# SO-101 MuJoCo Teleop

SO-101 robotic arm teleoperation and dataset collection in MuJoCo simulation, built on top of HuggingFace LeRobot.

Supports keyboard teleoperation today, with architecture预留 (reserved) for Joy-Con, Xbox gamepad, and real SO-101 leader arm teleoperation.

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

## ROCm Environment

```bash
make rocm-sync      # or: ./scripts/setup-rocm.sh
source .venv-rocm/bin/activate
```

## Project Status

| Component | Status |
|-----------|--------|
| SO-101 MuJoCo robot (`so101_mujoco`) | In development |
| Keyboard teleop (`so101_keyboard`) | In development |
| Joy-Con teleop (`so101_joycon`) | Planned |
| SO-101 leader arm teleop (`so101_leader`) | Planned |
| Training with train/eval splits | Planned |

## Architecture

See [DESIGN.md](DESIGN.md) for full architecture and migration plan.

## License

Apache-2.0
