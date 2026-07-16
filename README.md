# SO-101 SimStudio

SO-101 simulation studio for expert trajectory generation with MuJoCo and LeRobot.

## Overview

SO-101 SimStudio is a MuJoCo-based teleoperation platform for collecting high-quality expert demonstration datasets for behavior cloning.

**Core capabilities:**
- Multiple teleop backends (keyboard, Joy-Con, leader arm)
- High-fidelity MuJoCo physics simulation
- LeRobot v3.0 dataset format
- Modular, extensible architecture

## Features

### Teleoperation

| Method | Status | Notes |
|--------|--------|-------|
| Keyboard | ✅ | World-frame velocity (WASD + keys) |
| Joy-Con | ✅ | Cylindrical reach/swing, left or right |
| Leader arm | ✅ | Feetech STS3215, 1:1 position mapping |

### Dataset pipeline

| Feature | Status | Notes |
|---------|--------|-------|
| Record | ✅ | Multi-episode, resume |
| Replay | ✅ | Single or all episodes |
| Validate | ✅ | Quality checks |
| Visualize | ✅ | Rerun preview + `dataset_viz` |

### Simulation

- **Robot**: SO-101 6-DOF arm
- **Scene**: Table pick task (`simple_pick`)
- **Cameras**: front / top / wrist
- **Physics**: MuJoCo

## Requirements

### Tested environment

- **OS**: Ubuntu 24.04
- **GPU**: AMD ROCm 7.2.x (recommended)
- **Python**: 3.12+
- **RAM**: 8 GB+
- **Storage**: 10 GB+

### Dependencies

- MuJoCo 3.x
- PyTorch 2.x (ROCm backend on AMD)
- LeRobot (git submodule, pinned to v0.6.0)

### Known limits

- Primary validation on Ubuntu 24.04 + ROCm 7.2.x
- macOS / NVIDIA CUDA support not yet documented

## Layout

```
so101-simstudio/
├── src/simstudio/           # Core package
│   ├── robots/              # Robot implementations
│   ├── teleoperators/       # Keyboard, Joy-Con, leader
│   ├── scripts/             # record, replay, teleoperate, …
│   └── common/              # Shared utilities
├── configs/                 # YAML configs
├── SO101/                   # MuJoCo assets
├── scripts/smoke/           # Parameterized manual smoke tests
├── scripts/quicktest/       # Fixed 2-episode collaboration runs
├── lerobot/                 # LeRobot submodule
└── third_party/             # joycon-robotics, etc.
```

## Quick start

See [QUICKSTART.md](QUICKSTART.md).

```bash
git clone --recursive https://github.com/alexhegit/so101-simstudio.git
cd so101-simstudio
make rocm-sync
# Joy-Con users: also run `make joycon-sync` (see QUICKSTART Install)

.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml \
    --view_mode mujoco   # or rerun
```

**Manual tests:**

```bash
make smoke-keyboard-record VIEW_MODE=mujoco EPISODES=1
make smoke-joycon-record SIDE=right VIEW_MODE=mujoco
./scripts/quicktest/keyboard.cmd    # fixed 2-episode run → test.log
```

## Status

| Component | Status |
|-----------|--------|
| MuJoCo sim robot | ✅ |
| Keyboard teleop | ✅ |
| Joy-Con teleop (cylindrical + one-handed recording) | ✅ |
| Leader arm teleop | ✅ |
| Record / replay / validate | ✅ |
| Rerun record preview | ✅ |
| Real follower hardware | 🔲 Planned |
| Behavior cloning training | 🔲 Planned |

## Version history

- **v0.1.2** (`release-v0.1.2`): Joy-Con cylindrical reach/swing; decoupled HID sticks; gripper toggle; one-handed A/Y/+ recording; `scripts/quicktest/`; joycon-robotics upstream pin + install-time patch; English docs/prints; repo `so101-simstudio`
- **v0.1.1** (`release-v0.1.1`): Unified `--view_mode mujoco/rerun`; evdev recording controls; leader reliability; `reset_arm` / `reset_cube`; smoke scripts in `scripts/smoke/`
- **v0.1.0** (`release-v0.1.0`): First stable release; keyboard, Joy-Con, leader
- **v0.0.2** (`release-v0.0.2`): Leader arm support
- **v0.0.1** (`release-v0.0.1`): Initial release

## Documentation

- [Quick start](QUICKSTART.md)
- [Architecture](DESIGN.md)
- [Roadmap](ROADMAP.md)
- [Agent / dev guide](AGENTS.md)

## License

Apache-2.0
