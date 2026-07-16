# SO-101 SimStudio

SO-101 simulation studio for expert trajectory generation with MuJoCo and LeRobot.

## Overview

SO-101 SimStudio is a MuJoCo-based teleoperation platform for collecting high-quality expert demonstration datasets for behavior cloning.

**Core capabilities:**
- Multiple teleop backends (keyboard, Joy-Con, leader arm)
- Two recording GUIs: **MuJoCo** (GLFW 3D window) and **Rerun** (multi-camera stream)
- High-fidelity MuJoCo physics simulation
- LeRobot v3.0 dataset format
- Modular, extensible architecture

**Platform focus:** Validated on **Ubuntu 24.04 + AMD ROCm**. macOS and NVIDIA CUDA are **not supported yet** (see [ROADMAP](ROADMAP.md)).

## Features

### Recording GUI (`--view_mode`)

Both modes record the same LeRobot v3.0 dataset; only the live preview differs.

| GUI | Config flag | Status | Notes |
|-----|-------------|--------|-------|
| **MuJoCo** | `--view_mode mujoco` (default) | ✅ | Native GLFW 3D simulation window; interactive camera |
| **Rerun** | `--view_mode rerun` | ✅ | LeRobot Rerun viewer; multi-camera stream; useful on Wayland / headless-friendly workflows |

Post-recording visualization: `dataset_viz` (Rerun) and replay scripts.

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
| Visualize | ✅ | Rerun (`dataset_viz`) + replay |

### Simulation

- **Robot**: SO-101 6-DOF arm
- **Scene**: Table pick task (`simple_pick`)
- **Cameras**: front / top / wrist
- **Physics**: MuJoCo

## Requirements

### Supported platform (current release)

| Environment | Status |
|-------------|--------|
| **Ubuntu 24.04 + AMD ROCm 7.2.x** | ✅ **Primary** — tested and documented |
| **NVIDIA CUDA (Linux)** | 🔲 **Not supported** — planned |
| **macOS (incl. Apple Silicon / MPS)** | 🔲 **Not supported** — planned |

SimStudio is developed as a **ROCm-first** simulation toolkit for expert trajectory
collection on AMD GPUs. Other platforms may install dependencies manually, but they
are outside the supported matrix until listed on the roadmap as done.

CPU-only runs are possible (e.g. CI smoke tests with `render_window: false`) but
camera rendering is slow without a GPU.

### Tested environment

- **OS**: Ubuntu 24.04
- **GPU**: AMD ROCm 7.2.x
- **Python**: 3.12+
- **RAM**: 8 GB+
- **Storage**: 10 GB+

### Dependencies

- MuJoCo 3.x
- PyTorch 2.x (**ROCm** backend on AMD — installed via `make rocm-sync`)
- LeRobot (git submodule, pinned to v0.6.0)

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
| MuJoCo recording GUI (`view_mode=mujoco`) | ✅ |
| Rerun recording GUI (`view_mode=rerun`) | ✅ |
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
- [Acknowledgements](ACKNOWLEDGEMENTS.md) · [Third-party licenses](THIRD_PARTY_NOTICES.md) · [Simulation asset attribution](SO101/ATTRIBUTION.md)

## Acknowledgements

This project stands on open-source work from [LeRobot](https://github.com/huggingface/lerobot),
[MuJoCo](https://github.com/google-deepmind/mujoco), [The Robot Studio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
[box2ai joycon-robotics](https://github.com/box2ai-robotics/joycon-robotics), [Rerun](https://github.com/rerun-io/rerun),
and many other libraries. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for the full list.

## License

**SO-101 SimStudio** (original code in this repository) is licensed under
[Apache-2.0](LICENSE).

This project depends on and bundles third-party software and robot assets under
their own licenses (LeRobot, MuJoCo, joycon-robotics, SO-ARM100 lineage, Python
dependencies, etc.). Those components are **not** relicensed under Apache-2.0.

| Document | Purpose |
|----------|---------|
| [LICENSE](LICENSE) | Apache-2.0 terms for project-authored code |
| [NOTICE](NOTICE) | Apache `NOTICE` file for distributions |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Submodules, dependencies, compatibility notes |
| [SO101/ATTRIBUTION.md](SO101/ATTRIBUTION.md) | MuJoCo / URDF / mesh provenance |

When redistributing releases, include `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md` (and submodule license files if you ship submodules).
