# SO-101 MuJoCo Teleop

SO-101 robotic arm teleoperation and dataset collection in MuJoCo simulation, built on top of HuggingFace LeRobot.

Supports keyboard teleoperation and real SO-101 leader arm teleoperation (Feetech STS3215).

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
| Leader arm teleop (`so101_leader`) | Working (Feetech STS3215) |
| Dataset recording and replay | Working |
| Dataset visualization | Working (via LeRobot Rerun) |
| Joy-Con teleop (`so101_joycon`) | Working |
| Real follower arm (`so101_real_follower`) | Stub |
| Behavior cloning training | Planned |

## Notes

- Python 3.12+ is required.
- `lerobot/` is a git submodule; if missing run `git submodule update --init --recursive`.
- The project keeps `lerobot/` unmodified at runtime. SO-101 plugin registration and keyboard-listener integration are handled by project wrapper scripts under `src/so101_mujoco_teleop/scripts/`.
- Known-good upstream LeRobot commit: `c746ca2d`.
- Do not assume the latest LeRobot release tag is compatible with this project; upgrade the submodule only after validating record and replay flows.
- The live recording window uses GLFW default hints. On Ubuntu 24.04 / GNOME, `FOCUSED=FALSE` and `FOCUS_ON_SHOW=FALSE` are **not** set, so the window remains visible.
- Rendering performance depends heavily on GPU availability. On CPU-only machines the loop will run slower than 30 Hz and emit a warning, but it will still record frames and save episodes.

## Camera Feed (Teleop)

Teleop 时可在 Rerun 中实时显示摄像头画面，操作者可从摄像头视角而非第三人称仿真视角进行操控。

```bash
# 只显示 Rerun 摄像头画面（默认）
uv run python -m so101_mujoco_teleop.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml --view_mode rerun

# 只显示 MuJoCo 窗口
uv run python -m so101_mujoco_teleop.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml --view_mode mujoco

# 两个都显示
uv run python -m so101_mujoco_teleop.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml --view_mode both
```

## Joy-Con Teleop

支持 Nintendo Switch Joy-Con 单手柄操控 SO-101 机械臂。使用 [joycon-robotics](https://github.com/box2ai-robotics/joycon-robotics) 库实现位置-速度转换。

### 安装

```bash
# 安装 joycon-robotics（submodule + 补丁）
make joycon-sync
```

### 按键映射

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Joy-Con 按键映射                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐          ┌─────────────────────┐          │
│  │     左手 Joy-Con    │          │     右手 Joy-Con    │          │
│  ├─────────────────────┤          ├─────────────────────┤          │
│  │                     │          │                     │          │
│  │    [L]              │          │              [R]    │          │
│  │   Z轴上升           │          │   Z轴上升           │          │
│  │                     │          │                     │          │
│  │  ┌───┐              │          │              ┌───┐  │          │
│  │  │ ↑ │ 重录         │          │         [Y] │   │  │          │
│  │  ├───┤              │          │              ├───┤  │          │
│  │←─┤   ├─→ 下一ep    │          │  [A] [B]     │   │  │          │
│  │  ├───┤              │          │              ├───┤  │          │
│  │  │ ↓ │              │          │         [X] │   │  │          │
│  │  └───┘              │          │              └───┘  │          │
│  │                     │          │                     │          │
│  │  [ZL] 夹爪关闭      │          │      [ZR] 夹爪关闭  │          │
│  │  松开=夹爪打开      │          │      松开=夹爪打开  │          │
│  │                     │          │                     │          │
│  │  [-]  停止录制      │          │      [+]  停止录制  │          │
│  │                     │          │                     │          │
│  │  摇杆: XY平移       │          │      摇杆: XY平移   │          │
│  │  倾斜: 腕部旋转     │          │      倾斜: 腕部旋转 │          │
│  └─────────────────────┘          └─────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 控制模式

| 功能 | 描述 |
|------|------|
| X轴移动 | 摇杆左右 |
| Y轴移动 | 摇杆前后 |
| Z轴上升 | R键（右手）/ L键（左手）|
| Z轴下降 | 摇杆按下 |
| 腕部旋转 | 倾斜手柄（陀螺仪）|
| 夹爪关闭 | ZR按住（右手）/ ZL按住（左手）|
| 夹爪打开 | 松开 ZR/ZL |
| 下一episode | A键（右手）/ 左方向键（左手）|
| 重录当前episode | Y键（右手）/ 上方向键（左手）|
| 停止录制 | Plus（右手）/ Minus（左手）|

### 使用方法

```bash
# 右手 Joy-Con 录制
uv run python -m so101_mujoco_teleop.scripts.record \
    --config configs/so101_mujoco_joycon.yaml

# 左手 Joy-Con 录制
uv run python -m so101_mujoco_teleop.scripts.record \
    --config configs/so101_mujoco_joycon_left.yaml

# Joy-Con 实时控制（不录制）
uv run python -m so101_mujoco_teleop.scripts.teleoperate \
    --config configs/so101_mujoco_joycon_teleop.yaml
```

## Commands

```bash
# Record with keyboard
uv run python -m so101_mujoco_teleop.scripts.record --config configs/so101_mujoco_keyboard.yaml

# Record with Joy-Con (right hand)
uv run python -m so101_mujoco_teleop.scripts.record --config configs/so101_mujoco_joycon.yaml

# Record with Joy-Con (left hand)
uv run python -m so101_mujoco_teleop.scripts.record --config configs/so101_mujoco_joycon_left.yaml

# Teleoperate with leader arm (live control, no recording)
uv run python -m so101_mujoco_teleop.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml

# Record with leader arm
uv run python -m so101_mujoco_teleop.scripts.record --config configs/so101_mujoco_leader.yaml

# Replay one episode
uv run python -m so101_mujoco_teleop.scripts.replay --config configs/so101_mujoco_replay.yaml

# Replay all episodes sequentially
uv run python -m so101_mujoco_teleop.scripts.replay_multi --config configs/so101_mujoco_replay_multi.yaml

# Visualize dataset (opens Rerun viewer)
uv run python -m so101_mujoco_teleop.scripts.dataset_viz \
    --repo-id <repo_id> --root <root> --episode 0

# Validate dataset quality
uv run python -m so101_mujoco_teleop.scripts.validate_dataset \
    --root ./datasets/leader-test

# Setup Joy-Con environment
make joycon-sync

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
