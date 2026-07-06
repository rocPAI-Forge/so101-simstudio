# SO-101 SimStudio

SO-101 simulation studio: expert trajectory generation with MuJoCo and LeRobot.

## 项目定位

SO-101 SimStudio 是一个基于 MuJoCo 仿真环境的机器人遥操作平台，用于生成高质量的专家轨迹数据集，支持行为克隆训练。

**核心能力**：
- 多种遥操作方式（Keyboard、Joy-Con、Leader Arm）
- 高保真 MuJoCo 物理仿真
- 标准化数据集格式（LeRobot）
- 可扩展的模块化架构

## 功能特性

### 遥操作支持

| 方式 | 状态 | 说明 |
|------|------|------|
| Keyboard | ✅ | WASD + 方向键控制 |
| Joy-Con | ✅ | 支持左手/右手 |
| Leader Arm | ✅ | Feetech STS3215 实物臂 |

### 数据集功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 录制 | ✅ | 支持多 episode、续录 |
| 回放 | ✅ | 单集/多集回放 |
| 验证 | ✅ | 数据质量检查 |
| 可视化 | ✅ | Rerun 3D 可视化 |

### 仿真环境

- **机器人**: SO-101 6自由度机械臂
- **场景**: 桌面抓取任务
- **相机**: 前视/俯视/腕部 三视角
- **物理**: MuJoCo 高保真仿真

## 系统要求

### 最低配置

- **OS**: Ubuntu 20.04+ / macOS 12+
- **Python**: 3.12+
- **GPU**: 支持 OpenGL 3.3+ 的显卡
- **RAM**: 8GB+

### 推荐配置

- **GPU**: AMD ROCm 兼容显卡（如 RX 6800 XT）
- **RAM**: 16GB+
- **存储**: 10GB+ 可用空间

### 软件依赖

- MuJoCo 3.x
- PyTorch 2.x（ROCm 或 CUDA）
- LeRobot（作为 git submodule）

## 架构概览

```
so101-simstudio/
├── src/simstudio/           # 核心代码
│   ├── robots/              # 机器人实现
│   │   ├── so101_mujoco/    # MuJoCo 仿真机器人
│   │   └── so101_real_follower/  # 真实机器人（预留）
│   ├── teleoperators/       # 遥操作实现
│   │   ├── so101_keyboard/  # 键盘控制
│   │   ├── so101_joycon/    # Joy-Con 控制
│   │   └── so101_leader/    # Leader arm 控制
│   ├── scripts/             # 入口脚本
│   └── common/              # 共享工具
├── configs/                 # 配置文件
├── SO101/                   # MuJoCo 场景资产
├── lerobot/                 # LeRobot submodule
└── third_party/             # 第三方依赖
```

## 快速开始

详见 [QUICKSTART.md](QUICKSTART.md)

```bash
# 安装
git clone --recursive https://github.com/alexhegit/so101-mujoco-teleop.git
cd so101-mujoco-teleop
make rocm-sync

# 录制数据集
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml
```

## 项目状态

| 组件 | 状态 |
|------|------|
| MuJoCo 仿真机器人 | ✅ Working |
| Keyboard 遥操作 | ✅ Working |
| Joy-Con 遥操作 | ✅ Working |
| Leader Arm 遥操作 | ✅ Working |
| 数据集录制/回放 | ✅ Working |
| 数据集验证 | ✅ Working |
| 真实机器人 | 🔲 Planned |
| 行为克隆训练 | 🔲 Planned |

## 版本历史

- **v0.1.0** (2026-07-06): 首个正式版本，支持多种遥操作
- **v0.0.3**: 添加 Joy-Con 支持
- **v0.0.2**: 添加 Leader Arm 支持
- **v0.0.1**: 初始版本

## 相关文档

- [快速开始](QUICKSTART.md)
- [项目架构](DESIGN.md)
- [项目路线图](ROADMAP.md)
- [开发指南](AGENTS.md)

## 许可证

Apache-2.0
