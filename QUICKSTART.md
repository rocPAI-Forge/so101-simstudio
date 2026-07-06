# Quick Start Guide

SO-101 SimStudio 快速上手指南。

## 环境要求

- Python 3.12+
- ROCm GPU（推荐）或 CUDA GPU
- MuJoCo 3.x

## 安装

```bash
# 克隆仓库
git clone --recursive https://github.com/alexhegit/so101-mujoco-teleop.git
cd so101-mujoco-teleop

# 安装依赖（ROCm 版本）
make rocm-sync

# 激活环境
source .venv-rocm/bin/activate
```

## 快速验证

```bash
# 预览 MuJoCo 场景
python -m mujoco.viewer --mjcf=SO101/pick_scene.xml

# 运行 smoke test
make rocm-smoke-record
```

## 使用方法

### 1. Keyboard 遥操作

**实时控制（不录制）**：

```bash
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_keyboard_teleop.yaml \
    --view_mode mujoco
```

**录制数据集**：

```bash
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_keyboard.yaml
```

**按键映射**：

| 按键 | 功能 |
|------|------|
| W/S | Y轴（前后） |
| A/D | X轴（左右） |
| Z/X | Z轴（上下） |
| I/K | 腕部弯曲 |
| [/] | 腕部旋转 |
| C | 夹爪关闭 |
| O | 夹爪打开 |

**录制控制**（录制模式下）：

| 按键 | 功能 |
|------|------|
| Left arrow | 取消当前 episode |
| Right arrow | 保存 episode，下一个 |
| ESC | 停止录制 |

### 2. Joy-Con 遥操作

**安装 Joy-Con 支持**：

```bash
make joycon-sync
```

**实时控制**：

```bash
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_joycon_teleop.yaml \
    --view_mode mujoco
```

**录制数据集**：

```bash
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_joycon.yaml
```

**按键映射（右手）**：

| 按键 | 功能 |
|------|------|
| 摇杆 | XY 移动 |
| R | Z轴上升 |
| 摇杆按下 | Z轴下降 |
| 倾斜 | 腕部旋转 |
| ZR 按住 | 夹爪关闭 |
| Plus | 停止 |

**按键映射（左手）**：

| 按键 | 功能 |
|------|------|
| 摇杆 | XY 移动 |
| L | Z轴上升 |
| 摇杆按下 | Z轴下降 |
| 倾斜 | 腕部旋转 |
| ZL 按住 | 夹爪关闭 |
| Minus | 停止 |

### 3. Leader Arm 遥操作

**实时控制**：

```bash
.venv-rocm/bin/python -m simstudio.scripts.teleoperate \
    --config configs/so101_mujoco_leader_teleop.yaml \
    --view_mode mujoco
```

**录制数据集**：

```bash
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_leader.yaml
```

## 配置参数

### 配置文件

所有配置文件位于 `configs/` 目录：

| 文件 | 用途 |
|------|------|
| `so101_mujoco_keyboard.yaml` | Keyboard 录制配置 |
| `so101_mujoco_keyboard_teleop.yaml` | Keyboard 实时控制配置 |
| `so101_mujoco_joycon.yaml` | Joy-Con 录制配置 |
| `so101_mujoco_joycon_teleop.yaml` | Joy-Con 实时控制配置 |
| `so101_mujoco_leader.yaml` | Leader arm 录制配置 |
| `so101_mujoco_leader_teleop.yaml` | Leader arm 实时控制配置 |

### 常用参数覆盖

```bash
# 修改录制集数
--dataset.num_episodes 5

# 修改保存路径
--dataset.root ./my-datasets

# 续录已有数据集
--resume true

# 修改 Joy-Con 手柄
--teleop.side left
```

### 自定义配置

```bash
# 复制配置模板
cp configs/so101_mujoco_keyboard.yaml configs/my_config.yaml

# 编辑配置
vim configs/my_config.yaml

# 使用自定义配置
.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/my_config.yaml
```

## 数据集管理

### 查看数据集

```bash
# 可视化数据集
.venv-rocm/bin/python -m simstudio.scripts.dataset_viz \
    --repo-id alexhegit/so101_mujoco_keyboard_test \
    --root ./datasets/keyboard-test \
    --episode 0

# 验证数据集
.venv-rocm/bin/python -m simstudio.scripts.validate_dataset \
    --root ./datasets/keyboard-test
```

### 回放数据集

```bash
# 回放单个 episode
.venv-rocm/bin/python -m simstudio.scripts.replay \
    --config configs/so101_mujoco_replay.yaml

# 回放所有 episodes
.venv-rocm/bin/python -m simstudio.scripts.replay_multi \
    --config configs/so101_mujoco_replay_multi.yaml
```

## 常见问题

### Q: MuJoCo 窗口不显示？

设置 `render_window: true`，或使用 `--view_mode mujoco`。

### Q: Joy-Con 连接不上？

1. 运行 `make joycon-sync` 安装驱动
2. 确保 Joy-Con 已配对（蓝牙设置）
3. 检查设备权限：`ls -la /dev/hidraw*`

### Q: 录制速度慢？

- 使用 GPU 加速渲染
- 降低 `camera_width` 和 `camera_height`
- 减少 `camera_names` 数量

### Q: 如何上传数据集到 HuggingFace？

在配置中设置：
```yaml
dataset:
  push_to_hub: true
  repo_id: your-username/your-dataset
```

## 更多信息

- [项目架构](DESIGN.md)
- [项目路线图](ROADMAP.md)
- [AGENTS 开发指南](AGENTS.md)
