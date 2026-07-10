# ROADMAP

## v0.0.1 ✅ (release-v0.0.1)
- [x] MuJoCo SO-101 仿真场景
- [x] 键盘遥操作（Z轴 Z/X 键）
- [x] 录制控制（Left/Right/ESC）
- [x] LeRobot 数据集录制
- [x] 单集/多集 Replay
- [x] ROCm 环境搭建
- [x] Headless smoke test
- [x] 零 submodule 修改
- [x] 插件注册 wrapper 机制

## v0.0.2 ✅ (release-v0.0.2)

### 硬件在环
- [x] Leader arm 遥操作（Feetech STS3215，位置直通）
- [x] Leader arm 校准流程（复用 LeRobot SOLeader）

### 工程
- [x] 双控制范式（velocity / position）
- [x] 自动缩放（归一化值 → MuJoCo 弧度）
- [x] teleoperate 脚本
- [x] llm-wiki 知识库
- [x] ROADMAP 项目规划

## v0.0.3 ✅ (merged into v0.1.0; no separate tag)

> v0.0.3 功能在 `release-v0.1.0` 中一并发布，仓库未打 `release-v0.0.3` 标签。

### 遥操作体验
- [x] 画中画摄像头画面（早期 teleop `view_mode` 实验；录制侧统一方案见 v0.1.1）
- [x] 录制后自动校验（帧率、动作范围、完整性）
- [x] Joy-Con 遥操作（速度范式，支持左手/右手）

## v0.1.0 ✅ (release-v0.1.0)

### 项目重命名
- [x] 重命名为 so101-simstudio（Python包: simstudio）
- [x] 更新所有文档和命令引用

### 功能完整
- [x] 多种遥操作：键盘、Joy-Con（左/右手）、Leader arm
- [x] 数据集录制/回放/验证
- [x] 所有测试通过

### 行为克隆（延后）
- > **注**: 待遥操作方案优化后再进行，确保数据质量

## v0.1.1 ✅ (main; tag pending)

> 自 `release-v0.1.0` 起 18 commits，含 Rerun 录制统一方案与键盘 evdev 修复。建议打 `release-v0.1.1`。

### 录制显示
- [x] `--view_mode mujoco | rerun`（record 脚本；默认 mujoco）
- [x] 两种模式均写入 LeRobot dataset v3.0
- [x] Rerun streaming patch（去除 `static=True`，录制时摄像头实时更新）
- [x] 设计 spec：`docs/superpowers/specs/2026-07-06-unified-rerun-recording-design.md`

### 键盘输入（Rerun / Wayland）
- [x] Rerun 模式 evdev 焦点无关输入（`SO101_PREFER_EVDEV=1`）
- [x] MuJoCo 模式默认 pynput（与 v0.1.0 手感一致）
- [x] 修复 Linux 字母 scancode 映射与 press/release 反转

### 工程
- [x] 手动 smoke 脚本迁至 `scripts/smoke/` + `make smoke-*`
- [x] 修复 `so101_mujoco_joycon_left.yaml` 旧 schema
- [x] 单元测试：`tests/test_record_view_mode.py`、`tests/test_keyboard_teleop.py`

### 场景与录制
- [x] 默认场景迁移至 `SO101/scenes/simple_pick/`
- [x] Multi-episode 自动 reset（`reset_mode: auto`，机械臂 home + cube 位姿）
- [x] 单元测试：`tests/test_episode_reset.py`

## 项目演进

### 命名路线图

```
v0.0.x: so101-mujoco-teleop
        ↓ (v0.1.0)
        ↓ (v0.1.1 — unified record view_mode + evdev keyboard)
当前: so101-simstudio
        │  定位: 通用仿真平台
        │  核心: 专家轨迹数据生成（遥操作、场景、录制、回放、验证）
        │  扩展: 场景生成、域随机化、RL训练、策略评估
        └─ 模块化架构，按需扩展功能
```

### 演进原则

- **so101-simstudio** 聚焦「通用仿真平台」，专家轨迹数据生成是核心能力
- 模块化设计：核心功能优先，扩展功能按需添加
- 保持工具专一性，避免过度工程化

## 远期任务

### 自动化数据生成流水线
- [ ] 数据集标注（lerobot_annotate，VLM 自动生成语言标签）
- [ ] 行为克隆训练（lerobot_train）
- [ ] Policy 推理 + MuJoCo rollout（自动生产轨迹）
- > **流程**: 录制少量种子数据 → 标注 → 训练策略 → 策略自动 rollout → 生成大规模数据集

### 真机硬件
- [ ] Real follower 驱动（stub → 真实 SO-101 从臂实现）
