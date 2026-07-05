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

## v0.0.3 (dev-v0.0.3)

### 遥操作体验
- [x] 画中画摄像头画面（rerun 实时显示 front/top/wrist，操作者看摄像头而非仿真窗口）
- [x] 录制后自动校验（帧率、动作范围、完整性）
- [x] Joy-Con 遥操作（速度范式，支持左手/右手）

### 行为克隆（延后）
- > **注**: 待遥操作方案优化后再进行，确保数据质量

## 项目演进

### 命名路线图

```
当前: so101-mujoco-teleop
        ↓ (v0.1.0 功能稳定后)
阶段1: so101-dataforge
        │  核心: 专家轨迹数据生成
        │  包含: 遥操作、场景、录制、回放、验证、自动化流水线
        ↓ (v1.0+)
阶段2: so101-simstudio
        │  基于 dataforge 构建完整仿真平台
        │  扩展: 场景生成、域随机化、RL训练、策略评估
        └─ dataforge 作为核心子模块/库
```

### 演进原则

- **dataforge** 聚焦「生产高质量专家轨迹数据」，保持工具专一性
- **simstudio** 聚焦「通用仿真平台」，数据生成是其中一个模块
- dataforge 可独立使用，也可被 simstudio 集成

## 远期任务

### 自动化数据生成流水线
- [ ] 数据集标注（lerobot_annotate，VLM 自动生成语言标签）
- [ ] 行为克隆训练（lerobot_train）
- [ ] Policy 推理 + MuJoCo rollout（自动生产轨迹）
- > **流程**: 录制少量种子数据 → 标注 → 训练策略 → 策略自动 rollout → 生成大规模数据集

### 真机硬件
- [ ] Real follower 驱动（stub → 真实 SO-101 从臂实现）
