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
- [ ] Real follower 驱动（stub → 实现）

### 数据质量
- [ ] 录制后自动校验（帧率、动作范围、完整性）
- [ ] 数据集可视化（lerobot_dataset_viz）

### 行为克隆
- [ ] 数据集标注（lerobot_annotate）
- [ ] 行为克隆训练（lerobot_train）
- [ ] Policy 推理 + MuJoCo rollout

### 工程
- [x] 双控制范式（velocity / position）
- [x] 自动缩放（归一化值 → MuJoCo 弧度）
- [x] teleoperate 脚本
- [x] llm-wiki 知识库
- [x] ROADMAP 项目规划

### 扩展输入
- [ ] Joy-Con 遥操作（速度范式）

## v0.0.3 (dev-v0.0.3)

### 遥操作体验
- [x] 画中画摄像头画面（rerun 实时显示 front/top/wrist，操作者看摄像头而非仿真窗口）
- [x] 录制后自动校验（帧率、动作范围、完整性）
- [ ] Joy-Con 遥操作（速度范式）

### 行为克隆（延后）
- [ ] 行为克隆训练（lerobot_train）
- [ ] Policy 推理 + MuJoCo rollout
- > **注**: 待遥操作方案优化后再进行，确保数据质量
