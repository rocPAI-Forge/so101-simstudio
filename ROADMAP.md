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

## v0.0.2 (dev-v0.0.2)

### 硬件在环
- [ ] Leader arm 遥操作（Feetech STS3215，位置直通）
- [ ] Leader arm 校准流程
- [ ] Real follower 驱动（stub → 实现）

### 数据质量
- [ ] 录制后自动校验（帧率、动作范围、完整性）
- [ ] 数据集可视化（lerobot_dataset_viz）

### 行为克隆
- [ ] 数据集标注（lerobot_annotate）
- [ ] 行为克隆训练（lerobot_train）
- [ ] Policy 推理 + MuJoCo rollout

### 扩展输入
- [ ] Joy-Con 遥操作（速度范式）

## 控制范式
- **位置直通**: Leader arm → `{joint.pos}` → MuJoCo（1:1 映射）
- **速度命令**: Keyboard / Joy-Con / VR → `{vx, vy, vz, ...}` → MuJoCo IK
- MuJoCo robot 已支持双路径（`send_action` 自动检测）
