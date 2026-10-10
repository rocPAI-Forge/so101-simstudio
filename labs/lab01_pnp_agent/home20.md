# Lab01-PnP-Home20

日期：2026-10-10

这是评测协议，不是训练集。20 个方块位姿冻在 [`configs/smolvla_v4_div50_home20_spawns.json`](configs/smolvla_v4_div50_home20_spawns.json)。193 条训练数据仍在仓库根 `datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`。权重仍在 `outputs/` 或当时的训练目录。

旧路径 `labs/lab01_pnp/configs/smolvla_v4_div50_home20_spawns.json` 是指向这份清单的符号链接。

## 协议

这 20 个点是 2026-10-02 v4-div SmolVLA 4K、`home`、n=20 那次评测里，在完整记录盒中均匀抽到的一次（`outputs/eval/smolvla_v4_div50_4k_egl_home_n20.log`）。不是按难度挑的。`z` 是桌面高度。每一行还写着手臂起点 `qpos` `[0.0, -0.3, 0.6, 1.2, 0.0, 0.85]`（home 关节，夹爪张开）。不要改方块位姿，也不要改这个 `qpos`。

跑的时候：`LAB01_EPISODE_INIT` 指向这份清单，`num_episodes=20`，`reset_arm: home`，`n_action_steps=50`，每集 60 秒。成功是回合结束时方块在容器里，日志字段 `final_success`。复位日志里会先出现一次随机方块和 YAML 里的 home，然后评测用这份文件覆盖两者；引用时带上 `Episode init` 那一行，包括 `qpos`。归一化统计用该检查点训练时的数据集。同一 20 个点上夹爪起点 **−0.1** 的旧分数暂时留着，不写入 0.85 这一列。

正式报分用 `LAB01_EVAL_SEED` **1000、1001、1002** 各跑完这 20 集。种子只改变策略采样动作块时的噪声序列，不增加方块点，也不是 Home60。

跨模型比较时用 `reset_arm: home`、夹爪 **0.85**、这份方块清单。每个策略保留自己的相机映射、状态维数和推理后端，不同就写明。ACT 如果把手臂留在 `follow`，就不是这项考试。另一组 20 个点的分数不记在这里。

全范围新抽（不设 `LAB01_EPISODE_INIT`）是另一列。单个演示位姿 `(0.27, 0.20, yaw −8°)` 也不是这一列。

`LAB01_HOME20_SPAWNS` 在本目录 `_env.sh` 里，默认就是这份清单。训练和评测入口仍是 `../lab01_pnp/train_*.cmd` 和 `eval.cmd`。

## 0.85 成绩（2026-10-03 当时的三行）

EGL，A30，`n_action_steps=50`。

| 策略 / 检查点 | 训练数据 | 推理 | 成功 |
|---|---|---|---:|
| ACT 6 维 v4 50K | agent-yaw v4（140 条） | sync | **53/60**（18/20、17/20、18/20） |
| SmolVLA 6 维 actonly-follow2 2.5K | 171 条再加 22 条 ACT follow（193；最后 22 条 ×3） | RTC | **51/60**（16/20、18/20、17/20） |
| SmolVLA 6 维 actonly 3K | agent-yaw v4 加 31 条 ACT follow（171 条） | RTC | **45/60** |

ACT 三次都是接近 20/20、闭合 20/20，漏在抬起之后。日志 `outputs/eval/act_state6_v4_lab01_pnp_home20_g085_seed1000.log` 及 `seed1001`、`seed1002`。更早未设种子的 **17/20**（`..._g085.log`）不是这一列。

SmolVLA follow2 是采用的 0.85 检查点，高于 actonly 3K 的 **45/60**。从 `actonly3k/checkpoints/003000` 用新优化器、学习率 `1e-5` 再训 2500 步。统计根目录是 `datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`。日志 `outputs/eval/smolvla_v4_actonly_follow2_lab01_pnp_home20_g085_seed1000.log` 及另外两个种子。权重 `outputs/train/lab01_pnp_smolvla_agent_yaw_v4_actonly_follow2/checkpoints/002500/pretrained_model`。

actonly 3K 三次是 **16/20、15/20、14/20**。未设种子的 **12/20** 不作为这一列。

为何加上那 31 条 ACT follow：同一 20 个点、夹爪起点 **−0.1**、学习率 `1e-5`、3000 步。只有原始 v4 时是 **0/20**，而且一次都不闭合。加上这 31 条（那 31 条帧 ×3）后是 **17/20**。关节踢开、搬运踢开不是这次跳跃所必需的。完整账在 [`smolvla_v4_diversity.md`](smolvla_v4_diversity.md)。**0/20** 和 **17/20** 是夹爪 −0.1 的选数据证据，不是 0.85 的 benchmark。同一闭合起点上 ACT 6 维是 **18/20**，同样只是探针。

## 后来加进同一协议的模型

MolmoAct2、pi05、GR00T、FastWAM、JEPA 的 Home20 行在 [`vla_home20_193.md`](vla_home20_193.md)。pi05 全量解冻到纯模型 **34/60** 的路径在 [`pi05_unfreeze_home20.md`](pi05_unfreeze_home20.md)。

## 发布

脚本在本目录。`labs/lab01_pnp/` 下的同名文件会转到这里。

| 脚本 | 内容 |
|---|---|
| `push_act_agent_yaw_v4_model.cmd` | ACT agent-yaw v4 |
| `push_smolvla_follow2_model.cmd` | SmolVLA follow2，Home20 **51/60** |
| `push_molmoact2_agent193_model.cmd` | MolmoAct2 加长 20K，Home20 **39/60** |
| `push_pi05_agent193_model.cmd` | pi05 只训动作专家 10K，Home20 **12/60** |
| `push_pi05_agent193_unfreeze_model.cmd` | pi05 全量解冻，Home20 **34/60** |
| `push_groot_agent193_abs_model.cmd` | GR00T 绝对关节 10K，Home20 **14/60** |
| `push_agent_yaw_follow2_dataset.cmd` | 193 条数据集 |

50 条 teleop 数据集和对应的 ACT / SmolVLA / MolmoAct2 / GR00T / JEPA 上传脚本仍在 `labs/lab01_pnp/`。
