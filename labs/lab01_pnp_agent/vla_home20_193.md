# 193 条上的模型阶段结论

日期：2026-10-05

数据是 `datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`，**193 条 / 130835 帧**。磁盘上的 `observation.state` 仍是 15 维；下面这些训练都只取前 6 维关节 `.pos`。评测是 **Lab01-PnP-Home20**：20 个固定方块位姿，手臂 `home`，夹爪起点 0.85，种子 1000 / 1001 / 1002。成绩引用 `final_success`。

这一页是 2026-10-05 停手时的结论。之后 pi05 全量解冻到了纯模型 Home20 **34/60**，路径和配方在 [pi05_unfreeze_home20.md](pi05_unfreeze_home20.md)。下面表格里的 pi05 **12/60** 仍是更早那一版，两个仓库不要混用。ACT 和 SmolVLA 的来历在 [lab01_pnp_agent.md](lab01_pnp_agent.md)。

## Home20

| 模型 | 训练 | 成功 | 停在哪 |
|---|---|---|---|
| ACT 6 维 | 只在前 140 条上，50K，从零训 | **18/20、17/20、18/20 = 53/60** | 对照。权重已在 Hub |
| SmolVLA follow2 | 193 条上再训 2500 步，绝对关节 | **16/20、18/20、17/20 = 51/60** | 对照 |
| MolmoAct2 | 10K，然后从该权重再训 20K | 10K **36/60**；加长后 **13/20、13/20、13/20 = 39/60** | 6 个远侧姿态仍几乎不合拢 |
| GR00T N1.7 绝对关节 | 从 `nvidia/GR00T-N1.7-3B` 训 10K，`use_relative_actions=false` | **7/20、2/20、5/20 = 14/60** | 合拢 45/60，抬起 17/60 |
| pi05 | 从 `ant0nh/pi05_pnp_425_25k` 训 10K，绝对关节，只训动作专家 | **4/20、6/20、2/20 = 12/60** | 合拢 39/60，抬起 15/60 |
| GR00T N1.7 相对动作 | 同一基座，手臂学增量，夹爪保持绝对值 | 10K **0/60**（检查点已删）；30K **1/60** | 10K 能靠近（8 cm 内 54/60），30K 连靠近都退到 11/60 |
| FastWAM | 4K，视频网络冻结 | **0/60** | 4 cm 内合计 3/60，合拢 0 |
| JEPA | 冻结 Qwen，只训动作头，10K | **0/60** | 闭环里不发合拢。这条线停止 |

ACT 的 53/60 用的是 140 条，不是这 193 条。其余行都在这 193 条上训练或微调。

## 可以定下来的话

同一套轨迹够用来学会这个抓放。动作就是数据集里的 6 个关节角度时，ACT 53/60，SmolVLA 51/60。

相对动作不适合这次评测。GR00T 按 LeRobot 官方 SO-101 示例把 `use_relative_actions` 设为 true 之后，10K 步能走到 8 cm，最后几厘米停住；把同一配方拉到 30K，靠近从 54/60 掉到 11/60，成功只有 1/60。

把标签改回绝对关节角度之后，GR00T 10K 升到 14/60。靠近和合拢都有了，失败变成夹空：合拢 45 次里只有 17 次把方块抬起。pi05 是同一类，合拢 39 次，抬起 15 次。

在同一批数据上继续加步数，没有再抬高这几个缺口。MolmoAct2 从 36/60 到 39/60，多出来的是原先已经抬起、只差松开的局；姿态 0、1、5、6、9、17 仍然几乎不在 4 cm 内合拢。绝对关节 GR00T 再用学习率 `1e-5` 续训 5K，种子 1000 从 7/20 降到 4/20（抬起 8 次降到 4 次）。种子 1001、1002 还在测，这个续训不作为新的成绩列。

FastWAM 和 JEPA 的 0/60 不是再加轨迹就能补上的。FastWAM 的原权重是 7 维动作，改成 6 维时相关张量重新初始化，视频网络仍冻着。JEPA 不继续解冻 Qwen。

因此这阶段停止新的微调，也不为了冲成功率去补轨迹。

## 和 NVIDIA SO-101 GR00T 课程不是同一张考卷

课程是 [Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)。仿真评测页写的期望成功率是 **50–70%**：[Sim Evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)。示例日志是跑到第 7、8 集时的累计 66.7% 和 71.4%，不是冻住 20 个点、三个种子的总分。

那个 50–70% 不能拿来对我们的 14/60。两边具体差在：

| | NVIDIA 课程 | 这里的 Home20 |
|---|---|---|
| 模型 | GR00T **N1.6**-3B | GR00T **N1.7**-3B |
| 仿真 | Isaac Lab，离心管放进黄色试管架 | MuJoCo，方块放进盒子 |
| 数据 | 75 条仿真遥操作 | 193 条脚本、DAgger 和 ACT follow |
| 微调 | `global_batch_size=64`，最多 20000 步，学习率 `1e-4` | batch 4，10000 步。样本数大约是课程配方的 1/32 |
| 动作 | SO-100 配置：手臂 **相对** 当前关节，夹爪 **绝对**。见 [so100_config.py](https://github.com/NVIDIA/Isaac-GR00T/blob/n1d6/examples/SO100/so100_config.py) | 相对动作 0/60 和 1/60；绝对关节 14/60 |
| 成功 | 瓶子开始进入架子槽位就可以重置 | 要抬起、移到盒子上方、松开，并且最后仍在盒子里 |

课程自己的配方也是相对手臂动作。他们在 Isaac 里、用大约 32 倍的样本，在另一个任务上做到教程期望的 50–70%。这不说明同样的相对动作配方在我们的 Home20 上应该有同样的分数。我们这边相对动作已经测过，绝对关节更高，但抬起仍然是缺口。

## Hub

| 模型 | 仓库 | 本地权重 |
|---|---|---|
| ACT 6 维 50K | [alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4) | `outputs/train/lab01_pnp_act_state6_agent_yaw_v4/checkpoints/050000` |
| SmolVLA follow2 2.5K | [alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2) | `outputs/train/lab01_pnp_smolvla_agent_yaw_v4_actonly_follow2/checkpoints/002500` |
| MolmoAct2 20K | [alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k) | `outputs/train/lab01_pnp_molmoact2_state6_agent193_plus20k/checkpoints/020000` |
| pi05 10K | [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193) | `outputs/train/lab01_pnp_pi05_state6_agent193/checkpoints/010000` |
| pi05 全量解冻 | [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze) | `/tmp/lab01_pi05_front_unfreezeA2_15k/checkpoints/005000` |
| 193 条数据 | [alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2) | `datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2` |
| GR00T 绝对关节 10K | [alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k) | `outputs/train/lab01_pnp_groot_n17_state6_agent193_abs10k/checkpoints/010000` |

不上传：GR00T 相对动作 30K、低学习率 5K 续训、FastWAM 4K、这次的 JEPA 10K。50 条 teleop 上的旧 GR00T 负结果仍在 [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17)，不要和上面的绝对关节仓库混用。
