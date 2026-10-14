# Lab01-PnP-Home20

日期：2026-10-14

这是评测协议，也是 193 条这条线上的成绩页。20 个方块位姿冻在 [`configs/smolvla_v4_div50_home20_spawns.json`](configs/smolvla_v4_div50_home20_spawns.json)。训练数据是仓库根 `datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2`，**193 条 / 130835 帧**。旧路径 `labs/lab01_pnp/configs/smolvla_v4_div50_home20_spawns.json` 是指向这份清单的符号链接。

磁盘上的 `observation.state` 仍是 15 维。下面这些训练都只取前 6 维关节 `.pos`。成绩引用 `final_success`。ACT 和 SmolVLA 的数据来历在 [`lab01_pnp_agent.md`](lab01_pnp_agent.md)。SmolVLA 的夹爪 **−0.1** 旧分数在 [`smolvla_v4_diversity.md`](smolvla_v4_diversity.md)，不写入下面的 0.85 列。

## 协议

这 20 个点是 2026-10-02 v4-div SmolVLA 4K、`home`、n=20 那次评测里，在完整记录盒中均匀抽到的一次（`outputs/eval/smolvla_v4_div50_4k_egl_home_n20.log`）。不是按难度挑的。`z` 是桌面高度。每一行还写着手臂起点 `qpos` `[0.0, -0.3, 0.6, 1.2, 0.0, 0.85]`（home 关节，夹爪张开）。不要改方块位姿，也不要改这个 `qpos`。

跑的时候：`LAB01_EPISODE_INIT` 指向这份清单，`num_episodes=20`，`reset_arm: home`，每集 60 秒。成功是回合结束时方块在容器里。复位日志里会先出现一次随机方块和 YAML 里的 home，然后评测用这份文件覆盖两者；引用时带上 `Episode init` 那一行，包括 `qpos`。归一化统计用该检查点训练时的数据集。

正式报分用 `LAB01_EVAL_SEED` **1000、1001、1002** 各跑完这 20 集。种子只改变策略采样动作块时的噪声序列，不增加方块点，也不是 Home60。ACT 与 SmolVLA 的正式行用 `n_action_steps=50`。其他模型的开环步数写在后面各节。

跨模型比较时用 `reset_arm: home`、夹爪 **0.85**、这份方块清单。每个策略保留自己的相机映射、状态维数和推理后端，不同就写明。ACT 如果把手臂留在 `follow`，就不是这项考试。另一组 20 个点的分数不记在这里。

全范围新抽（不设 `LAB01_EPISODE_INIT`）是另一列。单个演示位姿 `(0.27, 0.20, yaw −8°)` 也不是这一列。

`LAB01_HOME20_SPAWNS` 在本目录 `_env.sh` 里，默认就是这份清单。训练和评测入口仍是 `../lab01_pnp/train_*.cmd` 和 `eval.cmd`。A30 上的分数用 `.venv-cuda` 和 `MUJOCO_GL=egl` 直接调用 `simstudio.scripts.eval`。`eval.cmd` 会优先选 ROCm 环境，不用于这些数字。

## 成绩

ACT 的 53/60 用的是前 140 条，不是这 193 条。其余行都在这 193 条上训练或微调。评测是纯模型，没有专家接管。

| 模型 | 训练 | 成功 | 停在哪 |
|---|---|---|---|
| ACT 6 维 | 只在前 140 条上，50K，从零训 | **18/20、17/20、18/20 = 53/60** | 对照。三次都是接近 20/20、闭合 20/20，漏在抬起之后 |
| SmolVLA follow2 | 193 条上再训 2500 步，绝对关节 | **16/20、18/20、17/20 = 51/60** | 对照。当前采用的 SmolVLA 检查点 |
| MolmoAct2，VLM LoRA | 10K，再从该权重训 20K。视觉和语言只训 LoRA，动作专家全量 | 10K **13/20、11/20、12/20 = 36/60**；20K **13/20、13/20、13/20 = 39/60** | 这条线的成绩。抬起 39，闭合 42 |
| MolmoAct2，合并 LoRA 后全量 5000 | 从 39/60 的权重合并适配器，再训视觉和语言的原权重 | **8/20、6/20、9/20 = 23/60** | 负结果。抬起从 39 降到 25。不替换 39/60 |
| GR00T N1.7 绝对关节，冻结骨干 10K | 从 `nvidia/GR00T-N1.7-3B` 训 10K，`use_relative_actions=false` | **7/20、2/20、5/20 = 14/60** | 合拢 45，抬起 17 |
| GR00T N1.7 绝对关节，全量解冻 | 从上面的 10K 再训两段 5000 步 | **6/20、4/20、5/20 = 15/60** | 这条线的成绩。闭合 57，抬起 20。之后加重修正 12/60、动作块 8 步 11/60，已暂停 |
| pi05，只训动作专家 10K | 从 `ant0nh/pi05_pnp_425_25k` 训 10K，绝对关节 | **4/20、6/20、2/20 = 12/60** | 合拢 39，抬起 15。不要和全量解冻的仓库混用 |
| pi05 全量解冻 | 前方相机，三段 5000 步 | **12/20、11/20、11/20 = 34/60** | 这条线的成绩。闭合 58，抬起 38 |
| DM0.5 | 从 `lerobot/dm05_base` 训 10K，俯视加腕部，绝对关节 | **7/20、7/20、11/20 = 25/60** | 闭合 59，抬起 27。见下文 |
| GR00T N1.7 相对动作 | 同一基座，手臂学增量，夹爪保持绝对值 | 10K **0/60**（检查点已删）；30K **1/60** | 10K 能靠近（8 cm 内 54/60），30K 连靠近都退到 11/60 |
| FastWAM | 4K，视频网络冻结 | **0/60** | 4 cm 内合计 3/60，合拢 0 |
| JEPA，冻结 Qwen，6 维关节 | 只训动作头，10K | **0/60** | 能靠近，闭环里不发合拢 |
| JEPA，原生 7 维末端 | 冻结 Qwen，动作是末端位姿增量，10K | **0/60** | 三个种子的 `reach_8cm` 都是 0/20。比 6 维更差，这条线停止 |

SmolVLA 上一档 actonly 3K（171 条）是 **16/20、15/20、14/20 = 45/60**，不是当前对照。未设种子的 ACT **17/20** 和 SmolVLA **12/20** 也不是这一列。

ACT 日志 `outputs/eval/act_state6_v4_lab01_pnp_home20_g085_seed1000.log` 及 `seed1001`、`seed1002`。SmolVLA follow2 日志 `outputs/eval/smolvla_v4_actonly_follow2_lab01_pnp_home20_g085_seed1000.log` 及另外两个种子。这两份权重的下载链接在下面的表里。

DM0.5 的训练和三种子漏斗在下面的 DM0.5 一节。下载链接在基座表里。

### 可以定下来的话

同一套轨迹够用来学会这个抓放。动作就是数据集里的 6 个关节角度时，ACT 53/60，SmolVLA 51/60。

相对动作不适合这次评测。GR00T 按 LeRobot 官方 SO-101 示例把 `use_relative_actions` 设为 true 之后，10K 步能走到 8 cm，最后几厘米停住；把同一配方拉到 30K，靠近从 54/60 掉到 11/60，成功只有 1/60。

把标签改回绝对关节角度之后，GR00T 冻结骨干 10K 升到 14/60。靠近和合拢都有了，失败变成夹空：合拢 45 次里只有 17 次把方块抬起。当时的 pi05 只训动作专家，成功 12/60，合拢 39 次，抬起 15 次。后来全量解冻把 pi05 抬到成功 34/60、抬起 38，把 GR00T 抬到成功 15/60、抬起 20。GR00T 的抬起停在 20–21，没有再涨。

在同一批数据上继续加步数，没有再抬高这几个缺口。MolmoAct2 的 LoRA 从 36/60 到 39/60，抬起两次都是 39，多出来的 3 次成功是原先已经抬起、只差松开的局，掉落从 4 次变成 0。姿态 0、1、5、6、9、17 在这版 LoRA 上仍然几乎不在 4 cm 内合拢。2026-10-11 把这版 LoRA 合并后再全量训 5000 步，成功退到 23/60，抬起退到 25。绝对关节 GR00T 再用学习率 `1e-5` 续训 5K，种子 1000 从 7/20 降到 4/20（抬起 8 次降到 4 次）。种子 1001、1002 还在测，这个续训不作为新的成绩列。

FastWAM 和两版 JEPA 的 0/60 不是再加轨迹就能补上的。FastWAM 的原权重是 7 维动作，改成 6 维时相关张量重新初始化，视频网络仍冻着。6 维 JEPA 不继续解冻 Qwen。7 维末端连 8 cm 都进不去，正常化统计和派生数据集是对齐的，不是尺度写错。

因此 10 月 5 日停止了当时那一批微调，也不为了冲成功率去补轨迹。之后续了 pi05、GR00T 的全量解冻，以及 MolmoAct2 的一次全量对照。pi05 停在全量解冻那一行。GR00T 停在全量解冻那一行。MolmoAct2 仍引用 LoRA 那一行。再往后的 DM0.5 10K 是 **25/60**，闭合 59、抬起 27，缺口仍是闭合之后的抬起。

## 基座和微调

「基座」是这次实际加载的 Hugging Face 权重。pi05 的上游公共基座是 [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)，训练加载的是它在真机抓放上的一版微调。ACT 没有基座。上传过的微调给出下载链接，用来复现或接着训。实验中间结果没有上传，微调权重一列留空。

| 模型 | 基座 | 微调权重 |
|---|---|---|
| ACT 6 维 50K | 无，从零训练 | [alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4) |
| SmolVLA follow2 2.5K | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | [alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2) |
| MolmoAct2 LoRA 20K | [lerobot/MolmoAct2-SO100_101-LeRobot](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) | [alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k) |
| MolmoAct2 全量 5000 | 同上，从 LoRA 20K 合并后继续 | |
| pi05 只训动作专家 10K | [ant0nh/pi05_pnp_425_25k](https://huggingface.co/ant0nh/pi05_pnp_425_25k) | [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193) |
| pi05 全量解冻 | 同上 | [alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze) |
| GR00T 绝对关节、冻结骨干 10K | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | [alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k) |
| GR00T 全量解冻 | 同上，从冻结 10K 继续 | |
| GR00T 相对动作 | 同上 | |
| FastWAM（193 条） | [lerobot/fastwam_base](https://huggingface.co/lerobot/fastwam_base) | |
| JEPA 6 维（193 条） | [lerobot/VLA-JEPA-LIBERO](https://huggingface.co/lerobot/VLA-JEPA-LIBERO) | |
| JEPA 7 维末端 | 同上 | |
| DM0.5 10K | [lerobot/dm05_base](https://huggingface.co/lerobot/dm05_base) | [alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k) |
| LingBot-VA | [lerobot/lingbot_va_libero_long](https://huggingface.co/lerobot/lingbot_va_libero_long) | |

训练数据：[alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2)。50 条 teleop 上的旧 GR00T 在 [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17)，不要和绝对关节仓库混用。

更早的 50 条遥操作用的是同一批基座，仓库不同，成绩不在这一列：

| 模型 | 基座 | 微调权重 |
|---|---|---|
| SmolVLA 50K | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | [alexhegit/so101-simstudio-lab01-pnp-smolvla](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla) |
| ACT 15 维 | 无，从零训练 | [alexhegit/so101-simstudio-lab01-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act) |
| ACT 6 维 | 无，从零训练 | [alexhegit/so101-simstudio-lab01-pnp-act-state6](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-state6) |
| MolmoAct2 | [lerobot/MolmoAct2-SO100_101-LeRobot](https://huggingface.co/lerobot/MolmoAct2-SO100_101-LeRobot) | [alexhegit/so101-simstudio-lab01-pnp-molmoact2](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-molmoact2) |
| VLA-JEPA | [lerobot/VLA-JEPA-LIBERO](https://huggingface.co/lerobot/VLA-JEPA-LIBERO) | [alexhegit/so101-simstudio-lab01-pnp-vla-jepa](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-vla-jepa) |
| FastWAM | [ZibinDong/fastwam_libero_uncond_2cam224](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224) | [alexhegit/so101-simstudio-lab01-pnp-fastwam](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-fastwam) |
| GR00T N1.7 | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | [alexhegit/so101-simstudio-lab01-pnp-groot-n17](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-groot-n17) |

## MolmoAct2

39/60 是全量实验之前就已经得到的成绩。全量实验只用来检验：把 LoRA 合并进原权重、再按 pi05 的办法训练视觉和语言，能不能超过 LoRA。结果是回退。

相机是俯视和腕部（`camera_top` → `cam0`，`camera_wrist` → `cam1`），前方相机不进这个模型。`n_action_steps=30`。

| 版本 | 种子 1000 / 1001 / 1002 | 成功 | 4 cm 内 | 闭合 | 抬起 | 松开后仍在盒内 | 掉落 |
|---|---|---:|---:|---:|---:|---:|---:|
| LoRA 10K | 13 / 11 / 12 | 36/60 | 60 | 43 | 39 | 36 | 4 |
| LoRA 再训 20K | 13 / 13 / 13 | **39/60** | 60 | 42 | 39 | 39 | 0 |
| 合并后再全量 5000 | 8 / 6 / 9 | 23/60 | 59 | 42 | 25 | 23 | 1 |

8 cm 内三版都是 60/60。

两段 LoRA 的开关相同：`enable_lora_vlm=true`，`enable_lora_action_expert=false`，`freeze_embedding=true`。加载时先冻住全部原权重。

| 部分 | 梯度 | 原权重 |
|---|---|---|
| 语言 transformer，36 层的 `att_proj`、`attn_out`、`ff_proj`、`ff_out` | LoRA，秩 64，`lora_alpha=16` | 冻住 |
| 视觉 backbone 的 `wq/wk/wv/wo`、`w1/w2/w3`，以及 `patch_embedding` | 同上 | 冻住 |
| 动作专家，588 个张量 | 全部打开，没有 LoRA | 在训 |
| 词嵌入 `wte`、`lm_head` | 关闭 | 冻住 |

LoRA 打开时，视觉、语言和连接层的学习率被代码写成 `5e-5`，动作专家是 `5e-5`。命令行上的 `optimizer_lr=1e-5` 在这个开关下不作用到这些组。

第一段从基座训 10000 步，4 张 MI210 做 FSDP，每卡批量 1。权重 `outputs/train/lab01_pnp_molmoact2_state6_agent193/checkpoints/010000`。

第二段不 `--resume`。从第一段权重再训 20000 步，2 张 MI210，warmup 500，`scheduler_decay_steps=20000`，末端学习率 `1e-6`。

2026-10-11 从 LoRA 20K 加载，把 LoRA 合并进原权重，然后关掉 LoRA，训练视觉、语言和动作专家的原权重。词嵌入仍冻结。可训练参数 5,046,031,088，总参数 5,442,196,208。

| | |
|---|---|
| 数据 / 相机 | 仍是这 193 条，仍是俯视加腕部 |
| 步数 / 批量 | 5000 / 1，单张 MI210，显存 42.9 GB |
| dtype | bfloat16，梯度检查点开着 |
| 学习率 | 语言峰值 `1e-5`，视觉 `5e-6`，连接层 `5e-6`，动作专家 `5e-5` |
| 调度 | warmup 200，`scheduler_decay_steps=5000`，末端 `1e-6`。优化器新开 |
| 用时 | 约 1.33 秒/步，5000 步约 1 小时 54 分 |

4 cm 内 59，闭合 42，和 LoRA 20K 一样；抬起从 39 降到 25，成功少了 16 次。全量更新改坏了 LoRA 已经学会的抬起，没有补上那 21 次没做成的局。权重不上传，也不覆盖 Hub 上的 LoRA 仓库。同一配方的全量微调不再做。

pi05 全量解冻有效，是因为之前视觉塔是冻住的，接触这一下学不到。MolmoAct2 的 LoRA 已经在更新视觉和语言，动作专家又是全量的。缺的不是原权重完全没梯度。

## pi05

相机是腕部和前方，俯视相机在数据里但不进全量解冻这一版。`n_action_steps=50`。

| 种子 | 成功 | 抬起 | 闭合 | 掉落 |
|---|---:|---:|---:|---:|
| 1000 | 12/20 | 13 | 20 | 1 |
| 1001 | 11/20 | 13 | 19 | 2 |
| 1002 | 11/20 | 12 | 19 | 0 |
| 合计 | **34/60** | **38** | **58** | **3** |

漏斗合计：8 cm 内 60，4 cm 内 60，闭合 58，抬起 38，到盒子上方 37，抬起后松开 35，某拍已放入 34，最终仍在盒内 34，抬起后掉落 3。

目录名是 `005000`，因为每一段 5000 步都重新计数。这是第三段的权重，不是从零数到 1.5 万步的同一个优化器。只训动作专家的 12/60 仍在单独的 Hub 仓库，不要覆盖。

另有一次数字也是 34/60：前方相机 10K 走到夹爪距方块 8 cm 且夹爪仍张开，然后由脚本专家完成抓取和放置，三次是 12/20、11/20、11/20。那次的抬起和放入来自专家。上面这张表是模型自己做完。专家再加一次重抓的 59/60 同样是专家分数，不记成模型成绩。

起点都是 `ant0nh/pi05_pnp_425_25k`。数据始终是这 193 条，除非下表写了额外轨迹。

| 尝试 | 训练了什么 | 成功 | 抬起 |
|---|---|---:|---:|
| 10K，俯视映射成 front，只训动作专家 | 动作专家 | 12/60 | 15 |
| 前方相机重新训 10K，仍只训动作专家 | 动作专家 | **17/60** | 22 |
| 把成功的 8 cm 抬起片段蒸馏回去，视觉仍冻结 | 动作专家 | 10/60 | 13 |
| 从前方 10K 全量解冻 2500 步 | 视觉、语言、投影、动作专家 | 20/60 | 23 |
| 只解冻视觉、投影和动作专家，语言模型冻结 | 约 11 亿参数 | 17/60 | 23 |
| 用上一行 20/60 的模型录重抓整段，混进 193 再全量解冻 2500 步 | 全模型 | 19/60 | 21 |
| 从前方 10K 重新全量解冻 5000 步，批量 2 | 全模型 | 20/60 | **28** |
| 从上一行权重再训 5000 步，新优化器，批量 1 | 全模型 | **33/60** | 36 |
| 再训 5000 步，同一配方 | 全模型 | **34/60** | 38 |

冻结视觉时，多训、降低学习率、重抓后缀、闭合前片段、抓取窗加权、把开环从 50 步改成 10 步，都只让闭合变多，抬起停在 22 次附近。最好的冻结权重是前方相机 10K 的 17/60。把专家抬起片段蒸馏进这个冻结模型，掉到 10/60。

只解冻视觉塔和投影、语言模型继续锁住，停在 17/60。全量解冻 2500 步到 20/60，但相对前方 10K 是 12 个姿态从失败变成功、9 个从成功变失败，抬起只多 1 次。

从前方 10K 重新开一段更长的全量解冻之后，抬起先到 28/60，成功仍是 20/60。再开一段新的余弦、峰值学习率仍是 `2.5e-6`，成功到 33/60，抬起 36。同一配方再加 5000 步，成功 34、抬起 38。靠近、闭合、掉落和 33/60 那一版相同（60、58、3）。这一步已经停住。

重抓整段（靠近、第一次空夹、专家重抓、放入）按约 19% 混进 193，从 20/60 的权重接着训，得到 19/60。这条没有超过只训 193 的全量解冻，没有并进最终权重。

最后这一版的配方：

| | |
|---|---|
| 初始化 | 上一段 33/60 的权重，不是从 `ant0nh` 重来 |
| 可训练参数 | 4,143,404,816，等于总参数 |
| `train_expert_only` | false |
| `freeze_vision_encoder` | false |
| 数据 | 193 条，不混重抓或蒸馏片段 |
| 相机 | `camera_wrist` → `wrist`，`camera_front` → `front` |
| 状态 / 动作 | 6 维绝对关节 `.pos`，`use_relative_actions=false` |
| 步数 / 批量 | 5000 / 1 |
| 学习率 | 峰值 `2.5e-6`，warmup 200，`decay_steps=5000`，末端 `2.5e-7` |
| 优化器 | 新开。不 `--resume` 已经走完的余弦 |
| `num_workers` | 0。全量解冻时 worker 大于 0 会在 CUDA 初始化之后卡死 |
| chunk / `n_action_steps` | 50 / 50 |
| dtype | bfloat16，梯度检查点开着 |

`decay_steps` 必须等于这次的 `--steps`。短于默认 30000 时，LeRobot 的余弦会把 warmup 和衰减再缩放一次，峰值就不是 `2.5e-6`。三段 5000 步都是这个学习率。第一段批量 2，后两段批量 1。后两段的优化器都重新开始，不接着上一段已经衰减到 `2.5e-7` 的状态。

没做成的，多数是闭合了但没有抬起：闭合 58，抬起 38。掉落只有 3 次。同一套 5000 步再训，成功从 33 到 34，抬起从 36 到 38，三个种子的波动和上一版同一量级。再加步数不是下一步。若还要抬这版模型，补的应是它自己空夹的轨迹，而不是前面已经试过的重抓整段或冻结蒸馏片段。

## GR00T

相机是前方、俯视、腕部三路，绝对关节，同步推理。正式行的 `n_action_steps=16`。

| 版本 | 种子 1000 / 1001 / 1002 | 成功 | 4 cm 内 | 闭合 | 抬起 | 到盒子上方 | 掉落 |
|---|---|---:|---:|---:|---:|---:|---:|
| 冻结骨干 10K（起点） | 7 / 2 / 5 | 14/60 | 51 | 45 | 17 | 14 | 4 |
| 全量解冻 2500 步 | 3 / 2 / 2 | 7/60 | 54 | 48 | 8 | 7 | 1 |
| 从起点重新全量解冻 5000 步 | 4 / 4 / 5 | 13/60 | 60 | 60 | **21** | 14 | 6 |
| 从上一行再训 5000 步 | **6 / 4 / 5** | **15/60** | 60 | 57 | 20 | 15 | 5 |
| 从 15/60 再训 10000 步，第 100–139 条采样 ×4 | 7 / 2 / 3 | 12/60 | 49 | 36 | 17 | 12 | 3 |
| 不训，只把 15/60 的 `n_action_steps` 改成 8 | 4 / 5 / 2 | 11/60 | 51 | 45 | 16 | 12 | 5 |

表里前四行的 8 cm 内都是 60/60。15/60 那一版松开后仍在盒内 15，某拍已放入 15。三个种子各差 1–2 次，比冻结骨干多出的 1 次还在噪声范围里。

15/60 的权重约 12 GB，目录名 `005000` 是因为每一段重新计数。上一段 13/60 在 `/tmp/lab01_groot_abs_unfreezeA2_5k/checkpoints/005000/pretrained_model`。两者都在 `/tmp`，重启前要拷走。冻结骨干的 Hub 仓库不要覆盖。训练取前 6 维关节时设 `LAB01_STATE_DIM=6`。

冻结骨干 10K 能走到方块旁、也会闭合，但闭合 45 次只抬起 17 次。pi05 冻结视觉时是同一种失败，全量解冻后抬起明显增加，所以 GR00T 先照搬全量解冻。

第一次只训 2500 步，从 14/60 掉到 7/60，抬起从 17 降到 8。闭合反而变多。GR00T 的余弦按 `max_steps` 衰减到接近 0，2500 步结束时学习率已是 `1.4e-10`，模型只看了约 0.02 个 epoch 就停了，行为先往更爱闭合、更少抬起偏。这些权重不再作为后续起点。

第二次回到冻结骨干 10K，重新开一段 5000 步的余弦。成功 13/60，比起点少 1 次，但 4 cm 内和闭合都到了 60/60，抬起升到 21。

第三次从 13/60 的权重再开一段 5000 步，优化器重新开始，学习率曲线和上一段相同。成功 15/60，抬起 20，闭合 57。多出来的 2 次成功来自种子 1000（4→6），另外两个种子不变。抬起没有继续上升。

最后这一版的配方：

| | |
|---|---|
| 初始化 | 13/60 那段 5000 步的权重，`--policy.path`，不传 `--policy.type` |
| 可训练参数 | 3,144,016,000，等于总参数 |
| `tune_llm` / `tune_visual` | true / true |
| `tune_projector` / `tune_diffusion_model` / `tune_vlln` | true（默认） |
| 数据 | 193 条，不混重抓或蒸馏片段 |
| 相机 | 前方、俯视、腕部三路 |
| 状态 / 动作 | 6 维绝对关节 `.pos`，`use_relative_actions=false`，`action_decode_transform=none` |
| chunk / `n_action_steps` | 16 / 16 |
| 步数 / 批量 | 5000 / 1 |
| 学习率 | 峰值 `optimizer_lr=2.5e-6`，`warmup_ratio=0.04`（200 步），`max_steps=5000`，余弦衰减到约 0 |
| 优化器 | 新开。不 `--resume` 已经走完的余弦 |
| `num_workers` | 0 |
| dtype | `use_bf16=true`，`use_flash_attention=false` |
| 显存 | 单张 MI210 约 58.75 GB，约 1.1 步/秒，5000 步约 76 分钟 |

GR00T 的调度器和 pi05 不同。`GrootConfig.get_scheduler_preset` 用 `max_steps` 和 `warmup_ratio` 算 warmup，并衰减到约 0，没有 pi05 那样的末端学习率。`--policy.max_steps` 必须等于这次的 `--steps`，否则 warmup 和衰减按默认步数算。pi05 的 `scheduler_decay_steps` 一类参数对 GR00T 不起作用。

全量解冻时 `tune_llm` 和 `tune_visual` 直接作为 `--policy.*` 覆盖传入，从微调过的检查点加载时也会生效；日志里 `num_learnable_params` 应等于 `num_total_params`。

闭合已经不是问题：4 cm 内 60，闭合 57–60。没做成的几乎都卡在闭合之后。闭合后能抬起的比例三版都在 35–38%：起点 17/45，13/60 那版 21/60，15/60 这版 20/57。pi05 全量解冻是 38/58。抬起之后还有 5–6 次掉落，或没送到盒子上方。缺的是夹住这一下的精度和搬运时的稳定。同样的 193 条已经训了三段，抬起停在 20–21。

2026-10-11 又试了两件仍只用这 193 条的事，都低于 15/60，这条线暂停。

加重已有修正。第 100–139 条（闭合前对准、再抓、DAgger）原来只占 16.6% 的帧。采样重复到 4 倍（日志 `corr4x episodes 100-139 x4: base_frames=130835 extra_frames=65103`），从 15/60 的权重再训 10000 步，峰值学习率仍是 `2.5e-6`，`warmup_ratio=0.02`，`max_steps=10000`，优化器重新开始。成功 12/60，抬起 17，4 cm 内 49，闭合 36。靠近和闭合都退了。权重 `/tmp/lab01_groot_abs_corr4x_10k/checkpoints/010000/pretrained_model`，不作为成绩。

只改评测。同一 15/60 权重，`n_action_steps` 从 16 改成 8：成功 11/60，抬起 16，闭合 45。

15/60 的空夹是夹爪偏开方块约 2–4 cm、两爪几乎没有同时夹上。193 条里已有的修正片段再加重，没有补上这个偏差。要再提高，得用这一版模型自己的空夹状态去采短修正，那一步没有开始。

不要再做的：同一配方再加步数；加重第 100–139 条；`n_action_steps=8`；从 7/60 或 12/60 的权重接着训；`--resume` 已经衰减完的余弦；两路相机；相对动作；冻结骨干再降学习率；旧的重抓整段或蒸馏片段。

## DM0.5

基座是 [`lerobot/dm05_base`](https://huggingface.co/lerobot/dm05_base)。仓库里钉住的 LeRobot 是 v0.6.0，没有 DM0.5，这次训练用的是一份更新的 LeRobot 检出，没有写进子模块。相机是俯视和腕部，前方相机在数据里但不进模型。`add_state=true`，绝对关节，chunk 和 `n_action_steps` 都是 10。图像缩到 448×448。

评测是纯模型 Home20，没有专家接管。日志 `outputs/eval/dm05_base_agent193_10k_lab01_pnp_home20_g085_seed1000.log` 及 `seed1001`、`seed1002`。2026-10-14 10:23 跑完。

| 种子 | 成功 | 4 cm 内 | 闭合 | 抬起 | 到盒子上方 | 掉落 |
|---|---:|---:|---:|---:|---:|---:|
| 1000 | 7/20 | 20 | 20 | 7 | 7 | 0 |
| 1001 | 7/20 | 20 | 19 | 8 | 7 | 0 |
| 1002 | 11/20 | 20 | 20 | 12 | 11 | 1 |
| 合计 | **25/60** | **60** | **59** | **27** | **25** | **1** |

8 cm 内是 60/60。松开后仍在盒内 25，某拍已放入 25。没做成的几乎都是闭合了但没有抬起。

发布的是 10000 步的检查点。[alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k)。5000 步的中间检查点没有上传。

配方：

| | |
|---|---|
| 初始化 | `lerobot/dm05_base`，不是 `dm05_libero` |
| 可训练参数 | 4,486,607,248。`use_peft=false`，词嵌入 `freeze_vlm_embedding=true` |
| 数据 | 193 条。磁盘上的状态是 15 维，训练取前 6 维关节 `.pos` |
| 相机 | `camera_top`、`camera_wrist` |
| 状态 / 动作 | 6 维绝对关节，`use_relative_actions=false`，`add_state=true` |
| chunk / `n_action_steps` | 10 / 10 |
| 步数 / 批量 | 10000 / 每卡 1。4 张 MI210 做 DDP，`dp_replicate=4`，有效批量 4 |
| 学习率 | 峰值 `2.5e-5`，warmup 400，`decay_steps=10000`，末端 `2.5e-6` |
| dtype | bfloat16，视觉和动作专家的梯度检查点开着 |
| 用时 | 2026-10-13 15:58 到 20:34，约 4 小时 36 分。约 1.65 秒/步，每卡约 51 GB。结束时日志损失 0.007，约 0.31 个 epoch |

`decay_steps` 必须等于这次的 `--steps`，否则余弦会把峰值再缩放一次。

## 和 NVIDIA SO-101 GR00T 课程不是同一张考卷

课程是 [Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)。仿真评测页写的期望成功率是 **50–70%**：[Sim Evaluation](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/11-sim-evaluation.html)。示例日志是跑到第 7、8 集时的累计 66.7% 和 71.4%，不是冻住 20 个点、三个种子的总分。

那个 50–70% 不能拿来对我们的冻结骨干 14/60。两边具体差在：

| | NVIDIA 课程 | 这里的 Home20 |
|---|---|---|
| 模型 | GR00T **N1.6**-3B | GR00T **N1.7**-3B |
| 仿真 | Isaac Lab，离心管放进黄色试管架 | MuJoCo，方块放进盒子 |
| 数据 | 75 条仿真遥操作 | 193 条脚本、DAgger 和 ACT follow |
| 微调 | `global_batch_size=64`，最多 20000 步，学习率 `1e-4` | batch 4，10000 步。样本数大约是课程配方的 1/32 |
| 动作 | SO-100 配置：手臂 **相对** 当前关节，夹爪 **绝对**。见 [so100_config.py](https://github.com/NVIDIA/Isaac-GR00T/blob/n1d6/examples/SO100/so100_config.py) | 相对动作 0/60 和 1/60；绝对关节冻结骨干 14/60 |
| 成功 | 瓶子开始进入架子槽位就可以重置 | 要抬起、移到盒子上方、松开，并且最后仍在盒子里 |

课程自己的配方也是相对手臂动作。他们在 Isaac 里、用大约 32 倍的样本，在另一个任务上做到教程期望的 50–70%。这不说明同样的相对动作配方在我们的 Home20 上应该有同样的分数。我们这边相对动作已经测过，绝对关节更高，但抬起仍然是缺口。

## 发布

脚本在本目录。`labs/lab01_pnp/` 下的同名文件会转到这里。

| 脚本 | 内容 |
|---|---|
| `push_act_agent_yaw_v4_model.cmd` | ACT agent-yaw v4 |
| `push_smolvla_follow2_model.cmd` | SmolVLA follow2 |
| `push_molmoact2_agent193_model.cmd` | MolmoAct2 LoRA 20K |
| `push_pi05_agent193_model.cmd` | pi05 只训动作专家 10K |
| `push_pi05_agent193_unfreeze_model.cmd` | pi05 全量解冻 |
| `push_groot_agent193_abs_model.cmd` | GR00T 绝对关节、冻结骨干 10K |
| `push_dm05_agent193_model.cmd` | DM0.5 10K |
| `push_agent_yaw_follow2_dataset.cmd` | 193 条数据集 |

50 条 teleop 数据集和对应的 ACT / SmolVLA / MolmoAct2 / GR00T / JEPA 上传脚本仍在 `labs/lab01_pnp/`。Hub 卡片 `hf_model_card_*.md`、`hf_dataset_card_*.md` 是上传稿，不在这里维护对照表。
