# Lab 01 ACT agent-yaw 快速改进 — 2026-09-27

**目标：** 尽快把 home 协议下「闭合后抬不起」的比例降下来。  
**已知：** home n=20 无头 close 20/20、lift 4/20、success 3/20；.43 GLFW n=3 同样 3/3 闭合、1/3 抬起。follow 的接近失败先不管。  
**不做：** GR00T；home n=50 基线；逐项大网格；200 条满配额；四个 checkpoint 都评；follow / teleop 对照。这些等 v2 的 n=20 明显好于 15% 再做。  
**不覆盖** `so101-simstudio-lab01-pnp-agent-yaw`，不 resume `lab01_pnp_act_agent_yaw/080000`。

## 时间

评估放在 .43 的 GLFW：刚才 3 局约 1 分钟一局。训练仍在 DORobot。串行大约 **12–16 小时**。

| 步 | 内容 | 预估 |
|---|---|---:|
| 1 | 脚本抓取小测试：无扰动、XY ±5 mm、roll ±5°，以及夹爪收到 -0.17 | 1–2 h |
| 2 | 专家加上一次重抓和 home 附近小抖动，采集约 80 条成功 | 4–6 h |
| 3 | ACT 15-D 从零训 40K，bs64 | 5–8 h |
| 4 | .43 GLFW，home，n_action_steps=50，n=20 | ~0.5 h |

n=20 若 lift 仍约 20%，停，回到第 1 步改抓取，不要直接加到 80K。若 lift 明显上升，再把同一任务续到 80K，然后补 home n=50。

## 1. 脚本小测试 — 已完成（2026-09-27，.43 EGL，每种 10 次，只看到抬起）

日志：`outputs/eval/grasp_probe_p0927.log`。偏差加在对准之后、闭合之前。XY 偏移这一步经常没走完（剩余误差约 6 mm），所以不是干净的 ±5 mm。

| 条件 | 抬起 |
|---|---:|
| 无扰动，夹爪 -0.08 | 8/10 |
| X +5 mm | 8/10 |
| X −5 mm | 7/10 |
| Y +5 mm | 7/10 |
| Y −5 mm | 7/10 |
| 腕角 +5° | 6/10 |
| 腕角 −5° | 6/10 |
| 夹爪 -0.17 | 8/10 |
| 夹爪 -0.17 且 X +5 mm | 7/10 |
| 夹爪 -0.17 且 X −5 mm | 9/10 |

脚本自己在这些偏差下仍能抬起约 60%–90%。把夹爪收到 -0.17 没有提高抬起率。策略在 home 下闭合后只有约 20% 抬起，差距不在“夹得不够紧”。下一步是第 2 步，这一步已经做完，见下节。

## 2. 采集 v2 — 三类补采完成（2026-09-27，.43 EGL）

数据集：`datasets/so101-simstudio-lab01-pnp-agent-yaw-v2`。最终落盘 **188 条、77471 帧**；原来的 agent-yaw 仍是 100 条，没有写入。故意失误的动作没有进标签，只保留纠正段。

构成：一次成功 78 条，自然重抓 5 条，固定 recovery 20 条（夹爪 +0.50），多样化 recovery 40 条，完全闭合空抓 recovery 20 条，闭合前修正 15 条，v1 ACT 80K 交接 10 条。recovery 共 40742 帧，约占 53%。

- 完全闭合空抓：夹爪收到 -0.08 且方块没抬起后才开始录，再打开、重新对准、抓取。日志 `outputs/collect/agent_yaw_v2_closed_empty_p0927.log`、`..._pass2_p0927.log`。
- 闭合前修正：夹爪仍张开时做 8–12 mm / ±5–10° 的对准，然后录一次正常闭合。日志 `outputs/collect/agent_yaw_v2_preclose_p0927.log`。
- 策略交接：home、`n_action_steps=50` 跑 v1 ACT 80K，只在闭合后 40 拍仍没抬起时交给专家。策略动作不保存。方块被挤过 12 mm 的样本丢掉。第一轮 40 次存下 5 条；放宽到 12 mm 后补到 10 条。日志 `outputs/collect/agent_yaw_v2_policy_handover_p0927.log`、`..._pass2_p0927.log`。

`validate_dataset`：**188 episodes、77471 frames，PASSED**，parquet 条数和帧数一致，没有临时视频。这不能保证 80% 成功率。按当前失败结构，比较现实的结果是 home 成功率大约 50%–70%。训练后的 home n=20 仍以 lift ≥ 12/20、success ≥ 8/20 作为是否续训的门槛。

### 末端位置修正（2026-09-27 晚）

采集用指尖 `gripperframe` 作为 `ee.pos`，但 `rollout_act.yaml` 未设 `ee_site_name`，默认用 `wrist_site`（同一姿态 z 高约 0.16 m）。所以之前所有 agent-yaw ACT 评估里，策略看到的末端位置都是错的；评估的初始夹爪也是 -0.1，而采集时是 0.85。v2 的 10 条策略交接（第 178–187 条）也录成了 `wrist_site`。这 10 条已用关节角重算为 `gripperframe`，每集统计和全局统计同步更新。原文件备份在 `.tmp/yaw-v2-ee-fix-p0927/backup/`。本机和 DORobot 的 `validate_dataset` 都通过。

v1 80K 重新评估：home，`n_action_steps=50`，n=20，EGL，`--robot.ee_site_name=gripperframe --robot.home_gripper=0.85`。结果 close 20/20、**lift 7/20、success 7/20（35%）**，抬起后没有掉落。之前错误配置是 lift 4/20、success 3/20。日志 `outputs/eval/act_yaw_80k_home_gripperframe_n20.log`。之后所有 agent-yaw 评估都必须带这两个覆盖参数。

## 3. 训练 — 40K 已完成（2026-09-27）

DORobot，目录 `/mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw_v2`。ACT 15-D，bs64，40K，`save_freq=20000`，只用 v2，没有 resume `080000`。用时 6 小时 31 分，40K loss **0.025**。本机已有 `outputs/train/lab01_pnp_act_agent_yaw_v2/checkpoints/040000`。

评估配置已单独写成 `labs/lab01_pnp/configs/rollout_act_agent_yaw.yaml`：`ee_site_name=gripperframe`，`home_gripper=0.85`，`reset_arm=home`。通用的 `rollout_act.yaml` 没有改。

## 4. 验收 — 40K 未过线，不续训 80K

.43 EGL，上述专用配置，`n_action_steps=50`，n=20。日志 `outputs/eval/act_yaw_v2_40k_home_n20.log`。

| 模型 | 接近 | 闭合 | 抬起 | 成功 |
|---|---:|---:|---:|---:|
| v1 80K，正确配置 | 20/20 | 20/20 | 7/20 | 7/20 |
| v2 40K | 20/20 | 17/20 | 8/20 | 7/20 |

门槛是 lift ≥ 12/20 且 success ≥ 10/20。40K 与 v1 新基线持平，**没有启动 80K**。抬起后放进盒子基本能完成（8 次抬起里 7 次成功，1 次超时仍举在盒子上方）。主要失败仍是闭合后没抬起。

## 5. 清洗数据集 v2-clean（2026-09-28）

`datasets/so101-simstudio-lab01-pnp-agent-yaw-v2-clean`：93 条 = 78 条一次抓取成功（v2 的 0–79 去掉二次抓取的 24、67）+ 15 条 pre_close（163–177）。
去掉了 80–82 的回撤重抓、83–162 与 178–187 这些不从 home 开始的恢复片段。
脚本 `labs/lab01_pnp/rebuild_clean_dataset.py`，对应关系在 `source_episodes.json`，`spawns.jsonl` 已按新编号过滤。validate PASSED，各路视频帧数都等于 episode 长度。v2 本身未改动，尚未上传 HF。

**视频尾巴的原因**：v2 有 30 条视频比数据长 2–8 秒。LeRobot v0.6 的 `clear_episode_buffer` 只删除 `image` 类型特征的临时 PNG，不删 `video` 类型，所以被丢弃尝试的高序号帧留在磁盘上，被下一条保存的 episode 编码到末尾（23/30 条的多余帧数正好等于上一条丢弃尝试的帧数减去本条长度）。训练只读前 `length` 帧，不受影响，但 viewer 会显示。
`ScriptedRecorder._discard_episode_buffer` 已在项目侧补删，复现脚本 `.tmp/expert-grasp-p0928/repro_clear.py`。
24、67 的三路相机互相关滞后为 0，HF viewer 上看到的"不同步"是 viewer 的问题；这两条因二次抓取已从 clean 中去掉。

## 6. 专家抓取修复（2026-09-28）

探针 `.tmp/expert-grasp-p0928/probe_contact.py`（只读仿真，不写数据集），闭合后记录两爪接触力和夹爪轴与接近面法向的夹角。

- 修复前首次抓取抬起 29/40。11 次失败中 8 次是空抓：两爪接触力约 0 N，夹爪轴与接近面法向几乎平行（1–5°），成功的都是 82–90°。原因是腕部转角按"方块四面对称"选最近的解，接近面则另外按"离夹爪最近"选，两者 90° 错位时两爪并排顶在面上。
- 修复：`_grasp_face_and_roll` 先比较两对面，选使夹爪轴垂直于接近面法向、且腕部转角离中性位最近的一对，但该面偏离夹爪来向不超过 `FACE_MAX_OFF_APPROACH_DEG=75`。转角超过约 1.1 rad 时即便夹爪轴垂直也抓不住，所以按转角排序。
- 抬起率：修复后 3 组种子合计 118/120（60° 阈值 114/120，45° 更差）。完整抓取加放置，新种子 4242：35/40，抬起的 35 次全部放进盒子。
- 接触确认：`GRASP_CONTACT_MIN_N=1.0`，闭合后任一爪接触力低于阈值就不抬起。
- 剩余失败：远端区域（x≈0.32–0.34, y≈0.17）指尖沿切向偏 17–21 mm，接触检查能识别为空抓，但还没修。
- 首次接触距离（`contact_reach_m`）扫描 12/16/18 mm 都没有变化，已撤掉。

## 7. ACT 首次闭合诊断（2026-09-28）

`src/simstudio/scripts/eval.py` 的阶段追踪器新增 `first_close`（写入每个 episode 的 funnel 日志）：第一次闭合指令时夹爪轴与接近面法向夹角、指尖相对方块的法向/切向/高度偏差，以及之后 30 拍内两爪最小接触力的最大值、夹爪实际位置、方块被推动距离；另有 `close_events`。
协议同第 4 节（home，`n_action_steps=50`，n=20，EGL），日志 `outputs/eval/act_yaw_v1_80k_graspdiag_n20.log`、`act_yaw_v2_40k_graspdiag_n20.log`。

| 模型 | 成功 | 首次闭合抓住 | 夹爪平行（<30°） | 侧向偏（轴 60–80°，切向约 13–15 mm） | 未闭合 | 首次失败后补救成功 |
|---|---:|---:|---:|---:|---:|---:|
| v1 80K | 9/20 | 9 | 7 | 4 | 0 | 0/11 |
| v2 40K | 12/20 | 8 | 4 | 7 | 1 | 4/12 |

- 本轮方块位置与第 4 节不同，成功数 9 和 12 与上轮的 7、7 都在 n=20 的波动范围内。
- 首次闭合抓住的 17 次：轴 80–90°，法向约 14 mm，切向 ≤9 mm，全部抬起。
- 失败的两种形态与旧专家的两个缺陷一一对应：夹爪平行对应腕部转角 90° 错位；侧向偏对应远端切向漂移。
- 失败集中在 x ≥ 0.30 且转角 −10°..+30°。v1 采集日志中这块区域旧专家 199 次尝试只留下 19 条（10%），其他区域 68–100%。模型在这块区域的示范少且是幸存者。
- v2 的 recovery 数据起了作用：首次失败后补救成功 4 次；v1 最多重试 3 次但 0 次补救成功。

## 8. 远端侧向偏差（2026-09-28）

探针 seed 4242 上残留的 5/40 失败：蠕动被边角擦碰冻住时指尖切向偏 16–21 mm，夹爪轴已垂直但闭合为空抓；再向方块中心 polish 会把方块推 20–30 mm，更糟。根因是工作空间边缘（x ≳ 0.32）手臂够不到中心。

改动（`auto_collect_yaw.py`）：
- 接近面排序加入 standoff 可达距离（在转角代价之后）。
- 首次蠕动 `FIRST_CONTACT_REACH_M=0.018`，忽略过早的边角擦碰。
- 仅当指尖已较近（xy_err ≤ 18 mm）且切向/法向超限时做 restandoff+creep polish。
- 闭合前若 `|tangent| > 10 mm` 质量拒绝（不保存）；法向 20–30 mm 仍允许闭合（指尖短一截也能抓住）。
- 两爪接触门限保留。

4 组种子 × 40 次完整抓取+放置：抬起 **153/160（95.6%）**，放置 **152/160**。seed 4242 仍有 5 次远端拒绝（干净丢弃，不进标签）。采集时预期约 5% 丢弃，远端区域靠配额补齐即可。

## 9. 采集 agent-yaw-v3（进行中，2026-09-28）

新根目录 `datasets/so101-simstudio-lab01-pnp-agent-yaw-v3`（不覆盖 v1/v2/v2-clean）。

| 阶段 | 条数 | 说明 |
|---|---:|---|
| 一次成功 | 100 | yaw 6 档 × XY 2×2 双重配额；无抖动、无高位重抓 |
| 闭合前修正 | 10 | `--recovery-style pre-close`，从 home 录完整纠正 |
| 闭合后原地再抓 | 10 | `--recovery-style closed-empty` + 原地张开再抓（无高位回撤） |

日志：`outputs/collect/agent_yaw_v3_first100_p0928.log`，随后 `..._preclose10_...`、`..._diverse10_...`。

**已完成（2026-09-28）**：120 条 / 52476 帧，validate PASSED，视频帧数与 length 一致。
构成：一次成功 100（yaw×XY 配额满）+ pre_close 10 + diverse 原地再抓 10。
`closed-empty` 在修好的专家上经常假空抓却抬起，改为半开偏置 miss + 原地再抓。
训练：**双机赛跑**（同配方：v3、全局 bs64、40K、`save_freq=20000`、不 resume）。

| 机 | 配置 | 启动 | 日志 / 输出 |
|---|---|---|---|
| **acc** Tailscale `opea@100.118.127.54` | 4×A30，`.venv-cuda`，`accelerate` bs16×4 | 较早 | `..._v3_40k_4xa30.log` → `outputs/train/lab01_pnp_act_agent_yaw_v3` |
| **DORobot** MI300X | 单卡 ROCm，bs64×1 | 后启（已清空目录重开） | `..._v3_40k_mi300x.log` → `/mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw_v3` |

step:200 吞吐：MI300X **smp/s≈79**（updt 0.66 / data 0.15）；4×A30 **smp/s≈47–49**（updt 0.85 / data 0.50，偏 decode-bound）。按此 MI300X 仍应先到 40K。
acc SSH：`Host acc`，IdentityFile `~/ssh_key/amd/id_ed25519`；勿用 acc 上 `.venv-rocm`（看到的是 MI210）。

**A30 已停**（未到 20K ckpt）。Smoke（单卡 A30，bs16，200 step，`accelerate --mixed_precision`）：
| | updt_s | data_s | smp/s | mem_gb |
|---|---|---|---|---|
| FP32 | 0.351 | 0.373 | 22 | 10.90 |
| BF16 | 0.323 | 0.407 | 22 | 10.33 |
BF16 只略减 update（~8%），整体仍 decode-bound，**墙钟几乎无加速**。日志：`acc:outputs/train/smoke_act_{fp32,bf16}_a30.log`。

**num_workers sweep**（acc EPYC 9554×2 / 256 线程，单卡 A30，bs16，200 step，pyav）：
| workers | data_s | smp/s | wall_s |
|---|---|---|---|
| 2 | 1.018 | 12 | 295 |
| 4（默认） | 0.370 | 22 | 161 |
| **8** | 0.050 | 40 | 99 |
| **16** | **0.031** | **41** | **96** |
| 32 | 0.057 | 38 | 104 |
| 64 | 0.114 | 33 | 119 |
推荐 **8–16**；4×GPU 时总 worker≈`num_workers×进程数`，别盲目上 32+。

**已重开 4×A30**（全局 bs64，40K）：
| 配置 | step:200 smp/s | data_s | updt_s | 日志 |
|---|---|---|---|---|
| workers=4（旧） | ~47–49 | ~0.50 | ~0.85 | `..._4xa30.log` |
| workers=8 | 76 | 0.24 | 0.60 | `..._4xa30_nw8.log` |
| workers=32（当前） | 77 | 0.22 | 0.61 | `..._4xa30_nw32.log` |
32 相对 8 **几乎无收益**（总 worker≈128，首步启动更慢）；继续跑 nw32。

**MI300X 计划**：当前无 ckpt 不能中途改 workers。已挂 watcher：等到 `.../checkpoints/020000` 写稳后杀进程，用 `--resume=true --num_workers=8` 续跑到 40K。
- watcher：`DORobot:scripts/mi300x_v3_resume_nw8_watch.sh`（log `outputs/collect/mi300x_v3_resume_nw8_watch.log`）
- resume 日志：`outputs/train/lab01_pnp_act_agent_yaw_v3_40k_mi300x_nw8_resume.log`

## 给下一个 Agent

```bash
ssh DORobot   # 只用于第 3 步训练
# 不要运行 tmp/lab01-agent-p0923.cmd
cat /root/so101-simstudio/tmp/lab01-act-yaw-v2-p0927.md
```

第 1、2、4 步在 .43（本机 `/home/amd/Repo/so101-simstudio`）。GR00T 继续暂停。
