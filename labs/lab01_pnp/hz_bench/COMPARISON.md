# LeRobot 闭环推理对比：8060S (halo) vs MI300X (DORobot)

Protocol: `./labs/lab01_pnp/hz_bench.cmd` → `eval.cmd`，`render_window=false`，固定方块，4 episodes，**丢掉 ep 1**（编译预热）。
halo = Radeon 8060S + EGL；DORobot = MI300X + OSMesa（软件渲染，**control Hz 不可比**，只看 GPU 列）。
原始逐条数据见 [RESULTS.md](RESULTS.md)。日志仍写在 `outputs/eval/hz_bench/`（不入库）。

## 该看哪个指标

sync 和 RTC **不要混在一张主表里**。控制环拿动作的方式不同，同一列「延迟」含义也不同。

三个容易混的长度（**不是都是 8**）：

| 名字 | 配置 | 含义 |
|------|------|------|
| **一块产出几步** `n_action_steps` | 策略 / checkpoint | 一次 GPU 前向写出多少个动作 |
| **RTC 执行窗** `execution_horizon` | `inference.rtc` | 新旧两块重叠、当「已承诺前缀」的步数。只有 GR00T 评测 YAML 写成 **8** |
| **何时再算下一块** `queue_threshold` | `inference.queue_threshold` | 队列剩余 ≤ 这个数时后台才再跑 `predict_action_chunk` |

20 Hz 一拍 = **50 ms**。RTC 队列不饿的粗判据：`chunk_Hz × execution_horizon ≥ 20`（假设大约每执行完一个执行窗就该有新块）。

## Sync：控制线程上算

一次 `get_action` 可能触发前向；`n_action_steps` 用完之前，后续 `get_action` 只是弹出已算好的步。环必须能跟上——前向若 ≥ 50 ms，20 Hz 环就会掉。

| 策略 | 一块产出 | 环上该看的延迟 | 8060S | MI300X | 相对 | 8060S 环 |
|------|--------:|----------------|------:|-------:|-----:|----------|
| ACT 6-D 50K | 50 | `get_action`（整次前向） | 4.3 ms | 8.7 ms | 0.5×* | 51.0 Hz，能 20 Hz |
| ACT 15-D 50K | 50 | 同上 | 4.3 ms | 7.8 ms | 0.6×* | 51.0 Hz，能 20 Hz |
| MolmoAct2 6-D 10K | 30 | `get_action` | 50.5 ms | 45.9 ms | ~1× | 15.3 Hz，贴边 |
| MolmoAct2 15-D 10K | 30 | `get_action` | 66.2 ms | 无权重 | — | 12.8 Hz，不行 |
| VLA-JEPA BC 10K | 30 | `get_action` / 稀疏 chunk | 31 ms / chunk 599 ms | 13 ms / chunk 68 ms | chunk 8.8× | 15.4 Hz |
| FastWAM 14K | 32 | — | 加载失败 | 加载失败 | — | — |

\* ACT 的 `get_action` 在 MI300X 上略慢，是小网络 + 不同软件栈，不是「大卡更慢所以没用」。Molmo 的 `predict_action_chunk` 调用很少，chunk 毫秒样本稀疏，**环里真正付钱的是 `get_action`**。

## RTC：后台算一块，前台每 50 ms 弹出一步

`get_action` 是队列弹出（日志里几千～几万 Hz），**不要拿来比 GPU**。要比的是 **一块 chunk 的墙钟延迟**，并写明这一块对应多少步。

Lab 01 这次 Hz 扫描里只有两个 RTC 策略，**chunk 步数不一样，也不是都是 8**：

| 策略 | 一块产出 `n_action_steps` | RTC 执行窗 `execution_horizon` | 队列阈值 | 8060S 一块延迟 | MI300X 一块延迟 | 相对 | 8060S chunk Hz | 20 Hz 够不够喂 |
|------|--------------------------:|-------------------------------:|---------:|---------------:|----------------:|-----:|---------------:|----------------|
| SmolVLA 50K | **50**（checkpoint 默认；YAML 未覆盖） | **10**（LeRobot RTC 默认；`rollout_smolvla.yaml` 只写了 `type: rtc`） | **30**（factory 默认） | 259 ms | 229 ms | 1.1× | 3.9 | 3.9×10 ≥ 20，队列够；视觉滞后 ~250 ms |
| GR00T N1.7 10K | **16**（Lab 01 `LAB01_GROOT_N_ACTION_STEPS`；N1.7 底模原生可到 40） | **8**（`rollout_groot.yaml` 写死） | **0** | 465 ms | 126 ms | 3.7× | 2.1 | 需要 ≳ 20/8 = **2.5 Hz**；MI300X 8 Hz 够，8060S 2.1 Hz 偏紧 |

读法：259 ms 是「画出接下来 50 步要多久」，不是「每一步 259 ms」。摊到每步大约 259/50 ≈ 5 ms，但手臂在这 259 ms 里用的是 **上一块已经在篮子里的步**，看到的世界也是 259 ms 前的。

GR00T 的 **8** 只是执行窗：一次 GPU 仍预测 16 步，RTC 大约每 8 拍重叠/换新前缀。不要把 8 写成「SmolVLA 也是 8 步一块」。

## RTC 怎么理解（FAQ）

导航比喻：ACT 是每个路口立刻重算；VLA+RTC 是一次画出接下来若干路口，开车同时后台算下一份。车会不会停，看篮子空不空；眼睛新不新，看画一份要多久。

### 「RTC 延迟大」是不是比错了？

是。表里 RTC 的几百毫秒是 **`predict_action_chunk`（画一整块）**，ACT 的十几毫秒是 **一次前向（往往就是下一步）**。不是同一种活，也不是 RTC 额外收的税。公平比法：各看自己那列的 **重规划延迟**。RTC 的价值是 `chunk_Hz × 执行窗 ≥ 控制 Hz`，队列不饿死，**不是**把 465 ms 变成 11 ms。

### ~250 ms 视觉滞后是什么？对执行有什么影响？

以 SmolVLA 8060S 为例：算一块约 259 ms ≈ **5 拍**（20 Hz）。

1. t=0 拍一张图。
2. t=0～259 ms GPU 在算；手臂不停止，走篮子里 **上一块剩下的步**。
3. t≈259 ms 新块入队——这 50 步全是按 t=0 那张 **旧图** 规划的。

不是摄像头晚了，也不是总线堵了，而是 **规划用的眼睛是旧的**。摊到每步 259/50≈5 ms 只是计算摊销，**不是**画面有多新。

| 场景 | 影响 |
|------|------|
| 空中接近、方块完全静止 | 路略钝、略滞后，不一定失败 |
| 接触 / 闭合 / 方块被碰动 | 按旧图里「已经够近」去夹，容易空夹、夹晚、夹偏、追被蹭跑的方块 |
| 真机：人递东西、传送带、障碍伸进工作区 | 盯着约 250 ms 前的目标；急停/让路也晚这么多 |
| 真机：力控、插孔 | 接触力在这 250 ms 里已经变了，开环 chunk 容易顶死或插偏 |

真机编码器是「现在」，相机规划是「一块之前」。RTC/相对动作能减轻这种错位，但 **补不掉没看见刚发生的事**。ACT ~11 ms ≈ 晚 1 拍；SmolVLA ≈ 晚 5 拍 + 一块未完成的开环。RTC 保证的是 **20 Hz 电机指令不断档**，不是眼睛也是 20 Hz。

### 把 `n_action_steps` 改小，甚至改成 1，能消掉滞后、变成 sync 吗？

不能。滞后首先是 **生成一块要多久**。SmolVLA 前向是看图 + VLM + flow matching（`num_steps=10`），张量宽度是 **`chunk_size=50`**。只改 `n_action_steps` 往往只是少往篮子里放几步，GPU 几乎还是整块 50 步，**250 ms 还在**。

短 chunk 真正变短的是 **新图算完后还要按旧计划再走多久**（少开环）。队列不饿的下界大约是 `ceil(一块延迟 / 50 ms)`：SmolVLA 在 8060S 上大约 **≥ 6 步**。50 偏长；收到 8～16 可能更利落。**设成 1 更差**：每 250 ms 才产出 1 个动作，篮子常空，手臂顿，环掉到 ~4 Hz。

`n_action_steps=1` **≠ sync**：

- **Sync**：控制线程 **堵住** 等算完。环频 = `min(20 Hz, 1/推理时间)`。SmolVLA 会变成 ~4 Hz 真闭环：图相对新，整机也只能 4 Hz 动。
- **RTC + 1 步**：后台算、前台仍按 20 Hz 来要；经常没货。新鲜度仍受 250 ms 限制，调度比 sync 还差。

要「每拍都看新图」，需要 **推理 ≤ 50 ms**。ACT 已经满足；SmolVLA / GR00T 不满足。

### 那 RTC 要改进视觉滞后，是不是还得把 chunk 算快？

对。滞后 ≈ 一块墙钟时间。259 ms → 约 5 拍旧图；若能 50 ms，就只晚 1 拍。调执行窗 / `n_action_steps` 切不掉这 250 ms。真能动的是生成速度：更小或蒸馏模型、更少 flow 步、更快 GPU、更少视觉预处理。本表里 GR00T 8060S→MI300X 是 465→126 ms（约 9 拍 → 约 2.5 拍）；SmolVLA 两张卡都是 ~230–260 ms，所以视觉滞后几乎没动。

### RTC 的好处是动作更平滑吗？

不完全是。主收益是：**推理比一拍还慢时，仍然按 20 Hz 不停出动作**。慢模型 + sync 会在一块用完后干等 GPU（SmolVLA ~250 ms），真机上就是一顿一顿。RTC 后台接着算，前台每 50 ms 弹一步；重叠/前缀对齐还能减轻切块接缝。所以更「顺」多半是 **少了停顿和跳变**，不是关节曲线本身更优。ACT 已经 11 ms 时，sync 也能连续 20 Hz，RTC 几乎没有「更平滑」可加。代价仍是眼睛旧大约一块时间。

## 原始 Hz（来自 RESULTS.md）

| 主机 | 策略 | infer Hz | chunk Hz | control Hz | 备注 |
|------|------|---------:|---------:|-----------:|------|
| halo | act6 | 234.5 | 90.0 | 51.0 | EGL |
| halo | act15 | 234.6 | 92.5 | 51.0 | EGL |
| halo | molmo6 | 19.8 | 1.3 | 15.3 | EGL |
| halo | molmo15 | 15.1 | 1.0 | 12.8 | EGL |
| halo | jepa | 32.3 | 1.7 | 15.4 | EGL |
| halo | smolvla | 8520.5 | 3.9 | 67.4 | RTC，infer Hz 是队列弹出 |
| halo | groot | 63736.3 | 2.1 | 50.1 | RTC，infer Hz 是队列弹出 |
| dorobot | act6 | 115.5 | 79.0 | 2.0 | OSMesa，control Hz 无意义 |
| dorobot | act15 | 128.2 | 78.0 | 2.0 | OSMesa |
| dorobot | molmo6 | 21.8 | 2.3 | 1.8 | OSMesa |
| dorobot | jepa | 77.3 | 14.6 | 2.0 | OSMesa |
| dorobot | smolvla | 11035.3 | 4.4 | 2.1 | OSMesa，RTC |
| dorobot | groot | 10871.6 | 8.0 | 2.0 | OSMesa，RTC |

重规划延迟 = `1000 / chunk Hz`（sync 策略同样成立）。

## 结论

1. **ACT 在两张卡上都是 ~11 ms**：策略太小，MI300X 帮不上忙，它是唯一能每拍都重新规划的。
2. **GPU 差距只体现在 VLA 的 chunk 前向**：GR00T 465 → 126 ms（3.7×），JEPA 599 → 68 ms（8.8×），SmolVLA 几乎没差（259 → 229 ms，推测被 CPU / 预处理卡住）。
3. **GR00T 的 RTC 够不够喂**：执行窗 8、20 Hz，需要 chunk ≳ 2.5 Hz。MI300X 8 Hz 富余，8060S 2.1 Hz 差一点。这 8 只对 GR00T YAML 成立，SmolVLA 不是 8。
4. **SmolVLA** 一块 50 步、执行窗默认 10，chunk ~4 Hz 能喂满队列；~250 ms 是视觉滞后（规划用旧图），不是每步 250 ms。详见上文 FAQ。
5. 改进 RTC 视觉滞后靠 **加快 chunk 生成**，不是把 `n_action_steps` 设成 1（那既不是 sync，还会饿死队列）。RTC 保的是 20 Hz 不断档，不是更跟手的眼睛。
6. 不需要额外写 microbench：闭环计时已经把 `get_action` 和 `predict_action_chunk` 分开了。去掉 MuJoCo 只会改善 MI300X 那列被 OSMesa 拖累的 control Hz，不会改变 VLA 的 chunk 毫秒数。
