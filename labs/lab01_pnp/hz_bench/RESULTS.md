# Lab 01 headless inference Hz (auto)

Protocol: `render_window=false`, fixed cube, 4 episodes, **drop ep 1** (compile).
halo = 8060S EGL. DORobot = MI300X OSMesa (control Hz is software render; trust GPU column).
GPU column: sync `infer` / SmolVLA+GR00T `chunk` (`predict_action_chunk`). Not RTC queue pops.
Sweep entry: `labs/lab01_pnp/hz_bench.cmd`. Logs: `outputs/eval/hz_bench/` (not committed).

| Host | Policy | GPU Hz (metric) | mean ms | infer Hz | chunk Hz | control Hz | status |
|------|--------|-----------------|--------:|---------:|---------:|-----------:|--------|
| halo | act6 | **234.5** (infer) | 4 | 234.5 | 90.0 | 51.0 | ok |
| halo | act15 | **234.6** (infer) | 4 | 234.6 | 92.5 | 51.0 | ok |
| halo | smolvla | **3.9** (chunk) | 259 | 8520.5 | 3.9 | 67.4 | ok |
| halo | molmo6 | **19.8** (infer) | 50 | 19.8 | 1.3 | 15.3 | ok |
| halo | molmo15 | **15.1** (infer) | 66 | 15.1 | 1.0 | 12.8 | ok |
| halo | jepa | **32.3** (infer) | 31 | 32.3 | 1.7 | 15.4 | ok |
| halo | groot | **2.1** (chunk) | 465 | 63736.3 | 2.1 | 50.1 | ok |
| halo | fastwam | **—** (infer) | — | — | — | — | partial |
| dorobot | act6 | **115.5** (infer) | 9 | 115.5 | 79.0 | 2.0 | ok |
| dorobot | act15 | **128.2** (infer) | 8 | 128.2 | 78.0 | 2.0 | ok |
| dorobot | smolvla | **4.4** (chunk) | 229 | 11035.3 | 4.4 | 2.1 | ok |
| dorobot | molmo6 | **21.8** (infer) | 46 | 21.8 | 2.3 | 1.8 | ok |
| dorobot | jepa | **77.3** (infer) | 13 | 77.3 | 14.6 | 2.0 | ok |
| dorobot | groot | **8.0** (chunk) | 126 | 10871.6 | 8.0 | 2.0 | ok |
| dorobot | fastwam | **—** (infer) | — | — | — | — | partial |

## Skips

```
SKIP molmo15 missing /mnt/doscratch/outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model

```

Sweep complete: halo=no, dorobot=no.
