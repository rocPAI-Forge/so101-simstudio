# Lab 01 — Training benchmark (teleop + agent-yaw v4)

Living table of **measured / launched** train jobs. Eval success belongs in the runbook / `COMPARE.md`, not here.

Each row is **one** training job under exactly one of these modes:

| Mode (`parallelism`) | Rule | What it measures |
|----------------------|------|------------------|
| `single-GPU` | `num_gpus` **must be 1** | Single-card train performance |
| `DDP` | `num_gpus` **≥ 2** | Multi-GPU data parallel (full model replica per GPU) |
| `FSDP` | `num_gpus` **≥ 2** | Multi-GPU sharded train (params/grads/optimizer split) |

Do **not** use `single-GPU` with `num_gpus>1`. Do **not** fold N independent single-card jobs into one row — list them separately as `single-GPU` / `num_gpus=1`, or omit them.

**Columns**

| Column | Meaning |
|--------|---------|
| `batch_size` | Per-process `--batch_size` |
| `effective_batch_size` | `batch_size × num_gpus` (omit or same as `batch_size` for `single-GPU`) |
| `precision` | Weight / autocast recipe actually used |
| `parallelism` | `single-GPU` / `DDP` / `FSDP` |
| `num_gpus` | GPUs in **this** job (1 for `single-GPU`) |
| `num_workers` | DataLoader `--num_workers` |
| `steps` | Compact form OK (`50K` = 50000, `7.5K` = 7500) |
| `status` | `done` / `running` / `aborted` |

Sources: Lab 01 runbook + Hub cards (teleop); session logs / `train_config.json` (v4). Update rows when a job finishes or recipe changes.

---

## Datasets

| Alias | Hub / local id | Episodes / frames | Notes |
|-------|----------------|-------------------|-------|
| **teleop** | `alexhegit/so101-simstudio-lab01-pnp` | 50 episodes | Leader teleop reference |
| **agent-yaw-v4** | `alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4` | 140 episodes / 67235 frames | Canonical v4 (v3 base + 20 DAgger) |

---

## Teleop — single-GPU reference (docs / Hub)

Host **DORobot** = 1× MI300X (192 GB) unless noted. ACT/SmolVLA: **fp32** (`use_amp=false`); MolmoAct2: **`model_dtype=bfloat16`**. All rows: `parallelism=single-GPU`, `num_gpus=1`.

| Model | Dataset | GPU | precision | batch_size | num_workers | steps | status | Output / Hub |
|-------|---------|-----|-----------|-----------:|------------:|------:|--------|--------------|
| ACT 15-D | teleop | MI300X | fp32 | 128 | 4 | 50K | done | `lab01_pnp_act` → […-pnp-act](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act); loss 0.053 |
| ACT 6-D | teleop | MI300X | fp32 | 128 | 8† | 50K | done | `lab01_pnp_act_state6` → […-act-state6](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-state6); loss 0.054 |
| SmolVLA | teleop | MI300X | fp32 | 64 | 8 | 50K | done | `lab01_pnp_smolvla` → […-pnp-smolvla](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-smolvla); loss 0.018; ~26 GB |
| SmolVLA (short) | teleop | 8060S | fp32 | 4 | 4 | 7.5K | done | historical laptop; not eval default |
| MolmoAct2 15-D | teleop | MI300X | bf16 | 32 | 8 | 10K | done | `lab01_pnp_molmoact2` → […-molmoact2](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-molmoact2); LoRA VLM; ~30 GB; ~13.7 h |
| MolmoAct2 6-D | teleop | MI300X | bf16 | 32 | 8 | 10K | done | `lab01_pnp_molmoact2_state6` |

† ACT 6-D `num_workers` from launch recipe; confirm in that run’s `train_config.json` if re-citing.

**Not listed here (teleop, other policies):** GR00T N1.7 10K, VLA-JEPA keeper, etc. — see runbook §6.4 / Hub cards.

---

## Agent-yaw v4

### Single-GPU

| Model | Dataset | Host | GPU | precision | batch_size | num_workers | steps | status | Notes |
|-------|---------|------|-----|-----------|-----------:|------------:|------:|--------|-------|
| ACT 15-D | v4 | DORobot | MI300X | fp32 | 64 | 8 | 80K | done | `lab01_pnp_act_agent_yaw_v4`; Hub […-act-agent-yaw-v4](https://huggingface.co/alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4); checkpoint every 20K; home n=20 → 17/20 @ 80K ([COMPARE](../../.tmp/yaw-v4-overnight/COMPARE.md)) |
| SmolVLA | v4 | DORobot | MI300X | fp32 | 64 | 8 | 50K | running‡ | `lab01_pnp_smolvla_agent_yaw_v4`; log `…_50k_mi300x.log`; same rename_map as teleop SmolVLA |

### Multi-GPU (DDP / FSDP)

> **Note:** These multi-GPU rows are **functional smoke / bring-up** recipes (prove DDP/FSDP runs to completion without OOM). Per-GPU `batch_size=1` was chosen for stability on MI210, not for peak throughput. They are **not** claimed best-performance configs — for speed, compare single-GPU MI210 (`batch_size` up to 4 in hardware smoke) or scale batch after a dedicated multi-GPU tuning pass.

| Model | Dataset | Host | GPU | num_gpus | parallelism | precision | batch_size | effective_batch_size | num_workers | steps | status | Notes |
|-------|---------|------|-----|---------:|-------------|-----------|-----------:|---------------------:|------------:|------:|--------|-------|
| MolmoAct2 | v4 | acc | MI210 | 2 | DDP | bf16 + accelerate bf16 | 1 | 2 | 8 | 10K | aborted | `…_2xmi210_10k`; paused ~1.4K when switching to FSDP |
| MolmoAct2 | v4 | acc | MI210 | 4 | FSDP FULL_SHARD | **fp32** weights + accelerate **bf16** | 1 | 4 | 8 | 10K | running | `lab01_pnp_molmoact2_agent_yaw_v4_4xmi210_fsdp`; wrap `MolmoAct2DecoderLayer,MolmoAct2VisionBlock,ActionExpertBlock`; launch via `cuda:LOCAL_RANK` remap (`.tmp/molmo_hwsmoke/fsdp_rank_entry.py`); ~2.2 s/step |

‡ Re-check DORobot log before quoting final step/loss.

### MolmoAct2 v4 — hardware notes (acc)

| Attempt | Result |
|---------|--------|
| DDP 2×/4× A30 | OOM (24 GB; model ~17 GB + DDP peak) |
| DDP 4× MI210 | OOM (placement / all-rank load on one GPU) |
| DDP 2× MI210 | OK smoke; full train started then paused |
| FSDP 4× MI210 | OK with `model_dtype=float32` + per-rank `cuda:LOCAL_RANK` (bf16 weights hit FSDP flatten dtype error) |
| FSDP 4× A30 | **Not viable** with current fp32 FSDP recipe (~40+ GB resident/GPU on MI210) |

---

## How to extend

1. Pick a mode first: **single-GPU** (`num_gpus=1`) **or** multi-GPU **DDP/FSDP** (`num_gpus≥2`). Never mix.
2. One row per launched job `(model × dataset × GPU type × parallelism × num_gpus)`.
3. Prefer `train_config.json` for `batch_size` / `steps` / `num_workers`.
4. Multi-GPU rows must include `effective_batch_size`.
5. Keep aborted / OOM notes so we do not re-discover dead ends.
6. Link eval tables elsewhere; optional one-line success pointer only for published Hub refs.
