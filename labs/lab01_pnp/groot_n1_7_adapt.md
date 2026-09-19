# Lab 01 — GR00T N1.7-3B (joint VLA, not N1.5)

**Status (2026-09-19):** **Keep the 10K checkpoint only.** MI300X train 10K (bs4, ~35 min) is the Lab 01 reference. Closed-loop is **0/10** with working approach (`reach_8cm≈9–10/10`) and no lift. Continuation 30K and visual-unfreeze 20K are **worse** — do not use for eval or Hub. Classroom demo remains ACT 6-D / MolmoAct2 6-D.

**Keeper path:** `outputs/train/lab01_pnp_groot_n17/checkpoints/010000/pretrained_model`  
(DORobot copy: `/mnt/doscratch/outputs/train/lab01_pnp_groot_n17/checkpoints/010000/pretrained_model`)

## Measured comparison (same 50-ep joint set)

Relative actions require RTC (`execution_horizon=8`). Log figures of 10k+ Hz are RTC queue pops; **chunk Hz** is `predict_action_chunk`.

| Run | Train | Keep? | Full-range n=10 | Approach / near / close / lift | Chunk Hz |
|-----|-------|-------|-----------------|--------------------------------|----------|
| **10K from NVIDIA N1.7** | Frozen LLM+vis, cosine → lr=0, loss ~0.03–0.1 | **Yes** | **0/10** | EGL 9/10 → 7/10 → 5/10 → **0/10**; OSMesa 10/10 → 6/10 → 1/10 → 0/10 | **2.53** (8060S EGL); OSMesa control ~2 Hz is render |
| 10K same, fixed cube | — | Yes (same weights) | **0/10** | 9/10 → 6/10 → 2/10 → 0/10 | 2.51 (8060S) |
| 10K → **cont 30K** | Fresh optimizer, 10K as `base_model_path` | **No** | **0/10** | 2/10 → 0 → 0 → 0 | ~5.8 (MI300X OSMesa) |
| cont30K → **vis 20K** | `tune_visual=true` | **No** | **0/10** | 1/10 → 0 → 0 → 0 (fixed: **0/10** reach) | ~5.4–5.8 |

**Do not keep / do not resume from:**  
`/mnt/doscratch/outputs/train/lab01_pnp_groot_n17_cont30k/`  
`/mnt/doscratch/outputs/train/lab01_pnp_groot_n17_vis20k/`

### Conclusions

1. **Decode is not garbage.** 10K reaches the cube on almost every episode — 6-D `*.pos` + `new_embodiment` works.
2. **Failure is grasp/lift on 50 expert episodes**, not inference FPS. Halo EGL keeps a 20 Hz control loop (compute ~57 Hz) while chunk is ~2.5 Hz via RTC.
3. **More steps on this set, with the 10K-as-base resume, hurt.** Cosine had already reached lr=0; reopening the optimizer left train loss ~1.1 and collapsed approach. Continuation checkpoints also stored `num_inference_timesteps: null` (eval must patch to 4).
4. **Unfreezing vision on 50 episodes hurt** (same family as JEPA unfreeze-Qwen): fixed-cube `reach_8cm` went 9/10 → 0/10.
5. **To raise k/n:** collect more successful grasp→lift→place demos, then fine-tune from NVIDIA N1.7 (or 10K) with a **live** lr schedule. Do not stack vis-unfreeze or 30K-style continuation first.

### Remote inference (prototype, not a Lab 01 feature)

Split eval — GLFW on halo, GR00T on DORobot via LeRobot async `policy_server` — was a **WAN experiment**, not shipped edge-cloud inference. Link ~**300 ms RTT** / ~15 Mbit/s; control fell to **5–12 Hz**. Local RTC on the same 10K weights is the reference (control ~20 Hz). Do not quote WAN rollouts in §6.4.

Keep the measurement; do not add this path to QUICKSTART. Overnight `campaign_groot_*.cmd` scripts are disposable (10K already trained). Follow-up / revival gate (LAN, ≥20 Hz): [issue #10](https://github.com/rocPAI-Forge/so101-simstudio/issues/10).

## Goal

Fine-tune [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) (`policy.type=groot`) on the **existing** Lab 01 50-episode joint dataset and eval in MuJoCo with the same `*.pos` contract as ACT 6-D.

Success for this workstream:

1. Train and eval stay on LeRobot **v0.6.x** already pinned in this repo (N1.7 only).
2. Closed-loop actions are SO-101 `shoulder_pan.pos` … `gripper.pos` (6-D). Padding to GR00T `max_action_dim` (132) is the official embodiment mechanism, **not** FastWAM-style stuffing into a 7-DoF joint slot or LingBot EEF channels.
3. §6.4 reports `k/n` with config path + checkpoint step. Do not quote NVIDIA LIBERO / SimplerEnv numbers as Lab 01 success.

## Why N1.7 here, not N1.5

| Checkpoint | Lab 01 fit |
|------------|------------|
| [`nvidia/GR00T-N1.5-3B`](https://huggingface.co/nvidia/GR00T-N1.5-3B) | **Out of scope.** LeRobot in this repo **removed** N1.5 (`GROOT_N1_5_REMOVAL_GUIDANCE`). Loading it requires `lerobot==0.5.1`. Card is non-commercial. Do not fork the submodule for this experiment. |
| [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | **In scope.** `groot` policy + `embodiment_tag=new_embodiment` is the documented SO-101 post-train path (`lerobot/docs/source/groot.mdx`). NVIDIA Open Model License. |
| `nvidia/GR00T-N1.7-LIBERO` / `embodiment_tag=libero_sim` | **Do not** use as Lab 01 init. That is a LIBERO embodiment, same failure family as JEPA/FastWAM if you pretend it is SO-101. |

Same-dataset family (runbook §6.4.2): GR00T N1.7 is **joint-space**, like ACT / SmolVLA / Molmo — closer to those than to LingBot-VA. 50 episodes may still underfit (SmolVLA ~22%); treat as a VLA contrast, not a classroom replacement.

## Frozen (do not do)

- `uv pip install 'lerobot[groot]'` or `pip install -e '.[groot]'` inside `.venv-rocm` (pulls CUDA torch).
- `--policy.base_model_path=nvidia/GR00T-N1.5-3B`
- `--policy.embodiment_tag=libero_sim` on Lab 01 joints
- Renaming cameras to LIBERO `image` / `image2` “because other VLAs did”
- 15-D state (pos+vel+ee) as the default GR00T recipe — keep **6-D joint pos**, `LAB01_STATE_DIM=6`
- Unfreezing the full 3B backbone as the first lever if open-loop is dead (JEPA unfreeze lesson)
- Blocking train on gated [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) when the public Qwen3-VL-2B-Instruct tokenizer files are enough for the N1.7 processor (override with `LAB01_GROOT_PROCESSOR_MODEL`)

## Delivery steps

### 1. ROCm extras

Add `scripts/install-groot-deps.sh` modeled on `install-molmoact2-deps.sh`: pin current ROCm torch via `--constraints`, install the `groot` extra packages from `lerobot/pyproject.toml` (`transformers`, `peft`, `diffusers`, `dm-tree`, `timm`, `decord` on x86_64 — skip if the wheel is CUDA-only and document the gap). Assert `torch.version.hip` and `+rocm` in the version string. `--repair-torch` if CUDA leaked.

Smoke: `python -c "from lerobot.policies.groot.configuration_groot import GrootConfig"`.

### 2. Train wrapper

New `labs/lab01_pnp/train_groot.cmd` + `_env.sh` section **5e**. Drive `python -m simstudio.scripts.train_act` (or `lerobot-train` after explicit groot config import) with:

| Knob | Lab 01 default (implementation) |
|------|----------------------------------|
| `LAB01_GROOT_BASE` | `nvidia/GR00T-N1.7-3B` |
| `LAB01_GROOT_EMBODIMENT_TAG` | `new_embodiment` |
| `LAB01_GROOT_USE_RELATIVE_ACTIONS` | `true` |
| `LAB01_GROOT_RELATIVE_EXCLUDE_JOINTS` | `["gripper"]` |
| `LAB01_STATE_DIM` | `6` |
| `LAB01_GROOT_CHUNK_SIZE` / `N_ACTION_STEPS` | start `16` (official SO-101 example); confirm vs checkpoint native horizon 40 |
| `LAB01_GROOT_USE_BF16` | `true` on MI300X |
| Dataset | `LAB01_DATASET_ROOT` unchanged (joint parquet) |

Cameras: keep Lab 01 keys (`camera_front` / `camera_top` / `camera_wrist`) unless the groot processor **requires** a rename; GR00T accepts a variable number of views. Do not drop `front` only to mimic LIBERO.

Short smoke: `LAB01_GROOT_STEPS=20 LAB01_GROOT_BATCH_SIZE=1`. Reference train: document MI300X batch/steps as env overrides (official example is bs64 × 20K — likely too heavy; pick a VRAM-safe batch and write wall-clock in the runbook).

### 3. Eval

`labs/lab01_pnp/configs/rollout_groot.yaml` (full-range) and optional `rollout_groot_demo_fixed.yaml`. `action_mode: position`. No IK. `eval.cmd` + `LAB01_POLICY_PATH`.

Relative actions **cannot** use `inference.type=sync` (`GrootPolicy.select_action` raises). Lab 01 `rollout_groot.yaml` uses the official SO-101 RTC path (`execution_horizon=8`, `queue_threshold=0`). Set `inference.rtc.enabled=false` only if RTC itself is unstable; you still need `type=rtc` to run `predict_action_chunk`.

### 4. Measure

| Gate | Pass |
|------|------|
| Smoke closed-loop | `reach_8cm` on ≥1 / few episodes (proves 6-D decode, not pad garbage) |
| Open-loop | RMSE vs ACT 6-D on the same episodes; if ~JEPA/FastWAM (≫ ACT) stop before a 20K run |
| n=10 fixed cube | Report k/10 + stage funnel |
| n=20 or n=50 full-range | Same spawn box as other VLAs |

Hub card + `LAB01_GROOT_HF_REPO_ID` (`alexhegit/so101-simstudio-lab01-pnp-groot-n17`) documents this **negative** 10K result (NVIDIA Open Model License). Do not Hub the 30K / vis-20K runs. Follow-up: [issue #6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6).

## Out of scope

- GR00T N1.5, Isaac-GR00T CUDA/TensorRT-only train path
- New teleop / OpenArm dataset (separate lab)
- LingBot EEF adapter (other PR)
- Replacing ACT as the classroom policy based on train loss alone
- Halo↔DORobot **WAN** async as a product (prototype notes: [issue #10](https://github.com/rocPAI-Forge/so101-simstudio/issues/10))

## Suggested file touch list (implementation)

| Piece | Path |
|-------|------|
| ROCm install | `scripts/install-groot-deps.sh` |
| Env | `_env.sh` §5e (`LAB01_GROOT_*`) |
| Train | `train_groot.cmd` |
| Eval YAML | `configs/rollout_groot.yaml` (+ optional fixed-pose) |
| Runbook | § train GR00T + §6.4 row |
| Hub | `push_groot_model.cmd` / `hf_model_card_groot.md` when publishing |

## Knobs (wired)

```bash
# sketches
LAB01_GROOT_OUTPUT=./outputs/train/lab01_pnp_groot_n17
LAB01_GROOT_BASE=nvidia/GR00T-N1.7-3B
LAB01_GROOT_EMBODIMENT_TAG=new_embodiment
LAB01_GROOT_STEPS=20
LAB01_GROOT_BATCH_SIZE=1
# optional: nvidia/Cosmos-Reason2-2B if Hub access is granted
# LAB01_GROOT_PROCESSOR_MODEL=Qwen/Qwen3-VL-2B-Instruct
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_groot.yaml
```
