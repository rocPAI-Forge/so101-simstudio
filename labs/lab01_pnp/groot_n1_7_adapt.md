# Lab 01 — GR00T N1.7-3B (joint VLA, not N1.5)

**Status:** specification for delivery. No `train_groot.cmd` / eval YAML in this change. Classroom demo policy remains ACT 6-D / MolmoAct2 6-D until a measured §6.4 row exists.

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

Watch RTC: SmolVLA iGPU lag was material; start **sync**. If LeRobot groot rollout defaults RTC on, set it off for the first Lab 01 numbers.

### 4. Measure

| Gate | Pass |
|------|------|
| Smoke closed-loop | `reach_8cm` on ≥1 / few episodes (proves 6-D decode, not pad garbage) |
| Open-loop | RMSE vs ACT 6-D on the same episodes; if ~JEPA/FastWAM (≫ ACT) stop before a 20K run |
| n=10 fixed cube | Report k/10 + stage funnel |
| n=20 or n=50 full-range | Same spawn box as other VLAs |

Hub card + `LAB01_GROOT_HF_REPO_ID` only after a fair eval. License: NVIDIA Open Model on the card.

## Out of scope

- GR00T N1.5, Isaac-GR00T CUDA/TensorRT-only train path
- New teleop / OpenArm dataset (separate lab)
- LingBot EEF adapter (other PR)
- Replacing ACT as the classroom policy based on train loss alone

## Suggested file touch list (implementation)

| Piece | Path |
|-------|------|
| ROCm install | `scripts/install-groot-deps.sh` |
| Env | `_env.sh` §5e (`LAB01_GROOT_*`) |
| Train | `train_groot.cmd` |
| Eval YAML | `configs/rollout_groot.yaml` (+ optional fixed-pose) |
| Runbook | § train GR00T + §6.4 row |
| Hub | `push_groot_model.cmd` / `hf_model_card_groot.md` when publishing |

## Knobs (not wired until the implementation PR)

```bash
# sketches
LAB01_GROOT_OUTPUT=./outputs/train/lab01_pnp_groot_n17
LAB01_GROOT_BASE=nvidia/GR00T-N1.7-3B
LAB01_GROOT_EMBODIMENT_TAG=new_embodiment
LAB01_GROOT_STEPS=20
LAB01_GROOT_BATCH_SIZE=1
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_groot.yaml
```
