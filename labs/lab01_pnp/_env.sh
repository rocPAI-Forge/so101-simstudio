#!/bin/bash
# =============================================================================
# Lab 01 — shared environment defaults
# =============================================================================
# Sourced by labs/lab01_pnp/*.cmd after scripts/quicktest/_common.sh.
# All Lab 01 knobs live here (centralized). Scripts only source this file and run.
#
# Override any value from the shell, e.g.:
#   LAB01_DATASET_NAME=my-run ./labs/lab01_pnp/record.cmd
#   LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_STEPS=50000 ./labs/lab01_pnp/train.cmd
#   LAB01_POLICY_PATH=... LAB01_EVAL_CONFIG=... ./labs/lab01_pnp/eval.cmd
#
# Sections:
#   1) Shared   — lab identity / dataset (record, train, eval, push)
#   2) Record   — record.cmd only
#   3) Train    — train.cmd / train_smolvla_resume.cmd (SmolVLA)
#   4) Train ACT — train_act.cmd
#   5) Train MolmoAct2 — train_molmoact2.cmd (DORobot / MI300X)
#   5b) Train VLA-JEPA — train_vla_jepa.cmd (DORobot / MI300X)
#   6) Eval     — eval.cmd (any LeRobot policy)
#   7) Hub      — push_*.cmd
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Shared — used by record / train / eval / push
# -----------------------------------------------------------------------------
LAB01_DATASET_NAME="${LAB01_DATASET_NAME:-so101-simstudio-lab01-pnp}"
LAB01_DATASET_HF_USER="${LAB01_DATASET_HF_USER:-alexhegit}"
LAB01_DATASET_REPO_ID="${LAB01_DATASET_REPO_ID:-${LAB01_DATASET_HF_USER}/${LAB01_DATASET_NAME}}"
LAB01_DATASET_ROOT="${LAB01_DATASET_ROOT:-./datasets/${LAB01_DATASET_NAME}}"
# Hugging Face `hf download --local-dir ./datasets/<name>` may nest `<name>/<name>/`.
# Also tolerate a stale LAB01_DATASET_ROOT that points at a missing nested path.
if [[ ! -d "${LAB01_DATASET_ROOT}/meta" ]]; then
  for _lab01_ds_cand in \
    "./datasets/${LAB01_DATASET_NAME}" \
    "./datasets/${LAB01_DATASET_NAME}/${LAB01_DATASET_NAME}"
  do
    if [[ -d "${_lab01_ds_cand}/meta" ]]; then
      LAB01_DATASET_ROOT="${_lab01_ds_cand}"
      break
    fi
  done
  unset _lab01_ds_cand
fi

# SmolVLA camera rename (dataset keys → smolvla_base). Used by train.cmd.
LAB01_RENAME_MAP='{"observation.images.camera_top":"observation.images.camera1","observation.images.camera_front":"observation.images.camera2","observation.images.camera_wrist":"observation.images.camera3"}'

# -----------------------------------------------------------------------------
# 2) Record — record.cmd
# -----------------------------------------------------------------------------
LAB01_RECORD_CONFIG="${LAB01_RECORD_CONFIG:-configs/so101_mujoco_pick_leader.yaml}"
LAB01_NUM_EPISODES="${LAB01_NUM_EPISODES:-50}"
LAB01_EPISODE_TIME_S="${LAB01_EPISODE_TIME_S:-90}"
LAB01_RESET_TIME_S="${LAB01_RESET_TIME_S:-5}"

# -----------------------------------------------------------------------------
# 3) Train (SmolVLA) — train.cmd / train_smolvla_resume.cmd
#    Set batch/steps/workers for your GPU VRAM (see lab01_pnp.md §5 recipes).
# -----------------------------------------------------------------------------
LAB01_TRAIN_OUTPUT="${LAB01_TRAIN_OUTPUT:-./outputs/train/lab01_pnp_smolvla}"
LAB01_TRAIN_WARMUP="${LAB01_TRAIN_WARMUP:-500}"
LAB01_TRAIN_STEPS="${LAB01_TRAIN_STEPS:-7500}"
LAB01_TRAIN_BATCH_SIZE="${LAB01_TRAIN_BATCH_SIZE:-4}"
LAB01_TRAIN_SAVE_FREQ="${LAB01_TRAIN_SAVE_FREQ:-2000}"
LAB01_TRAIN_NUM_WORKERS="${LAB01_TRAIN_NUM_WORKERS:-4}"
LAB01_TRAIN_RESUME_FROM="${LAB01_TRAIN_RESUME_FROM:-007500}"
LAB01_TRAIN_RESUME_STEPS="${LAB01_TRAIN_RESUME_STEPS:-20000}"
# Optional log path for resume; default is derived in train_smolvla_resume.cmd.
# LAB01_TRAIN_RESUME_LOG=

# -----------------------------------------------------------------------------
# 4) Train (ACT) — train_act.cmd
#    LAB01_ACT_STATE_DIM=6 → joint positions only (dataset stays 15-D on disk).
#    Empty / unset → full 15-D (pos+vel+ee). Use a distinct OUTPUT dir for 6-D.
# -----------------------------------------------------------------------------
LAB01_ACT_STATE_DIM="${LAB01_ACT_STATE_DIM:-}"
if [[ "${LAB01_ACT_STATE_DIM}" == "6" ]]; then
  _lab01_act_out_default="./outputs/train/lab01_pnp_act_state6"
  _lab01_act_log_default="train_act_state6.log"
  _lab01_act_job_default="lab01_pnp_act_state6"
else
  _lab01_act_out_default="./outputs/train/lab01_pnp_act"
  _lab01_act_log_default="train_act.log"
  _lab01_act_job_default="lab01_pnp_act"
fi
LAB01_ACT_OUTPUT="${LAB01_ACT_OUTPUT:-${_lab01_act_out_default}}"
LAB01_ACT_LOG="${LAB01_ACT_LOG:-${_lab01_act_log_default}}"
LAB01_ACT_JOB_NAME="${LAB01_ACT_JOB_NAME:-${_lab01_act_job_default}}"
unset _lab01_act_out_default _lab01_act_log_default _lab01_act_job_default
LAB01_ACT_STEPS="${LAB01_ACT_STEPS:-10000}"
LAB01_ACT_BATCH_SIZE="${LAB01_ACT_BATCH_SIZE:-8}"
LAB01_ACT_SAVE_FREQ="${LAB01_ACT_SAVE_FREQ:-10000}"

# -----------------------------------------------------------------------------
# 5) Train (MolmoAct2) — train_molmoact2.cmd
#    Defaults target single-GPU Instinct MI300X (192 GB HBM), same class as
#    DORobot. Smaller GPUs: lower LAB01_MOLMO_BATCH_SIZE (try 4–8).
#    Base: lerobot/MolmoAct2-SO100_101-LeRobot (2 cams → rename top+wrist).
#    Extras: ./scripts/install-molmoact2-deps.sh  (not uv pip install 'lerobot[molmoact2]')
# -----------------------------------------------------------------------------
LAB01_MOLMO_OUTPUT="${LAB01_MOLMO_OUTPUT:-./outputs/train/lab01_pnp_molmoact2}"
LAB01_MOLMO_BASE="${LAB01_MOLMO_BASE:-lerobot/MolmoAct2-SO100_101-LeRobot}"
LAB01_MOLMO_STEPS="${LAB01_MOLMO_STEPS:-10000}"
LAB01_MOLMO_BATCH_SIZE="${LAB01_MOLMO_BATCH_SIZE:-32}"
LAB01_MOLMO_SAVE_FREQ="${LAB01_MOLMO_SAVE_FREQ:-2000}"
LAB01_MOLMO_NUM_WORKERS="${LAB01_MOLMO_NUM_WORKERS:-8}"
LAB01_MOLMO_CHUNK_SIZE="${LAB01_MOLMO_CHUNK_SIZE:-30}"
LAB01_MOLMO_N_ACTION_STEPS="${LAB01_MOLMO_N_ACTION_STEPS:-30}"
LAB01_MOLMO_NUM_FLOW_TIMESTEPS="${LAB01_MOLMO_NUM_FLOW_TIMESTEPS:-8}"
# VLM LoRA + fully trainable action expert (small Lab01 set). Set false to FT all.
LAB01_MOLMO_ENABLE_LORA_VLM="${LAB01_MOLMO_ENABLE_LORA_VLM:-true}"
# Dataset keys → SO100_101-LeRobot cam0/cam1 (2-view warm start).
# Dataset keys → SO100_101-LeRobot cam0/cam1 (2-view warm start).
# Override before sourcing if needed: export LAB01_MOLMO_RENAME_MAP='{...}'
if [[ -z "${LAB01_MOLMO_RENAME_MAP:-}" ]]; then
  LAB01_MOLMO_RENAME_MAP='{"observation.images.camera_top":"observation.images.cam0","observation.images.camera_wrist":"observation.images.cam1"}'
fi
# Lab01 joints are radians; SO100_101 joint_offsets are degree-frame. Disable
# that transform for this fine-tune (identity signs/offsets).
if [[ -z "${LAB01_MOLMO_JOINT_SIGNS:-}" ]]; then
  LAB01_MOLMO_JOINT_SIGNS='[1.0,1.0,1.0,1.0,1.0,1.0]'
fi
if [[ -z "${LAB01_MOLMO_JOINT_OFFSETS:-}" ]]; then
  LAB01_MOLMO_JOINT_OFFSETS='[0.0,0.0,0.0,0.0,0.0,0.0]'
fi
# Run quantile stats augment once before train if meta/stats lack q01/q99.
LAB01_MOLMO_AUGMENT_QUANTILE_STATS="${LAB01_MOLMO_AUGMENT_QUANTILE_STATS:-true}"

# -----------------------------------------------------------------------------
# 5b) Train (VLA-JEPA) — train_vla_jepa.cmd
#    Base: lerobot/VLA-JEPA-LIBERO (2 cams, 7DoF). Lab01 is 6D + 3 cams:
#    map top/wrist → image/image2; reinit action/state heads; disable LIBERO
#    gripper snap (continuous SO-101 gripper is dim 5 of 6).
# -----------------------------------------------------------------------------
LAB01_JEPA_OUTPUT="${LAB01_JEPA_OUTPUT:-./outputs/train/lab01_pnp_vla_jepa}"
LAB01_JEPA_BASE="${LAB01_JEPA_BASE:-lerobot/VLA-JEPA-LIBERO}"
LAB01_JEPA_STEPS="${LAB01_JEPA_STEPS:-10000}"
LAB01_JEPA_BATCH_SIZE="${LAB01_JEPA_BATCH_SIZE:-16}"
LAB01_JEPA_SAVE_FREQ="${LAB01_JEPA_SAVE_FREQ:-2000}"
LAB01_JEPA_NUM_WORKERS="${LAB01_JEPA_NUM_WORKERS:-8}"
LAB01_JEPA_CHUNK_SIZE="${LAB01_JEPA_CHUNK_SIZE:-7}"
LAB01_JEPA_N_ACTION_STEPS="${LAB01_JEPA_N_ACTION_STEPS:-7}"
# Keep world-model co-training (the point of VLA-JEPA). Set false if VRAM/OOM.
LAB01_JEPA_ENABLE_WORLD_MODEL="${LAB01_JEPA_ENABLE_WORLD_MODEL:-true}"
LAB01_JEPA_FREEZE_QWEN="${LAB01_JEPA_FREEZE_QWEN:-false}"
if [[ -z "${LAB01_JEPA_RENAME_MAP:-}" ]]; then
  LAB01_JEPA_RENAME_MAP='{"observation.images.camera_top":"observation.images.image","observation.images.camera_wrist":"observation.images.image2"}'
fi
if [[ -z "${LAB01_JEPA_REINIT_MODULES:-}" ]]; then
  LAB01_JEPA_REINIT_MODULES='["model.action_model.action_encoder","model.action_model.state_encoder","model.action_model.action_decoder"]'
fi

# -----------------------------------------------------------------------------
# 6) Eval — eval.cmd (policy-agnostic)
#    Point POLICY_PATH at any pretrained_model/; match EVAL_CONFIG under
#    labs/lab01_pnp/configs/ (SmolVLA needs rename_map yaml; ACT uses rollout_act*).
# -----------------------------------------------------------------------------
LAB01_MUJOCO_GL="${LAB01_MUJOCO_GL:-egl}"                 # egl | glfw | osmesa
LAB01_RENDER_WINDOW="${LAB01_RENDER_WINDOW:-false}"     # true with glfw
LAB01_POLICY_PATH="${LAB01_POLICY_PATH:-./outputs/train/lab01_pnp_smolvla/checkpoints/050000/pretrained_model}"
LAB01_EVAL_CONFIG="${LAB01_EVAL_CONFIG:-labs/lab01_pnp/configs/rollout_smolvla_demo.yaml}"
LAB01_EVAL_EPISODES="${LAB01_EVAL_EPISODES:-10}"
LAB01_EVAL_LOG="${LAB01_EVAL_LOG:-./outputs/eval/eval_${LAB01_MUJOCO_GL}.log}"
# Empty = do not pass --policy.n_action_steps (use checkpoint default).
# ACT full-range reference often uses 50; fixed-pose demos sometimes 100.
LAB01_N_ACTION_STEPS="${LAB01_N_ACTION_STEPS:-}"

# -----------------------------------------------------------------------------
# 7) Hub — push_dataset.cmd / push_smolvla_model.cmd / push_act_model_card.cmd /
#    push_act_state6_model.cmd
# -----------------------------------------------------------------------------
LAB01_SMOLVLA_HF_REPO_ID="${LAB01_SMOLVLA_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-smolvla}"
LAB01_ACT_HF_REPO_ID="${LAB01_ACT_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act}"
LAB01_ACT_STATE6_HF_REPO_ID="${LAB01_ACT_STATE6_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act-state6}"
LAB01_ACT_STATE6_CKPT_STEP="${LAB01_ACT_STATE6_CKPT_STEP:-050000}"
LAB01_MOLMO_HF_REPO_ID="${LAB01_MOLMO_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-molmoact2}"
LAB01_MOLMO_CKPT_STEP="${LAB01_MOLMO_CKPT_STEP:-010000}"
LAB01_JEPA_HF_REPO_ID="${LAB01_JEPA_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-vla-jepa}"
LAB01_JEPA_CKPT_STEP="${LAB01_JEPA_CKPT_STEP:-020000}"
# Checkpoint step name under LAB01_TRAIN_OUTPUT when push_smolvla_model does not
# get an explicit LAB01_POLICY_PATH.
LAB01_SMOLVLA_CKPT_STEP="${LAB01_SMOLVLA_CKPT_STEP:-050000}"
# Optional: force official Hub instead of mirror, e.g. https://huggingface.co
# LAB01_HF_UPLOAD_ENDPOINT=
