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
#   5) Eval     — eval.cmd (any LeRobot policy)
#   6) Hub      — push_*.cmd
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Shared — used by record / train / eval / push
# -----------------------------------------------------------------------------
LAB01_DATASET_NAME="${LAB01_DATASET_NAME:-so101-simstudio-lab01-pnp}"
LAB01_DATASET_HF_USER="${LAB01_DATASET_HF_USER:-alexhegit}"
LAB01_DATASET_REPO_ID="${LAB01_DATASET_REPO_ID:-${LAB01_DATASET_HF_USER}/${LAB01_DATASET_NAME}}"
LAB01_DATASET_ROOT="${LAB01_DATASET_ROOT:-./datasets/${LAB01_DATASET_NAME}}"

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
# -----------------------------------------------------------------------------
LAB01_ACT_OUTPUT="${LAB01_ACT_OUTPUT:-./outputs/train/lab01_pnp_act}"
LAB01_ACT_STEPS="${LAB01_ACT_STEPS:-10000}"
LAB01_ACT_BATCH_SIZE="${LAB01_ACT_BATCH_SIZE:-8}"
LAB01_ACT_SAVE_FREQ="${LAB01_ACT_SAVE_FREQ:-10000}"

# -----------------------------------------------------------------------------
# 5) Eval — eval.cmd (policy-agnostic)
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
# 6) Hub — push_dataset.cmd / push_smolvla_model.cmd / push_act_model_card.cmd
# -----------------------------------------------------------------------------
LAB01_SMOLVLA_HF_REPO_ID="${LAB01_SMOLVLA_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-smolvla}"
LAB01_ACT_HF_REPO_ID="${LAB01_ACT_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act}"
# Checkpoint step name under LAB01_TRAIN_OUTPUT when push_smolvla_model does not
# get an explicit LAB01_POLICY_PATH.
LAB01_SMOLVLA_CKPT_STEP="${LAB01_SMOLVLA_CKPT_STEP:-050000}"
# Optional: force official Hub instead of mirror, e.g. https://huggingface.co
# LAB01_HF_UPLOAD_ENDPOINT=
