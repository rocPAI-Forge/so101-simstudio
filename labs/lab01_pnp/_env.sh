#!/bin/bash
# Lab 01 shared defaults — sourced by labs/lab01_pnp/*.cmd after scripts/quicktest/_common.sh.
#
# Override any value via environment, e.g.:
#   LAB01_DATASET_NAME=my-run ./labs/lab01_pnp/record.cmd
#   export LAB01_NUM_EPISODES=10 && ./labs/lab01_pnp/record.cmd

# Dataset (local folder name + HuggingFace repo_id)
LAB01_DATASET_NAME="${LAB01_DATASET_NAME:-so101-simstudio-lab01-pnp}"
LAB01_DATASET_HF_USER="${LAB01_DATASET_HF_USER:-alexhegit}"
LAB01_DATASET_REPO_ID="${LAB01_DATASET_REPO_ID:-${LAB01_DATASET_HF_USER}/${LAB01_DATASET_NAME}}"
LAB01_DATASET_ROOT="${LAB01_DATASET_ROOT:-./datasets/${LAB01_DATASET_NAME}}"

# Recording
LAB01_NUM_EPISODES="${LAB01_NUM_EPISODES:-50}"
LAB01_EPISODE_TIME_S="${LAB01_EPISODE_TIME_S:-90}"
LAB01_RESET_TIME_S="${LAB01_RESET_TIME_S:-5}"
LAB01_RECORD_CONFIG="${LAB01_RECORD_CONFIG:-configs/so101_mujoco_pick_leader.yaml}"

# SmolVLA training
LAB01_TRAIN_OUTPUT="${LAB01_TRAIN_OUTPUT:-./outputs/train/lab01_pnp_smolvla}"
LAB01_TRAIN_STEPS="${LAB01_TRAIN_STEPS:-7500}"
LAB01_TRAIN_BATCH_SIZE="${LAB01_TRAIN_BATCH_SIZE:-4}"
LAB01_TRAIN_WARMUP="${LAB01_TRAIN_WARMUP:-500}"
LAB01_TRAIN_SAVE_FREQ="${LAB01_TRAIN_SAVE_FREQ:-2000}"
LAB01_RENAME_MAP='{"observation.images.camera_top":"observation.images.camera1","observation.images.camera_front":"observation.images.camera2","observation.images.camera_wrist":"observation.images.camera3"}'

# SmolVLA resume (train_smolvla_resume.cmd)
LAB01_TRAIN_RESUME_FROM="${LAB01_TRAIN_RESUME_FROM:-007500}"
LAB01_TRAIN_RESUME_STEPS="${LAB01_TRAIN_RESUME_STEPS:-20000}"

# ACT training / Hub
LAB01_ACT_HF_REPO_ID="${LAB01_ACT_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act}"
LAB01_ACT_OUTPUT="${LAB01_ACT_OUTPUT:-./outputs/train/lab01_pnp_act}"
LAB01_ACT_STEPS="${LAB01_ACT_STEPS:-10000}"
LAB01_ACT_BATCH_SIZE="${LAB01_ACT_BATCH_SIZE:-8}"
LAB01_ACT_SAVE_FREQ="${LAB01_ACT_SAVE_FREQ:-10000}"

# Eval
# MUJOCO_GL: egl (headless GPU), glfw (window), osmesa (CPU headless)
LAB01_MUJOCO_GL="${LAB01_MUJOCO_GL:-}"
LAB01_SMOLVLA_CKPT_STEP="${LAB01_SMOLVLA_CKPT_STEP:-050000}"
LAB01_ACT_CKPT_STEP="${LAB01_ACT_CKPT_STEP:-050000}"
LAB01_EVAL_EPISODES="${LAB01_EVAL_EPISODES:-10}"
LAB01_ACT_EVAL_EPISODES="${LAB01_ACT_EVAL_EPISODES:-20}"
# ACT inference: 50 beat 100 on lab01 50K EGL eval (64% vs 46% @ 50 ep)
LAB01_ACT_N_ACTION_STEPS="${LAB01_ACT_N_ACTION_STEPS:-50}"
