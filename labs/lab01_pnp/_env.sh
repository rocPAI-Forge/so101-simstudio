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
#   5c) Train LingBot-VA — train_lingbot_va.cmd (DORobot / MI300X)
#   5d) Train FastWAM — train_fastwam.cmd (DORobot / MI300X)
#   5e) Train GR00T N1.7 — planned (groot_n1_7_adapt.md; not wired)
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
#    Lab 01 status: CLOSED transfer-failure case on the 50-ep set (runbook §6.4.2).
#    Best measured recipe was freeze-Qwen BC 10K (1/10 fixed); unfreeze Qwen
#    from that ckpt regressed to 0/10 — do not treat FREEZE_QWEN=false as a
#    “next lever” for classroom success.
# -----------------------------------------------------------------------------
LAB01_JEPA_OUTPUT="${LAB01_JEPA_OUTPUT:-./outputs/train/lab01_pnp_vla_jepa}"
LAB01_JEPA_BASE="${LAB01_JEPA_BASE:-lerobot/VLA-JEPA-LIBERO}"
LAB01_JEPA_STEPS="${LAB01_JEPA_STEPS:-10000}"
LAB01_JEPA_BATCH_SIZE="${LAB01_JEPA_BATCH_SIZE:-16}"
LAB01_JEPA_SAVE_FREQ="${LAB01_JEPA_SAVE_FREQ:-2000}"
LAB01_JEPA_NUM_WORKERS="${LAB01_JEPA_NUM_WORKERS:-8}"
LAB01_JEPA_CHUNK_SIZE="${LAB01_JEPA_CHUNK_SIZE:-7}"
LAB01_JEPA_N_ACTION_STEPS="${LAB01_JEPA_N_ACTION_STEPS:-7}"
LAB01_JEPA_WARMUP_STEPS="${LAB01_JEPA_WARMUP_STEPS:-1000}"
# Defaults match upstream VLA-JEPA; Lab 01's best measured run used
# FREEZE_QWEN=true and ENABLE_WORLD_MODEL=false (see §6.4.2). Set false/true
# only when intentionally reproducing a closed negative recipe.
LAB01_JEPA_ENABLE_WORLD_MODEL="${LAB01_JEPA_ENABLE_WORLD_MODEL:-true}"
LAB01_JEPA_FREEZE_QWEN="${LAB01_JEPA_FREEZE_QWEN:-false}"
if [[ -z "${LAB01_JEPA_RENAME_MAP:-}" ]]; then
  LAB01_JEPA_RENAME_MAP='{"observation.images.camera_top":"observation.images.image","observation.images.camera_wrist":"observation.images.image2"}'
fi
if [[ -z "${LAB01_JEPA_REINIT_MODULES:-}" ]]; then
  LAB01_JEPA_REINIT_MODULES='["model.action_model.action_encoder","model.action_model.state_encoder","model.action_model.action_decoder"]'
fi

# -----------------------------------------------------------------------------
# 5c) Train (LingBot-VA) — train_lingbot_va.cmd
#    Base: lerobot/lingbot_va_libero_long (2 cams image/image2, 30-d EEF layout).
#    Lab 01 status: the current default (joints in channels 0–5) is the
#    **unfair stuffed-joint recipe**, not a fair EEF transfer (runbook §6.4.2).
#    Do not treat a pending/failed campaign eval as LingBot-VA Lab 01 success.
#    Fair path: convert teleop joints→EEF labels, 7 channels, eval IK
#    (follow-up adaptation PR / lingbot_va_eef_adapt.md).
#    Extras: ./scripts/install-lingbot-va-deps.sh  (not pip install -e '.[lingbot_va]')
#    20K@32 fits MI300X VRAM (~155 GB) but ~200 s/step. For a ~10 h window use
#    labs/lab01_pnp/campaign_lingbot_fastwam_10h.cmd (bs4, 4 h cap, save every 200).
# -----------------------------------------------------------------------------
LAB01_LINGBOT_OUTPUT="${LAB01_LINGBOT_OUTPUT:-./outputs/train/lab01_pnp_lingbot_va}"
LAB01_LINGBOT_BASE="${LAB01_LINGBOT_BASE:-lerobot/lingbot_va_libero_long}"
LAB01_LINGBOT_STEPS="${LAB01_LINGBOT_STEPS:-20000}"
LAB01_LINGBOT_BATCH_SIZE="${LAB01_LINGBOT_BATCH_SIZE:-32}"
LAB01_LINGBOT_SAVE_FREQ="${LAB01_LINGBOT_SAVE_FREQ:-2000}"
LAB01_LINGBOT_NUM_WORKERS="${LAB01_LINGBOT_NUM_WORKERS:-8}"
LAB01_LINGBOT_ATTN_MODE="${LAB01_LINGBOT_ATTN_MODE:-flex}"
LAB01_LINGBOT_USE_PEFT="${LAB01_LINGBOT_USE_PEFT:-true}"
LAB01_LINGBOT_PEFT_TARGET_MODULES="${LAB01_LINGBOT_PEFT_TARGET_MODULES:-all-linear}"
LAB01_LINGBOT_PEFT_R="${LAB01_LINGBOT_PEFT_R:-16}"
LAB01_LINGBOT_PEFT_ALPHA="${LAB01_LINGBOT_PEFT_ALPHA:-32}"
LAB01_LINGBOT_AUGMENT_QUANTILE_STATS="${LAB01_LINGBOT_AUGMENT_QUANTILE_STATS:-true}"
if [[ -z "${LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS:-}" ]]; then
  LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS='[0,1,2,3,4,5]'
fi
if [[ -z "${LAB01_LINGBOT_RENAME_MAP:-}" ]]; then
  LAB01_LINGBOT_RENAME_MAP='{"observation.images.camera_top":"observation.images.image","observation.images.camera_wrist":"observation.images.image2"}'
fi

# -----------------------------------------------------------------------------
# 5d) Train (FastWAM) — train_fastwam.cmd
#    Base: lerobot/fastwam_base (2 cams image/image2 concat 224×448, 7-DoF / 8-D
#    proprio). Lab01: action_dim=6, proprio_dim=6 (LAB01_STATE_DIM=6), freeze
#    the ~5B video expert, supervise actions only. Cameras: top→image, wrist→image2.
#    Extras: ./scripts/install-fastwam-deps.sh  (not pip install -e '.[fastwam]')
#    Resume 4K→20K (do not rm output): LAB01_FASTWAM_KEEP_OUTPUT=1 \
#      LAB01_FASTWAM_BATCH_SIZE=4 LAB01_FASTWAM_STEPS=20000 \
#      ./labs/lab01_pnp/train_fastwam.cmd --resume true
#
#    Lab 01 status: CLOSED transfer-failure case (runbook §6.4.2). Best recipe
#    (LIBERO pad7 + unfreeze last 4 MoT) still 0% closed-loop; open-loop RMSE
#    0.136 rad. Default LOSS keeps lambda_video=0 — that is the historical Lab
#    01 recipe (not paper video co-training), not a “next lever” for classroom
#    success on this 50-ep set.
#
#    fastwam_base ships NO action_encoder / action head / proprio_encoder, so a
#    6-D run trains that pathway from scratch (see runbook §6.4.2). To inherit a
#    trained one, start from the LIBERO fine-tune and pad 6 joints into its 7-D
#    action / 8-D proprio channels:
#      LAB01_FASTWAM_BASE=ZibinDong/fastwam_libero_uncond_2cam224 \
#      LAB01_FASTWAM_ACTION_DIM=7 LAB01_FASTWAM_PROPRIO_DIM=8 \
#      LAB01_FASTWAM_PAD_ACTION=1 ./labs/lab01_pnp/train_fastwam.cmd
#    Unfreeze last N frozen video MoT layers after freeze_video_expert (resume or
#    continue from a pad7 checkpoint): LAB01_FASTWAM_UNFREEZE_VIDEO_LAST=4
# -----------------------------------------------------------------------------
LAB01_FASTWAM_OUTPUT="${LAB01_FASTWAM_OUTPUT:-./outputs/train/lab01_pnp_fastwam}"
LAB01_FASTWAM_BASE="${LAB01_FASTWAM_BASE:-lerobot/fastwam_base}"
LAB01_FASTWAM_STEPS="${LAB01_FASTWAM_STEPS:-20000}"
LAB01_FASTWAM_BATCH_SIZE="${LAB01_FASTWAM_BATCH_SIZE:-32}"
LAB01_FASTWAM_SAVE_FREQ="${LAB01_FASTWAM_SAVE_FREQ:-2000}"
LAB01_FASTWAM_NUM_WORKERS="${LAB01_FASTWAM_NUM_WORKERS:-8}"
LAB01_FASTWAM_ACTION_DIM="${LAB01_FASTWAM_ACTION_DIM:-6}"
LAB01_FASTWAM_STATE_DIM="${LAB01_FASTWAM_STATE_DIM:-6}"
LAB01_FASTWAM_ACTION_HORIZON="${LAB01_FASTWAM_ACTION_HORIZON:-32}"
LAB01_FASTWAM_N_ACTION_STEPS="${LAB01_FASTWAM_N_ACTION_STEPS:-32}"
LAB01_FASTWAM_IMAGE_SIZE="${LAB01_FASTWAM_IMAGE_SIZE:-[224,448]}"
LAB01_FASTWAM_FREEZE_VIDEO_EXPERT="${LAB01_FASTWAM_FREEZE_VIDEO_EXPERT:-true}"
LAB01_FASTWAM_GRADIENT_CHECKPOINTING="${LAB01_FASTWAM_GRADIENT_CHECKPOINTING:-true}"
# Published fine-tunes can carry local placeholders ("." for model_id /
# tokenizer_model_id) and LIBERO gripper toggle semantics. Pass the real ids
# explicitly so any checkpoint loads the same way; toggle stays off because
# Lab 01 commands joint positions, not a binary gripper.
LAB01_FASTWAM_MODEL_ID="${LAB01_FASTWAM_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
LAB01_FASTWAM_TOKENIZER_ID="${LAB01_FASTWAM_TOKENIZER_ID:-google/umt5-xxl}"
LAB01_FASTWAM_TEXT_ENCODER_ID="${LAB01_FASTWAM_TEXT_ENCODER_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
LAB01_FASTWAM_TOGGLE_DIMS="${LAB01_FASTWAM_TOGGLE_DIMS:-[]}"
# VISUAL must stay IDENTITY: lerobot_train overrides stats with dataset stats and
# a real dataset's per-channel image std would push frames far outside [-1, 1].
if [[ -z "${LAB01_FASTWAM_NORMALIZATION:-}" ]]; then
  LAB01_FASTWAM_NORMALIZATION='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
fi
if [[ -z "${LAB01_FASTWAM_LOSS:-}" ]]; then
  LAB01_FASTWAM_LOSS='{"lambda_video":0.0,"lambda_action":1.0}'
fi
if [[ -z "${LAB01_FASTWAM_RENAME_MAP:-}" ]]; then
  LAB01_FASTWAM_RENAME_MAP='{"observation.images.camera_top":"observation.images.image","observation.images.camera_wrist":"observation.images.image2"}'
fi

# -----------------------------------------------------------------------------
# 5e) Train (GR00T N1.7) — not wired yet (labs/lab01_pnp/groot_n1_7_adapt.md)
#    Base: nvidia/GR00T-N1.7-3B. Do NOT use GR00T-N1.5-3B (removed in LeRobot
#    v0.6). Embodiment: new_embodiment (SO-101), not libero_sim.
#    Never: uv pip install 'lerobot[groot]'  (CUDA torch)
# -----------------------------------------------------------------------------
# LAB01_GROOT_OUTPUT="${LAB01_GROOT_OUTPUT:-./outputs/train/lab01_pnp_groot_n17}"
# LAB01_GROOT_BASE="${LAB01_GROOT_BASE:-nvidia/GR00T-N1.7-3B}"
# LAB01_GROOT_EMBODIMENT_TAG="${LAB01_GROOT_EMBODIMENT_TAG:-new_embodiment}"

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
#    push_act_state6_model.cmd / push_vla_jepa_model.cmd
# -----------------------------------------------------------------------------
LAB01_SMOLVLA_HF_REPO_ID="${LAB01_SMOLVLA_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-smolvla}"
LAB01_ACT_HF_REPO_ID="${LAB01_ACT_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act}"
LAB01_ACT_STATE6_HF_REPO_ID="${LAB01_ACT_STATE6_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act-state6}"
LAB01_ACT_STATE6_CKPT_STEP="${LAB01_ACT_STATE6_CKPT_STEP:-050000}"
LAB01_MOLMO_HF_REPO_ID="${LAB01_MOLMO_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-molmoact2}"
LAB01_MOLMO_CKPT_STEP="${LAB01_MOLMO_CKPT_STEP:-010000}"
LAB01_JEPA_HF_REPO_ID="${LAB01_JEPA_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-vla-jepa}"
LAB01_JEPA_CKPT_STEP="${LAB01_JEPA_CKPT_STEP:-010000}"
# Checkpoint step name under LAB01_TRAIN_OUTPUT when push_smolvla_model does not
# get an explicit LAB01_POLICY_PATH.
LAB01_SMOLVLA_CKPT_STEP="${LAB01_SMOLVLA_CKPT_STEP:-050000}"
# Optional: force official Hub instead of mirror, e.g. https://huggingface.co
# LAB01_HF_UPLOAD_ENDPOINT=
