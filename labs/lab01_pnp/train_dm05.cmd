#!/bin/bash
# Lab 01 — DM0.5 fine-tune on the 193-episode agent-yaw set.
#
# Defaults are the Home20 20K recipe (labs/lab01_pnp/_env.sh section 5f), not a
# short smoke. Start from lerobot/dm05_base. Do not resume a finished 10K
# optimizer; decay_steps must equal --steps.
#
# The pinned LeRobot v0.6.0 submodule has no dm05. Point LAB01_LEROBOT_SRC at
# checkout LAB01_LEROBOT_REV.
#
# Published train: 4× AMD Instinct MI210, batch 1 per GPU (effective 4),
# .venv-rocm, about 9 h. Published Home20 eval ran on one NVIDIA GPU.
#
#   LAB01_LEROBOT_SRC=/path/to/lerobot ./labs/lab01_pnp/train_dm05.cmd
#
# Smoke:
#   LAB01_DM05_STEPS=20 LAB01_DM05_NPROC=1 LAB01_DM05_DECAY_STEPS=20 \
#     ./labs/lab01_pnp/train_dm05.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"
if [[ -n "${LAB01_PYTHON:-}" ]]; then
    PYTHON="$LAB01_PYTHON"
fi
lab01_use_lerobot_src dm05

export LAB01_STATE_DIM=6
if [[ ! -d "${LAB01_AGENT193_DATASET_ROOT}/meta" ]]; then
    echo "193-episode dataset not found: $LAB01_AGENT193_DATASET_ROOT" >&2
    echo "Download alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2 into that directory." >&2
    exit 1
fi
if [[ "$LAB01_DM05_DECAY_STEPS" != "$LAB01_DM05_STEPS" ]]; then
    echo "LAB01_DM05_DECAY_STEPS ($LAB01_DM05_DECAY_STEPS) must equal LAB01_DM05_STEPS ($LAB01_DM05_STEPS)." >&2
    echo "A shorter cosine rescales the peak learning rate." >&2
    exit 1
fi

_resume=false
_filtered_args=()
_skip_next=false
for _arg in "$@"; do
    if [[ "$_skip_next" == true ]]; then
        _skip_next=false
        case "$_arg" in
            true|True|TRUE|1) _resume=true ;;
            false|False|FALSE|0) _resume=false ;;
            *) _filtered_args+=("$_arg") ;;
        esac
        continue
    fi
    case "$_arg" in
        --resume) _resume=true; _skip_next=true ;;
        --resume=true|--resume=True|--resume=TRUE) _resume=true ;;
        --resume=false|--resume=False|--resume=FALSE) _resume=false ;;
        *) _filtered_args+=("$_arg") ;;
    esac
done
if [[ -d "$LAB01_DM05_OUTPUT" ]] && [[ -n "$(ls -A "$LAB01_DM05_OUTPUT" 2>/dev/null)" ]] && [[ "$_resume" != true ]]; then
    echo "Output dir exists and is non-empty: $LAB01_DM05_OUTPUT" >&2
    echo "Fresh 20K: rm -rf $LAB01_DM05_OUTPUT" >&2
    echo "Continue:  $0 --resume true" >&2
    exit 1
fi
_resume_args=()
if [[ "$_resume" == true ]]; then
    _resume_args+=(--resume=true)
fi
mkdir -p "$(dirname "$LAB01_DM05_LOG")"

echo "=== Lab 01: DM0.5 193-episode fine-tune ==="
echo "LeRobot:  $LAB01_LEROBOT_SRC  ($LAB01_LEROBOT_REV)"
echo "Base:     $LAB01_DM05_BASE"
echo "Dataset:  $LAB01_AGENT193_DATASET_ROOT"
echo "Output:   $LAB01_DM05_OUTPUT"
echo "Steps:    $LAB01_DM05_STEPS  per-GPU batch: $LAB01_DM05_BATCH_SIZE  nproc: $LAB01_DM05_NPROC"
echo "Cameras:  $LAB01_DM05_IMAGE_KEYS"
echo "State:    first 6 joint positions, absolute actions, add_state=true"
echo "Log:      $LAB01_DM05_LOG"
echo ""

set +e
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$LAB01_DM05_NPROC" \
    --master_port="$LAB01_DM05_MASTER_PORT" \
    -m simstudio.scripts.train_act \
    --policy.path="$LAB01_DM05_BASE" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.chunk_size="$LAB01_DM05_CHUNK_SIZE" \
    --policy.n_action_steps="$LAB01_DM05_N_ACTION_STEPS" \
    --policy.add_state=true \
    --policy.use_relative_actions=false \
    --policy.image_keys="$LAB01_DM05_IMAGE_KEYS" \
    --policy.optimizer_lr="$LAB01_DM05_LR" \
    --policy.scheduler_warmup_steps="$LAB01_DM05_WARMUP" \
    --policy.scheduler_decay_steps="$LAB01_DM05_DECAY_STEPS" \
    --policy.scheduler_decay_lr="$LAB01_DM05_DECAY_LR" \
    --dataset.repo_id="$LAB01_AGENT193_DATASET_REPO_ID" \
    --dataset.root="$LAB01_AGENT193_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --output_dir="$LAB01_DM05_OUTPUT" \
    --job_name="$LAB01_DM05_JOB_NAME" \
    --batch_size="$LAB01_DM05_BATCH_SIZE" \
    --steps="$LAB01_DM05_STEPS" \
    --num_workers="$LAB01_DM05_NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$LAB01_DM05_SAVE_FREQ" \
    --log_freq=20 \
    --wandb.enable=false \
    "${_resume_args[@]}" \
    "${_filtered_args[@]}" \
    2>&1 | tee "$LAB01_DM05_LOG"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
    echo "Training exited with error ($status). Log: $LAB01_DM05_LOG" >&2
    exit "$status"
fi
echo "Checkpoint: $LAB01_DM05_OUTPUT/checkpoints/$(printf '%06d' "$LAB01_DM05_STEPS")/pretrained_model"
echo "Home20: LAB01_POLICY_PATH=... LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_dm05.yaml LAB01_N_ACTION_STEPS=$LAB01_DM05_N_ACTION_STEPS LAB01_HOME20_TAG=dm05_base_agent193_20k ./labs/lab01_pnp/eval_home20.cmd"
