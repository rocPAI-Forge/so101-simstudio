#!/bin/bash
# Lab 01 — VLA-JEPA fine-tune from lerobot/VLA-JEPA-LIBERO.
#
# Defaults target DORobot / single Instinct MI300X (192 GB HBM). Knobs:
# labs/lab01_pnp/_env.sh (section 5b). Override via LAB01_JEPA_* env vars.
#
# Usage (from repo root, ROCm venv):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/train_vla_jepa.cmd
#   ./labs/lab01_pnp/train_vla_jepa.cmd --resume true
#
# Smoke:
#   LAB01_JEPA_BATCH_SIZE=2 LAB01_JEPA_STEPS=50 LAB01_JEPA_NUM_WORKERS=2 \
#     ./labs/lab01_pnp/train_vla_jepa.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

# Optional LAB01_STATE_DIM=6 slices observation.state after parquet load.
LEROBOT_TRAIN=("$PYTHON" -m simstudio.scripts.train_act)
LAB01_JEPA_STATE_DIM="${LAB01_STATE_DIM:-${LAB01_JEPA_STATE_DIM:-15}}"

if [[ ! -d "${LAB01_DATASET_ROOT}/meta" ]]; then
    echo "Dataset root not found or missing meta/: $LAB01_DATASET_ROOT" >&2
    echo "Unset LAB01_DATASET_ROOT if it is stale, or point it at the LeRobot v3 root (the folder that contains meta/)." >&2
    echo "Tried: ./datasets/${LAB01_DATASET_NAME} and nested hf-download layout." >&2
    ls -la ./datasets 2>/dev/null || true
    exit 1
fi

_out_avail_kb="$(df -Pk "$(dirname "$LAB01_JEPA_OUTPUT")" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$_out_avail_kb" && "$_out_avail_kb" -lt 20000000 && -z "${LAB01_JEPA_KEEP_OUTPUT:-}" ]]; then
    if [[ -d /mnt/doscratch ]]; then
        LAB01_JEPA_OUTPUT="${LAB01_JEPA_OUTPUT_FALLBACK:-/mnt/doscratch/outputs/train/lab01_pnp_vla_jepa}"
    else
        LAB01_JEPA_OUTPUT="${LAB01_JEPA_OUTPUT_FALLBACK:-/root/outputs/train/lab01_pnp_vla_jepa}"
    fi
    echo "Output volume low on space; using LAB01_JEPA_OUTPUT=$LAB01_JEPA_OUTPUT"
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
mkdir -p "$(dirname "$LAB01_JEPA_OUTPUT")"
if [[ -d "$LAB01_JEPA_OUTPUT" ]]; then
    if [[ -z "$(ls -A "$LAB01_JEPA_OUTPUT" 2>/dev/null)" ]]; then
        rmdir "$LAB01_JEPA_OUTPUT"
    elif [[ "$_resume" != true ]]; then
        echo "Output dir exists and is non-empty: $LAB01_JEPA_OUTPUT" >&2
        echo "Fresh 10K:  rm -rf $LAB01_JEPA_OUTPUT   then re-run." >&2
        echo "Continue:   $0 --resume true" >&2
        echo "Or set LAB01_JEPA_OUTPUT to a new path." >&2
        exit 1
    fi
fi
_resume_args=()
_do_resume=false
if [[ "$_resume" == true ]]; then
    _do_resume=true
    _resume_args+=(--resume=true)
    echo "Resuming from $LAB01_JEPA_OUTPUT"
fi
unset _arg _skip_next _resume
LAB01_JEPA_LOG="${LAB01_JEPA_LOG:-}"
if [[ -z "$LAB01_JEPA_LOG" ]]; then
    _log_avail_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$_log_avail_kb" && "$_log_avail_kb" -lt 1000000 ]]; then
        if [[ -d /mnt/doscratch ]]; then
            LAB01_JEPA_LOG="/mnt/doscratch/outputs/train_vla_jepa.log"
        else
            LAB01_JEPA_LOG="/root/outputs/train_vla_jepa.log"
        fi
    else
        LAB01_JEPA_LOG="$REPO_ROOT/train_vla_jepa.log"
    fi
fi
mkdir -p "$(dirname "$LAB01_JEPA_LOG")"

WM_ARGS=(--policy.enable_world_model="${LAB01_JEPA_ENABLE_WORLD_MODEL}")
if [[ "${LAB01_JEPA_FREEZE_QWEN}" == "true" ]]; then
    WM_ARGS+=(--policy.freeze_qwen=true)
else
    WM_ARGS+=(--policy.freeze_qwen=false)
fi

echo "=== Lab 01: VLA-JEPA fine-tune ==="
echo "Base:     $LAB01_JEPA_BASE"
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Output:   $LAB01_JEPA_OUTPUT"
echo "Steps:    $LAB01_JEPA_STEPS  batch_size: $LAB01_JEPA_BATCH_SIZE"
echo "Workers:  $LAB01_JEPA_NUM_WORKERS  save_freq: $LAB01_JEPA_SAVE_FREQ"
echo "Chunk:    $LAB01_JEPA_CHUNK_SIZE / n_action_steps=$LAB01_JEPA_N_ACTION_STEPS"
echo "World:    enable=$LAB01_JEPA_ENABLE_WORLD_MODEL  freeze_qwen=$LAB01_JEPA_FREEZE_QWEN"
echo "Rename:   $LAB01_JEPA_RENAME_MAP"
echo "Reinit:   $LAB01_JEPA_REINIT_MODULES"
echo "State:    policy.state_dim=$LAB01_JEPA_STATE_DIM"
echo "Log:      $LAB01_JEPA_LOG"
if [[ -n "${LAB01_STATE_DIM:-}" ]]; then
    export LAB01_STATE_DIM
fi
echo ""

set +e
if [[ "$_do_resume" == true ]]; then
    _jepa_config="$LAB01_JEPA_OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "$_jepa_config" ]]; then
        echo "Resume needs train_config.json at: $_jepa_config" >&2
        echo "Check checkpoints/last → a complete step with pretrained_model/ + training_state/." >&2
        exit 1
    fi
    echo "Resume config: $_jepa_config"
    "${LEROBOT_TRAIN[@]}" \
        --config_path="$_jepa_config" \
        --resume=true \
        --output_dir="$LAB01_JEPA_OUTPUT" \
        --steps="$LAB01_JEPA_STEPS" \
        --batch_size="$LAB01_JEPA_BATCH_SIZE" \
        --save_freq="$LAB01_JEPA_SAVE_FREQ" \
        --num_workers="$LAB01_JEPA_NUM_WORKERS" \
        --policy.push_to_hub=false \
        "${_filtered_args[@]}" \
        2>&1 | tee "$LAB01_JEPA_LOG"
else
    "${LEROBOT_TRAIN[@]}" \
        --policy.path="$LAB01_JEPA_BASE" \
        --policy.push_to_hub=false \
        --policy.device=cuda \
        --policy.action_dim=6 \
        --policy.state_dim="$LAB01_JEPA_STATE_DIM" \
        --policy.chunk_size="$LAB01_JEPA_CHUNK_SIZE" \
        --policy.n_action_steps="$LAB01_JEPA_N_ACTION_STEPS" \
        --policy.torch_dtype=bfloat16 \
        --policy.binarize_gripper_action=false \
        --policy.pre_snap_gripper_action=false \
        --policy.resize_images_to='[224,224]' \
        --policy.reinit_modules="$LAB01_JEPA_REINIT_MODULES" \
        --policy.scheduler_warmup_steps=1000 \
        --policy.scheduler_decay_steps="$LAB01_JEPA_STEPS" \
        "${WM_ARGS[@]}" \
        --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
        --dataset.root="$LAB01_DATASET_ROOT" \
        --dataset.video_backend=pyav \
        --dataset.image_transforms.enable=true \
        --rename_map="$LAB01_JEPA_RENAME_MAP" \
        --output_dir="$LAB01_JEPA_OUTPUT" \
        --job_name=lab01_pnp_vla_jepa \
        --batch_size="$LAB01_JEPA_BATCH_SIZE" \
        --steps="$LAB01_JEPA_STEPS" \
        --num_workers="$LAB01_JEPA_NUM_WORKERS" \
        --save_checkpoint=true \
        --save_freq="$LAB01_JEPA_SAVE_FREQ" \
        --log_freq=20 \
        "${_filtered_args[@]}" \
        2>&1 | tee "$LAB01_JEPA_LOG"
fi
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Training exited with error ($status). Last lines of $LAB01_JEPA_LOG:"
    tail -40 "$LAB01_JEPA_LOG"
    exit "$status"
fi

echo ""
echo "Training complete. Latest checkpoint:"
ls -td "$LAB01_JEPA_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1
