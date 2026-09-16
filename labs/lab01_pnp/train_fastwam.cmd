#!/bin/bash
# Lab 01 — FastWAM fine-tune from lerobot/fastwam_base.
#
# Defaults target DORobot / single Instinct MI300X (192 GB HBM). Knobs:
# labs/lab01_pnp/_env.sh (section 5d). Override via LAB01_FASTWAM_* env vars.
#
# Usage (from repo root, ROCm venv):
#   source .venv-rocm/bin/activate
#   ./scripts/install-fastwam-deps.sh   # never: pip install -e '.[fastwam]'
#   ./labs/lab01_pnp/train_fastwam.cmd
#
# Smoke:
#   LAB01_FASTWAM_BATCH_SIZE=1 LAB01_FASTWAM_STEPS=20 LAB01_FASTWAM_NUM_WORKERS=2 \
#     ./labs/lab01_pnp/train_fastwam.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

LEROBOT_TRAIN=("$PYTHON" -m simstudio.scripts.train_act)
LAB01_FASTWAM_STATE_DIM="${LAB01_STATE_DIM:-${LAB01_FASTWAM_STATE_DIM:-6}}"
# Dataset state stays at LAB01_FASTWAM_STATE_DIM; proprio_dim may be wider when
# padding Lab 01 joints into a checkpoint's channels (see _env.sh section 5d).
LAB01_FASTWAM_PROPRIO_DIM="${LAB01_FASTWAM_PROPRIO_DIM:-$LAB01_FASTWAM_STATE_DIM}"
if [[ -n "${LAB01_FASTWAM_PAD_ACTION:-}" ]]; then
    export LAB01_FASTWAM_PAD_ACTION
fi
if [[ -n "${LAB01_FASTWAM_UNFREEZE_VIDEO_LAST:-}" ]]; then
    export LAB01_FASTWAM_UNFREEZE_VIDEO_LAST
fi

if [[ ! -d "${LAB01_DATASET_ROOT}/meta" ]]; then
    echo "Dataset root not found or missing meta/: $LAB01_DATASET_ROOT" >&2
    exit 1
fi

if ! "$PYTHON" -c "import transformers, diffusers, torch; hip=getattr(torch.version,'hip',None); assert hip and '+rocm' in torch.__version__" 2>/dev/null; then
    echo "FastWAM extras missing or ROCm torch was replaced." >&2
    echo "  ./scripts/install-fastwam-deps.sh" >&2
    echo "  ./scripts/install-fastwam-deps.sh --repair-torch" >&2
    echo "Do NOT run: pip install -e '.[fastwam]' (pulls CUDA torch)." >&2
    exit 1
fi

_out_avail_kb="$(df -Pk "$(dirname "$LAB01_FASTWAM_OUTPUT")" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$_out_avail_kb" && "$_out_avail_kb" -lt 20000000 && -z "${LAB01_FASTWAM_KEEP_OUTPUT:-}" ]]; then
    if [[ -d /mnt/doscratch ]]; then
        LAB01_FASTWAM_OUTPUT="${LAB01_FASTWAM_OUTPUT_FALLBACK:-/mnt/doscratch/outputs/train/lab01_pnp_fastwam}"
    else
        LAB01_FASTWAM_OUTPUT="${LAB01_FASTWAM_OUTPUT_FALLBACK:-/root/outputs/train/lab01_pnp_fastwam}"
    fi
    echo "Output volume low on space; using LAB01_FASTWAM_OUTPUT=$LAB01_FASTWAM_OUTPUT"
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
mkdir -p "$(dirname "$LAB01_FASTWAM_OUTPUT")"
if [[ -d "$LAB01_FASTWAM_OUTPUT" ]]; then
    if [[ -z "$(ls -A "$LAB01_FASTWAM_OUTPUT" 2>/dev/null)" ]]; then
        rmdir "$LAB01_FASTWAM_OUTPUT"
    elif [[ "$_resume" != true ]]; then
        echo "Output dir exists and is non-empty: $LAB01_FASTWAM_OUTPUT" >&2
        echo "Fresh:     rm -rf $LAB01_FASTWAM_OUTPUT   then re-run." >&2
        echo "Continue:  $0 --resume true" >&2
        exit 1
    fi
fi
_do_resume=false
if [[ "$_resume" == true ]]; then
    _do_resume=true
    echo "Resuming from $LAB01_FASTWAM_OUTPUT"
fi
unset _arg _skip_next _resume

LAB01_FASTWAM_LOG="${LAB01_FASTWAM_LOG:-}"
if [[ -z "$LAB01_FASTWAM_LOG" ]]; then
    _log_avail_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$_log_avail_kb" && "$_log_avail_kb" -lt 1000000 ]]; then
        if [[ -d /mnt/doscratch ]]; then
            LAB01_FASTWAM_LOG="/mnt/doscratch/outputs/train_fastwam.log"
        else
            LAB01_FASTWAM_LOG="/root/outputs/train_fastwam.log"
        fi
    else
        LAB01_FASTWAM_LOG="$REPO_ROOT/train_fastwam.log"
    fi
fi
mkdir -p "$(dirname "$LAB01_FASTWAM_LOG")"

echo "=== Lab 01: FastWAM fine-tune ==="
echo "Base:     $LAB01_FASTWAM_BASE"
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Output:   $LAB01_FASTWAM_OUTPUT"
echo "Steps:    $LAB01_FASTWAM_STEPS  batch_size: $LAB01_FASTWAM_BATCH_SIZE"
echo "Workers:  $LAB01_FASTWAM_NUM_WORKERS  save_freq: $LAB01_FASTWAM_SAVE_FREQ"
echo "Action:   dim=$LAB01_FASTWAM_ACTION_DIM horizon=$LAB01_FASTWAM_ACTION_HORIZON n_action_steps=$LAB01_FASTWAM_N_ACTION_STEPS"
echo "Proprio:  dim=$LAB01_FASTWAM_PROPRIO_DIM  freeze_video=$LAB01_FASTWAM_FREEZE_VIDEO_EXPERT"
echo "Pad 6-D joints into checkpoint channels: ${LAB01_FASTWAM_PAD_ACTION:-0}"
echo "Unfreeze last video MoT layers: ${LAB01_FASTWAM_UNFREEZE_VIDEO_LAST:-0}"
echo "Rename:   $LAB01_FASTWAM_RENAME_MAP"
echo "Norm:     $LAB01_FASTWAM_NORMALIZATION  toggle_dims=$LAB01_FASTWAM_TOGGLE_DIMS"
echo "Log:      $LAB01_FASTWAM_LOG"
if [[ -n "${LAB01_STATE_DIM:-}" ]]; then
    export LAB01_STATE_DIM
else
    export LAB01_STATE_DIM="$LAB01_FASTWAM_STATE_DIM"
fi
echo "State slice LAB01_STATE_DIM=$LAB01_STATE_DIM"
echo ""

_t0="$(date +%s)"
set +e
if [[ "$_do_resume" == true ]]; then
    _cfg="$LAB01_FASTWAM_OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "$_cfg" ]]; then
        echo "Resume needs train_config.json at: $_cfg" >&2
        exit 1
    fi
    echo "Resume config: $_cfg"
    "${LEROBOT_TRAIN[@]}" \
        --config_path="$_cfg" \
        --resume=true \
        --output_dir="$LAB01_FASTWAM_OUTPUT" \
        --steps="$LAB01_FASTWAM_STEPS" \
        --batch_size="$LAB01_FASTWAM_BATCH_SIZE" \
        --save_freq="$LAB01_FASTWAM_SAVE_FREQ" \
        --num_workers="$LAB01_FASTWAM_NUM_WORKERS" \
        --policy.push_to_hub=false \
        "${_filtered_args[@]}" \
        2>&1 | tee "$LAB01_FASTWAM_LOG"
else
    "${LEROBOT_TRAIN[@]}" \
        --policy.path="$LAB01_FASTWAM_BASE" \
        --policy.push_to_hub=false \
        --policy.device=cuda \
        --policy.action_dim="$LAB01_FASTWAM_ACTION_DIM" \
        --policy.proprio_dim="$LAB01_FASTWAM_PROPRIO_DIM" \
        --policy.action_horizon="$LAB01_FASTWAM_ACTION_HORIZON" \
        --policy.n_action_steps="$LAB01_FASTWAM_N_ACTION_STEPS" \
        --policy.image_size="$LAB01_FASTWAM_IMAGE_SIZE" \
        --policy.torch_dtype=bfloat16 \
        --policy.model_id="$LAB01_FASTWAM_MODEL_ID" \
        --policy.tokenizer_model_id="$LAB01_FASTWAM_TOKENIZER_ID" \
        --policy.text_encoder_model_id="$LAB01_FASTWAM_TEXT_ENCODER_ID" \
        --policy.toggle_action_dimensions="$LAB01_FASTWAM_TOGGLE_DIMS" \
        --policy.normalization_mapping="$LAB01_FASTWAM_NORMALIZATION" \
        --policy.freeze_video_expert="$LAB01_FASTWAM_FREEZE_VIDEO_EXPERT" \
        --policy.use_gradient_checkpointing="$LAB01_FASTWAM_GRADIENT_CHECKPOINTING" \
        --policy.loss="$LAB01_FASTWAM_LOSS" \
        --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
        --dataset.root="$LAB01_DATASET_ROOT" \
        --dataset.video_backend=pyav \
        --dataset.image_transforms.enable=true \
        --rename_map="$LAB01_FASTWAM_RENAME_MAP" \
        --output_dir="$LAB01_FASTWAM_OUTPUT" \
        --job_name=lab01_pnp_fastwam \
        --batch_size="$LAB01_FASTWAM_BATCH_SIZE" \
        --steps="$LAB01_FASTWAM_STEPS" \
        --num_workers="$LAB01_FASTWAM_NUM_WORKERS" \
        --save_checkpoint=true \
        --save_freq="$LAB01_FASTWAM_SAVE_FREQ" \
        --log_freq=20 \
        "${_filtered_args[@]}" \
        2>&1 | tee "$LAB01_FASTWAM_LOG"
fi
status=${PIPESTATUS[0]}
set -e
_elapsed=$(( $(date +%s) - _t0 ))
echo ""
echo "Wall time: ${_elapsed}s ($(( _elapsed / 3600 ))h $(( (_elapsed % 3600) / 60 ))m)"

if [[ "$status" -ne 0 ]]; then
    echo "Training exited with error ($status). Last lines of $LAB01_FASTWAM_LOG:"
    tail -40 "$LAB01_FASTWAM_LOG"
    exit "$status"
fi

echo "Training complete. Latest checkpoint:"
ls -td "$LAB01_FASTWAM_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1
