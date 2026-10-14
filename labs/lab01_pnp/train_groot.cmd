#!/bin/bash
# Lab 01 — GR00T N1.7 fine-tune from nvidia/GR00T-N1.7-3B (not N1.5).
#
# Joint-space new_embodiment on the existing 50-ep dataset. Spec:
# labs/lab01_pnp/groot_n1_7_adapt.md
#
#   source .venv-rocm/bin/activate
#   ./scripts/install-groot-deps.sh
#   ./labs/lab01_pnp/train_groot.cmd          # default: 20-step smoke
#
# MI300X-ish (override; still not a classroom policy until §6.4 is filled):
#   LAB01_GROOT_STEPS=10000 LAB01_GROOT_BATCH_SIZE=4 LAB01_GROOT_SAVE_FREQ=2000 \
#     ./labs/lab01_pnp/train_groot.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

export LAB01_GROOT=1
if [[ -n "${LAB01_GROOT_PROCESSOR_MODEL:-}" ]]; then
    export LAB01_GROOT_PROCESSOR_MODEL
fi
LEROBOT_TRAIN=("$PYTHON" -m simstudio.scripts.train_act)
LAB01_GROOT_STATE_DIM="${LAB01_STATE_DIM:-${LAB01_GROOT_STATE_DIM:-6}}"

if [[ ! -d "${LAB01_DATASET_ROOT}/meta" ]]; then
    echo "Dataset root not found or missing meta/: $LAB01_DATASET_ROOT" >&2
    exit 1
fi

if ! "$PYTHON" -c "import transformers, peft, diffusers, timm, tree; import torch; hip=getattr(torch.version,'hip',None); assert hip and '+rocm' in torch.__version__" 2>/dev/null; then
    echo "GR00T extras missing or ROCm torch was replaced." >&2
    echo "  ./scripts/install-groot-deps.sh" >&2
    echo "  ./scripts/install-groot-deps.sh --repair-torch" >&2
    echo "Do NOT run: uv pip install 'lerobot[groot]' (pulls CUDA torch)." >&2
    exit 1
fi

case "${LAB01_GROOT_BASE}" in
    *n1.5*|*N1.5*|*n1_5*)
        echo "LAB01_GROOT_BASE looks like GR00T N1.5 ($LAB01_GROOT_BASE)." >&2
        echo "Lab 01 only supports nvidia/GR00T-N1.7-3B on this LeRobot pin." >&2
        exit 1
        ;;
esac
if [[ "${LAB01_GROOT_EMBODIMENT_TAG}" == "libero_sim" ]]; then
    echo "Do not use embodiment_tag=libero_sim on Lab 01 joints (JEPA/FastWAM class of mistake)." >&2
    exit 1
fi

_out_avail_kb="$(df -Pk "$(dirname "$LAB01_GROOT_OUTPUT")" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$_out_avail_kb" && "$_out_avail_kb" -lt 20000000 && -z "${LAB01_GROOT_KEEP_OUTPUT:-}" ]]; then
    if [[ -d /mnt/doscratch ]]; then
        LAB01_GROOT_OUTPUT="${LAB01_GROOT_OUTPUT_FALLBACK:-/mnt/doscratch/outputs/train/lab01_pnp_groot_n17}"
    else
        LAB01_GROOT_OUTPUT="${LAB01_GROOT_OUTPUT_FALLBACK:-/root/outputs/train/lab01_pnp_groot_n17}"
    fi
    echo "Output volume low on space; using LAB01_GROOT_OUTPUT=$LAB01_GROOT_OUTPUT"
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
mkdir -p "$(dirname "$LAB01_GROOT_OUTPUT")"
if [[ -d "$LAB01_GROOT_OUTPUT" ]]; then
    if [[ -z "$(ls -A "$LAB01_GROOT_OUTPUT" 2>/dev/null)" ]]; then
        rmdir "$LAB01_GROOT_OUTPUT"
    elif [[ "$_resume" != true ]]; then
        echo "Output dir exists and is non-empty: $LAB01_GROOT_OUTPUT" >&2
        echo "Fresh:     rm -rf $LAB01_GROOT_OUTPUT   then re-run." >&2
        echo "Continue:  $0 --resume true" >&2
        exit 1
    fi
fi
_resume_args=()
if [[ "$_resume" == true ]]; then
    _resume_args+=(--resume=true)
    echo "Resuming from $LAB01_GROOT_OUTPUT"
fi
set -- "${_filtered_args[@]}"

LAB01_GROOT_LOG="${LAB01_GROOT_LOG:-}"
if [[ -z "$LAB01_GROOT_LOG" ]]; then
    LAB01_GROOT_LOG="$REPO_ROOT/train_groot.log"
    if [[ -d /mnt/doscratch ]]; then
        _log_avail_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
        if [[ -n "$_log_avail_kb" && "$_log_avail_kb" -lt 1000000 ]]; then
            LAB01_GROOT_LOG="/mnt/doscratch/outputs/train_groot.log"
        fi
    fi
fi
mkdir -p "$(dirname "$LAB01_GROOT_LOG")"

RENAME_ARGS=()
if [[ -n "${LAB01_GROOT_RENAME_MAP:-}" && "$LAB01_GROOT_RENAME_MAP" != "{}" ]]; then
    RENAME_ARGS+=(--rename_map="$LAB01_GROOT_RENAME_MAP")
fi

echo "=== Lab 01: GR00T N1.7 fine-tune (joint new_embodiment) ==="
echo "Base:     $LAB01_GROOT_BASE"
echo "Tag:      $LAB01_GROOT_EMBODIMENT_TAG"
echo "Dataset:  $LAB01_DATASET_ROOT"
echo "Output:   $LAB01_GROOT_OUTPUT"
echo "Steps:    $LAB01_GROOT_STEPS  batch_size: $LAB01_GROOT_BATCH_SIZE"
echo "Chunk:    $LAB01_GROOT_CHUNK_SIZE / n_action_steps=$LAB01_GROOT_N_ACTION_STEPS"
echo "Relative: $LAB01_GROOT_USE_RELATIVE_ACTIONS  exclude=$LAB01_GROOT_RELATIVE_EXCLUDE_JOINTS"
echo "VLM tok:  ${LAB01_GROOT_PROCESSOR_MODEL:-Qwen/Qwen3-VL-2B-Instruct (Cosmos-Reason2-2B is gated)}"
echo "State:    first $LAB01_GROOT_STATE_DIM dims"
echo "Log:      $LAB01_GROOT_LOG"
echo ""

export LAB01_STATE_DIM="$LAB01_GROOT_STATE_DIM"

set +e
"${LEROBOT_TRAIN[@]}" \
    --policy.type=groot \
    --policy.base_model_path="$LAB01_GROOT_BASE" \
    --policy.embodiment_tag="$LAB01_GROOT_EMBODIMENT_TAG" \
    --policy.action_decode_transform=none \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size="$LAB01_GROOT_CHUNK_SIZE" \
    --policy.n_action_steps="$LAB01_GROOT_N_ACTION_STEPS" \
    --policy.use_relative_actions="$LAB01_GROOT_USE_RELATIVE_ACTIONS" \
    --policy.relative_exclude_joints="$LAB01_GROOT_RELATIVE_EXCLUDE_JOINTS" \
    --policy.use_bf16="$LAB01_GROOT_USE_BF16" \
    --policy.use_flash_attention=false \
    --policy.tune_llm="$LAB01_GROOT_TUNE_LLM" \
    --policy.tune_visual="$LAB01_GROOT_TUNE_VISUAL" \
    --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
    --dataset.root="$LAB01_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    "${RENAME_ARGS[@]}" \
    --output_dir="$LAB01_GROOT_OUTPUT" \
    --job_name=lab01_pnp_groot_n17 \
    --batch_size="$LAB01_GROOT_BATCH_SIZE" \
    --steps="$LAB01_GROOT_STEPS" \
    --num_workers="$LAB01_GROOT_NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$LAB01_GROOT_SAVE_FREQ" \
    --log_freq=1 \
    --wandb.enable=false \
    "${_resume_args[@]}" \
    "$@" \
    2>&1 | tee "$LAB01_GROOT_LOG"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Training exited with error ($status). Last lines of $LAB01_GROOT_LOG:"
    tail -50 "$LAB01_GROOT_LOG"
    exit "$status"
fi

echo ""
echo "Training complete. Latest checkpoint:"
ls -td "$LAB01_GROOT_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1
