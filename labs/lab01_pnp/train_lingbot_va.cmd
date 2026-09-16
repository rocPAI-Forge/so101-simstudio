#!/bin/bash
# Lab 01 — LingBot-VA stuffed-joint fine-tune (NOT a fair EEF transfer).
# Joints occupy used_action_channel_ids 0–5. Official LIBERO layout is EEF
# Δ + gripper on channels 0–6. See runbook §6.4.2 and lingbot_va_eef_adapt.md.
#
# Defaults target DORobot / single Instinct MI300X (192 GB HBM). Knobs:
# labs/lab01_pnp/_env.sh (section 5c). Override via LAB01_LINGBOT_* env vars.
#
# Usage (from repo root, ROCm venv):
#   source .venv-rocm/bin/activate
#   ./scripts/install-lingbot-va-deps.sh   # never: pip install -e '.[lingbot_va]'
#   ./labs/lab01_pnp/train_lingbot_va.cmd
#
# Smoke:
#   LAB01_LINGBOT_BATCH_SIZE=1 LAB01_LINGBOT_STEPS=20 LAB01_LINGBOT_NUM_WORKERS=2 \
#     ./labs/lab01_pnp/train_lingbot_va.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

LEROBOT_TRAIN=("$PYTHON" -m simstudio.scripts.train_act)

if [[ ! -d "${LAB01_DATASET_ROOT}/meta" ]]; then
    echo "Dataset root not found or missing meta/: $LAB01_DATASET_ROOT" >&2
    exit 1
fi

if ! "$PYTHON" -c "import transformers, diffusers, accelerate, peft, torch; hip=getattr(torch.version,'hip',None); assert hip and '+rocm' in torch.__version__" 2>/dev/null; then
    echo "LingBot-VA extras missing or ROCm torch was replaced." >&2
    echo "  ./scripts/install-lingbot-va-deps.sh" >&2
    echo "  ./scripts/install-lingbot-va-deps.sh --repair-torch" >&2
    echo "Do NOT run: pip install -e '.[lingbot_va]' (pulls CUDA torch)." >&2
    exit 1
fi

_out_avail_kb="$(df -Pk "$(dirname "$LAB01_LINGBOT_OUTPUT")" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$_out_avail_kb" && "$_out_avail_kb" -lt 20000000 && -z "${LAB01_LINGBOT_KEEP_OUTPUT:-}" ]]; then
    if [[ -d /mnt/doscratch ]]; then
        LAB01_LINGBOT_OUTPUT="${LAB01_LINGBOT_OUTPUT_FALLBACK:-/mnt/doscratch/outputs/train/lab01_pnp_lingbot_va}"
    else
        LAB01_LINGBOT_OUTPUT="${LAB01_LINGBOT_OUTPUT_FALLBACK:-/root/outputs/train/lab01_pnp_lingbot_va}"
    fi
    echo "Output volume low on space; using LAB01_LINGBOT_OUTPUT=$LAB01_LINGBOT_OUTPUT"
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
mkdir -p "$(dirname "$LAB01_LINGBOT_OUTPUT")"
if [[ -d "$LAB01_LINGBOT_OUTPUT" ]]; then
    if [[ -z "$(ls -A "$LAB01_LINGBOT_OUTPUT" 2>/dev/null)" ]]; then
        rmdir "$LAB01_LINGBOT_OUTPUT"
    elif [[ "$_resume" != true ]]; then
        echo "Output dir exists and is non-empty: $LAB01_LINGBOT_OUTPUT" >&2
        echo "Fresh:     rm -rf $LAB01_LINGBOT_OUTPUT   then re-run." >&2
        echo "Continue:  $0 --resume true" >&2
        exit 1
    fi
fi
_resume_args=()
_do_resume=false
if [[ "$_resume" == true ]]; then
    _do_resume=true
    _resume_args+=(--resume=true)
    echo "Resuming from $LAB01_LINGBOT_OUTPUT"
fi
unset _arg _skip_next _resume

LAB01_LINGBOT_LOG="${LAB01_LINGBOT_LOG:-}"
if [[ -z "$LAB01_LINGBOT_LOG" ]]; then
    _log_avail_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$_log_avail_kb" && "$_log_avail_kb" -lt 1000000 ]]; then
        if [[ -d /mnt/doscratch ]]; then
            LAB01_LINGBOT_LOG="/mnt/doscratch/outputs/train_lingbot_va.log"
        else
            LAB01_LINGBOT_LOG="/root/outputs/train_lingbot_va.log"
        fi
    else
        LAB01_LINGBOT_LOG="$REPO_ROOT/train_lingbot_va.log"
    fi
fi
mkdir -p "$(dirname "$LAB01_LINGBOT_LOG")"

if [[ "${LAB01_LINGBOT_AUGMENT_QUANTILE_STATS}" == "true" ]]; then
    if ! "$PYTHON" -c "
import json, sys
from pathlib import Path
p = Path('${LAB01_DATASET_ROOT}') / 'meta' / 'stats.json'
d = json.loads(p.read_text())
action = d.get('action', {})
ok = 'q01' in action and 'q99' in action
sys.exit(0 if ok else 1)
" 2>/dev/null; then
        AUGMENT_SCRIPT="$REPO_ROOT/lerobot/src/lerobot/scripts/augment_dataset_quantile_stats.py"
        if [[ ! -f "$AUGMENT_SCRIPT" ]]; then
            echo "Missing quantile stats and augment script: $AUGMENT_SCRIPT" >&2
            exit 1
        fi
        echo "=== Augmenting dataset quantile stats (q01/q99) ==="
        "$PYTHON" "$AUGMENT_SCRIPT" \
            --repo-id="$LAB01_DATASET_REPO_ID" \
            --root="$LAB01_DATASET_ROOT"
        echo ""
    fi
fi

PEFT_ARGS=()
if [[ "${LAB01_LINGBOT_USE_PEFT}" == "true" ]]; then
    # Load the full Hub checkpoint (policy.use_peft must stay false), then wrap
    # via TrainConfig.peft. policy.use_peft=true would treat the Hub repo as an
    # adapter and PeftConfig.from_pretrained would fail.
    PEFT_ARGS+=(
        --peft.method_type=LORA
        --peft.target_modules="$LAB01_LINGBOT_PEFT_TARGET_MODULES"
        --peft.r="$LAB01_LINGBOT_PEFT_R"
        --peft.lora_alpha="$LAB01_LINGBOT_PEFT_ALPHA"
    )
fi

echo "=== Lab 01: LingBot-VA fine-tune ==="
echo "Base:     $LAB01_LINGBOT_BASE"
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Output:   $LAB01_LINGBOT_OUTPUT"
echo "Steps:    $LAB01_LINGBOT_STEPS  batch_size: $LAB01_LINGBOT_BATCH_SIZE"
echo "Workers:  $LAB01_LINGBOT_NUM_WORKERS  save_freq: $LAB01_LINGBOT_SAVE_FREQ"
echo "attn:     $LAB01_LINGBOT_ATTN_MODE  peft: $LAB01_LINGBOT_USE_PEFT"
echo "Channels: $LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS"
echo "Rename:   $LAB01_LINGBOT_RENAME_MAP"
echo "Log:      $LAB01_LINGBOT_LOG"
echo "Note:     LIBERO checkpoint is EEF-delta; Lab 01 joints occupy used channels 0-5."
echo ""

_t0="$(date +%s)"
set +e
if [[ "$_do_resume" == true ]]; then
    _cfg="$LAB01_LINGBOT_OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    if [[ ! -f "$_cfg" ]]; then
        echo "Resume needs train_config.json at: $_cfg" >&2
        exit 1
    fi
    echo "Resume config: $_cfg"
    "${LEROBOT_TRAIN[@]}" \
        --config_path="$_cfg" \
        --resume=true \
        --output_dir="$LAB01_LINGBOT_OUTPUT" \
        --steps="$LAB01_LINGBOT_STEPS" \
        --batch_size="$LAB01_LINGBOT_BATCH_SIZE" \
        --save_freq="$LAB01_LINGBOT_SAVE_FREQ" \
        --num_workers="$LAB01_LINGBOT_NUM_WORKERS" \
        --policy.push_to_hub=false \
        "${_filtered_args[@]}" \
        2>&1 | tee "$LAB01_LINGBOT_LOG"
else
    "${LEROBOT_TRAIN[@]}" \
        --policy.path="$LAB01_LINGBOT_BASE" \
        --policy.push_to_hub=false \
        --policy.device=cuda \
        --policy.attn_mode="$LAB01_LINGBOT_ATTN_MODE" \
        --policy.image_hflip=false \
        --policy.used_action_channel_ids="$LAB01_LINGBOT_USED_ACTION_CHANNEL_IDS" \
        --policy.dtype=bfloat16 \
        --policy.text_encoder_device=cpu \
        "${PEFT_ARGS[@]}" \
        --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
        --dataset.root="$LAB01_DATASET_ROOT" \
        --dataset.video_backend=pyav \
        --dataset.image_transforms.enable=true \
        --rename_map="$LAB01_LINGBOT_RENAME_MAP" \
        --output_dir="$LAB01_LINGBOT_OUTPUT" \
        --job_name=lab01_pnp_lingbot_va \
        --batch_size="$LAB01_LINGBOT_BATCH_SIZE" \
        --steps="$LAB01_LINGBOT_STEPS" \
        --num_workers="$LAB01_LINGBOT_NUM_WORKERS" \
        --save_checkpoint=true \
        --save_freq="$LAB01_LINGBOT_SAVE_FREQ" \
        --log_freq=20 \
        "${_filtered_args[@]}" \
        2>&1 | tee "$LAB01_LINGBOT_LOG"
fi
status=${PIPESTATUS[0]}
set -e
_elapsed=$(( $(date +%s) - _t0 ))
echo ""
echo "Wall time: ${_elapsed}s ($(( _elapsed / 3600 ))h $(( (_elapsed % 3600) / 60 ))m)"

if [[ "$status" -ne 0 ]]; then
    echo "Training exited with error ($status). Last lines of $LAB01_LINGBOT_LOG:"
    tail -40 "$LAB01_LINGBOT_LOG"
    exit "$status"
fi

echo "Training complete. Latest checkpoint:"
ls -td "$LAB01_LINGBOT_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1
