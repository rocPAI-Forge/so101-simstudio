#!/bin/bash
# Lab 01 — MolmoAct2 fine-tune from lerobot/MolmoAct2-SO100_101-LeRobot.
#
# Defaults target DORobot / single Instinct MI300X (192 GB HBM). Knobs:
# labs/lab01_pnp/_env.sh (section 5). Override via LAB01_MOLMO_* env vars.
#
# Usage (from repo root, ROCm venv):
#   source .venv-rocm/bin/activate
#   ./scripts/install-molmoact2-deps.sh   # peft+transformers; never lerobot[molmoact2]
#   ./labs/lab01_pnp/train_molmoact2.cmd
#
# Smaller GPU / smoke:
#   LAB01_MOLMO_BATCH_SIZE=4 LAB01_MOLMO_STEPS=500 LAB01_MOLMO_NUM_WORKERS=2 \
#     ./labs/lab01_pnp/train_molmoact2.cmd
#
# Pass-through to lerobot-train:
#   ./labs/lab01_pnp/train_molmoact2.cmd --steps 20000 --batch_size 16
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

LEROBOT_TRAIN="$(dirname "$PYTHON")/lerobot-train"
if [[ ! -x "$LEROBOT_TRAIN" ]]; then
    echo "lerobot-train not found. Run: make rocm-sync" >&2
    exit 1
fi

if [[ ! -d "$LAB01_DATASET_ROOT" ]]; then
    echo "Dataset root not found: $LAB01_DATASET_ROOT" >&2
    echo "Set LAB01_DATASET_ROOT or place data under ./datasets/${LAB01_DATASET_NAME}" >&2
    exit 1
fi

# DORobot: /mnt/models_alehe is often full. Keep existing HF_HOME (model already
# cached there) but redirect checkpoints/logs to a volume with free space.
_out_avail_kb="$(df -Pk "$(dirname "$LAB01_MOLMO_OUTPUT")" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$_out_avail_kb" && "$_out_avail_kb" -lt 20000000 && -z "${LAB01_MOLMO_KEEP_OUTPUT:-}" ]]; then
    LAB01_MOLMO_OUTPUT="${LAB01_MOLMO_OUTPUT_FALLBACK:-/root/outputs/train/lab01_pnp_molmoact2}"
    echo "Output volume low on space; using LAB01_MOLMO_OUTPUT=$LAB01_MOLMO_OUTPUT"
fi
# LeRobot refuses to run if output_dir already exists (resume=False). Only make
# the parent; if a stale EMPTY dir is left from a failed run, remove it.
mkdir -p "$(dirname "$LAB01_MOLMO_OUTPUT")"
if [[ -d "$LAB01_MOLMO_OUTPUT" ]]; then
    if [[ -z "$(ls -A "$LAB01_MOLMO_OUTPUT" 2>/dev/null)" ]]; then
        rmdir "$LAB01_MOLMO_OUTPUT"
    else
        echo "Output dir exists and is non-empty: $LAB01_MOLMO_OUTPUT" >&2
        echo "Move/remove it, set LAB01_MOLMO_OUTPUT to a new path, or pass --resume true." >&2
        exit 1
    fi
fi
LAB01_MOLMO_LOG="${LAB01_MOLMO_LOG:-}"
if [[ -z "$LAB01_MOLMO_LOG" ]]; then
    _log_avail_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "$_log_avail_kb" && "$_log_avail_kb" -lt 1000000 ]]; then
        LAB01_MOLMO_LOG="/root/outputs/train_molmoact2.log"
    else
        LAB01_MOLMO_LOG="$REPO_ROOT/train_molmoact2.log"
    fi
fi
mkdir -p "$(dirname "$LAB01_MOLMO_LOG")"

if [[ "${LAB01_MOLMO_ENABLE_LORA_VLM}" == "true" ]]; then
    if ! "$PYTHON" -c "import peft" 2>/dev/null; then
        echo "peft is required for LAB01_MOLMO_ENABLE_LORA_VLM=true" >&2
        echo "  ./scripts/install-molmoact2-deps.sh" >&2
        echo "Do NOT run: uv pip install 'lerobot[molmoact2]' (pulls CUDA torch)." >&2
        echo "Or disable LoRA: LAB01_MOLMO_ENABLE_LORA_VLM=false ./labs/lab01_pnp/train_molmoact2.cmd" >&2
        exit 1
    fi
fi

# MolmoAct2 prefers quantile stats (q01/q99). Lab01 Hub dump may only have mean/std.
if [[ "${LAB01_MOLMO_AUGMENT_QUANTILE_STATS}" == "true" ]]; then
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

LORA_ARGS=()
if [[ "${LAB01_MOLMO_ENABLE_LORA_VLM}" == "true" ]]; then
    LORA_ARGS+=(--policy.enable_lora_vlm=true --policy.enable_lora_action_expert=false)
else
    LORA_ARGS+=(--policy.enable_lora_vlm=false)
fi

echo "=== Lab 01: MolmoAct2 fine-tune ==="
echo "Base:     $LAB01_MOLMO_BASE"
echo "Dataset:  $LAB01_DATASET_ROOT  ($LAB01_DATASET_REPO_ID)"
echo "Output:   $LAB01_MOLMO_OUTPUT"
echo "Steps:    $LAB01_MOLMO_STEPS  batch_size: $LAB01_MOLMO_BATCH_SIZE"
echo "Workers:  $LAB01_MOLMO_NUM_WORKERS  save_freq: $LAB01_MOLMO_SAVE_FREQ"
echo "Chunk:    $LAB01_MOLMO_CHUNK_SIZE / n_action_steps=$LAB01_MOLMO_N_ACTION_STEPS"
echo "LoRA VLM: $LAB01_MOLMO_ENABLE_LORA_VLM"
echo "Rename:   $LAB01_MOLMO_RENAME_MAP"
echo "Joints:   signs=$LAB01_MOLMO_JOINT_SIGNS  offsets=$LAB01_MOLMO_JOINT_OFFSETS (identity for Lab01 radians)"
echo "Log:      $LAB01_MOLMO_LOG"
echo ""

set +e
"$LEROBOT_TRAIN" \
    --policy.path="$LAB01_MOLMO_BASE" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.action_mode=continuous \
    --policy.chunk_size="$LAB01_MOLMO_CHUNK_SIZE" \
    --policy.n_action_steps="$LAB01_MOLMO_N_ACTION_STEPS" \
    --policy.model_dtype=bfloat16 \
    --policy.num_flow_timesteps="$LAB01_MOLMO_NUM_FLOW_TIMESTEPS" \
    --policy.gradient_checkpointing=true \
    --policy.joint_signs="$LAB01_MOLMO_JOINT_SIGNS" \
    --policy.joint_offsets="$LAB01_MOLMO_JOINT_OFFSETS" \
    "${LORA_ARGS[@]}" \
    --dataset.repo_id="$LAB01_DATASET_REPO_ID" \
    --dataset.root="$LAB01_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --rename_map="$LAB01_MOLMO_RENAME_MAP" \
    --output_dir="$LAB01_MOLMO_OUTPUT" \
    --job_name=lab01_pnp_molmoact2 \
    --batch_size="$LAB01_MOLMO_BATCH_SIZE" \
    --steps="$LAB01_MOLMO_STEPS" \
    --num_workers="$LAB01_MOLMO_NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$LAB01_MOLMO_SAVE_FREQ" \
    --log_freq=20 \
    "$@" \
    2>&1 | tee "$LAB01_MOLMO_LOG"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo ""
    echo "Training exited with error ($status). Last lines of $LAB01_MOLMO_LOG:"
    tail -40 "$LAB01_MOLMO_LOG"
    exit "$status"
fi

echo ""
echo "Training complete. Latest checkpoint:"
ls -td "$LAB01_MOLMO_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1
