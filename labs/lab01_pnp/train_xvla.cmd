#!/bin/bash
# Lab 01 — xVLA full fine-tune on the 193-episode agent-yaw set.
#
# Defaults are the Home20 10K recipe (labs/lab01_pnp/_env.sh section 5f), not a
# short smoke. The pinned LeRobot v0.6.0 submodule has no action_mode=auto.
# Point LAB01_LEROBOT_SRC at checkout LAB01_LEROBOT_REV.
#
# Published run: 4× NVIDIA A30, batch 4 per GPU (effective 16), .venv-cuda.
#
#   git clone https://github.com/huggingface/lerobot.git /path/to/lerobot
#   git -C /path/to/lerobot checkout "$LAB01_LEROBOT_REV"
#   LAB01_LEROBOT_SRC=/path/to/lerobot \
#   LAB01_PYTHON=.venv-cuda/bin/python \
#     ./labs/lab01_pnp/train_xvla.cmd
#
# Smoke:
#   LAB01_XVLA_STEPS=20 LAB01_XVLA_NPROC=1 LAB01_XVLA_BATCH_SIZE=1 \
#     LAB01_XVLA_DECAY_STEPS=20 ./labs/lab01_pnp/train_xvla.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"
if [[ -n "${LAB01_PYTHON:-}" ]]; then
    PYTHON="$LAB01_PYTHON"
fi
lab01_use_lerobot_src xvla

export LAB01_STATE_DIM=6
if [[ ! -d "${LAB01_AGENT193_DATASET_ROOT}/meta" ]]; then
    echo "193-episode dataset not found: $LAB01_AGENT193_DATASET_ROOT" >&2
    echo "Download alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2 into that directory." >&2
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
if [[ -d "$LAB01_XVLA_OUTPUT" ]] && [[ -n "$(ls -A "$LAB01_XVLA_OUTPUT" 2>/dev/null)" ]] && [[ "$_resume" != true ]]; then
    echo "Output dir exists and is non-empty: $LAB01_XVLA_OUTPUT" >&2
    echo "Fresh run: rm -rf $LAB01_XVLA_OUTPUT" >&2
    echo "Continue:  $0 --resume true" >&2
    exit 1
fi
_resume_args=()
if [[ "$_resume" == true ]]; then
    _resume_args+=(--resume=true)
fi

mkdir -p "$(dirname "$LAB01_XVLA_ADAPTED_BASE")" "$(dirname "$LAB01_XVLA_LOG")"
LAB01_XVLA_TRAIN_BASE="$(
    LAB01_XVLA_BASE="$LAB01_XVLA_BASE" \
    LAB01_XVLA_ADAPTED_BASE="$LAB01_XVLA_ADAPTED_BASE" \
    "$PYTHON" - <<'PY' | tail -n 1
import json
import os
from pathlib import Path

base = os.environ["LAB01_XVLA_BASE"]
dest = Path(os.environ["LAB01_XVLA_ADAPTED_BASE"])
src = Path(base)
if src.is_dir():
    snap = src
else:
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(base))

cfg = json.loads((snap / "config.json").read_text())
action = (cfg.get("output_features") or {}).get("action") or {}
if cfg.get("action_mode") == "auto" and action.get("shape") == [6]:
    print(snap.resolve())
    raise SystemExit(0)

dest.mkdir(parents=True, exist_ok=True)
cfg["action_mode"] = "auto"
cfg["push_to_hub"] = False
features = cfg.setdefault("output_features", {})
action = dict(features.get("action") or {"type": "ACTION"})
action["type"] = action.get("type", "ACTION")
action["shape"] = [6]
features["action"] = action
(dest / "config.json").write_text(json.dumps(cfg, indent=4) + "\n")
for src_file in snap.iterdir():
    if src_file.name == "config.json" or not src_file.is_file():
        continue
    link = dest / src_file.name
    if link.exists() or link.is_symlink():
        continue
    link.symlink_to(src_file.resolve())
if not (dest / "model.safetensors").exists() and not (dest / "model.safetensors.index.json").exists():
    raise SystemExit(f"adapted base {dest} has no model.safetensors")
print(dest.resolve())
PY
)"

echo "=== Lab 01: xVLA 193-episode fine-tune ==="
echo "LeRobot:  $LAB01_LEROBOT_SRC  ($LAB01_LEROBOT_REV)"
echo "Base:     $LAB01_XVLA_TRAIN_BASE"
echo "Dataset:  $LAB01_AGENT193_DATASET_ROOT"
echo "Output:   $LAB01_XVLA_OUTPUT"
echo "Steps:    $LAB01_XVLA_STEPS  per-GPU batch: $LAB01_XVLA_BATCH_SIZE  nproc: $LAB01_XVLA_NPROC"
echo "State:    first 6 joint positions; action_mode=auto keeps the 20-D head"
echo "Log:      $LAB01_XVLA_LOG"
echo ""

set +e
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$LAB01_XVLA_NPROC" \
    --master_port="$LAB01_XVLA_MASTER_PORT" \
    -m simstudio.scripts.train_act \
    --policy.path="$LAB01_XVLA_TRAIN_BASE" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.action_mode=auto \
    --policy.chunk_size="$LAB01_XVLA_CHUNK_SIZE" \
    --policy.n_action_steps="$LAB01_XVLA_N_ACTION_STEPS" \
    --policy.optimizer_lr="$LAB01_XVLA_LR" \
    --policy.scheduler_warmup_steps="$LAB01_XVLA_WARMUP" \
    --policy.scheduler_decay_steps="$LAB01_XVLA_DECAY_STEPS" \
    --policy.scheduler_decay_lr="$LAB01_XVLA_DECAY_LR" \
    --dataset.repo_id="$LAB01_AGENT193_DATASET_REPO_ID" \
    --dataset.root="$LAB01_AGENT193_DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true \
    --rename_map="$LAB01_XVLA_RENAME_MAP" \
    --output_dir="$LAB01_XVLA_OUTPUT" \
    --job_name="$LAB01_XVLA_JOB_NAME" \
    --batch_size="$LAB01_XVLA_BATCH_SIZE" \
    --steps="$LAB01_XVLA_STEPS" \
    --num_workers="$LAB01_XVLA_NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$LAB01_XVLA_SAVE_FREQ" \
    --log_freq=20 \
    --wandb.enable=false \
    "${_resume_args[@]}" \
    "${_filtered_args[@]}" \
    2>&1 | tee "$LAB01_XVLA_LOG"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
    echo "Training exited with error ($status). Log: $LAB01_XVLA_LOG" >&2
    exit "$status"
fi
echo "Checkpoint: $LAB01_XVLA_OUTPUT/checkpoints/$(printf '%06d' "$LAB01_XVLA_STEPS")/pretrained_model"
echo "Home20: LAB01_POLICY_PATH=... LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_xvla.yaml LAB01_N_ACTION_STEPS=$LAB01_XVLA_N_ACTION_STEPS LAB01_HOME20_TAG=xvla_agent193_10k ./labs/lab01_pnp/eval_home20.cmd"
