#!/bin/bash
# Lab 01 — LingBot-VLA-v2 DAgger finetune from MI210 20K on follow2+dagger.
#
# Dataset: LAB01_V2_DAGGER_DATASET_ROOT (default .tmp/...-v2dagger; not Hub).
# Base ckpt: LAB01_V2_POLICY_PATH (MI210 unfreeze 20K).
# Needs: PYTHONPATH side tree (HF LeRobot PR #3967), .venv-rocm, 4×MI210 typical.
# Knobs: labs/lab01_pnp/_env.sh section 5c2.
#
#   ./labs/lab01_pnp/train_lingbot_vla_v2_dagger.cmd
#   LAB01_V2_SKIP_LONG=1 ./labs/lab01_pnp/train_lingbot_vla_v2_dagger.cmd   # smoke only
#   LAB01_V2_DAGGER_TRAIN_EXPERT_ONLY=false ./labs/lab01_pnp/train_lingbot_vla_v2_dagger.cmd
#   # Repeat the last N (DAgger) episodes R× in the sampler:
#   LAB01_V2_DAGGER_UPWEIGHT_LAST=29 LAB01_V2_DAGGER_UPWEIGHT_REPEAT=4 ./labs/lab01_pnp/train_lingbot_vla_v2_dagger.cmd
#   # Sample only the expert suffix of DAgger episodes ({episode_index: prefix_frames}):
#   LAB01_V2_DAGGER_SKIP_PREFIX_JSON=.tmp/datasets/...-v2dagger/dagger_handover.json ...
#
# Scratch twin (do not commit): .tmp/train_lingbot_vla_v2_dagger.sh → calls this.
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_LAB01_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
# shellcheck source=/dev/null
source "$_LAB01_DIR/_env.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
  exit 0
fi

VENV_PY="${VLA_V2_PYTHON:-$REPO_ROOT/.venv-rocm/bin/python}"
export PYTHONPATH="${LAB01_V2_SIDE_SRC}:${PYTHONPATH:-}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
unset CUDA_VISIBLE_DEVICES || true

SLOTS_JSON="${_LAB01_DIR}/configs/lingbot_vla_v2_so101_slots.json"
DATASET_ROOT="${LAB01_V2_DAGGER_DATASET_ROOT}"
DATASET_REPO_ID="${LAB01_V2_DAGGER_DATASET_REPO_ID}"
POLICY_PATH="${LAB01_V2_POLICY_PATH}"
OUTPUT_DIR="${LAB01_V2_DAGGER_TRAIN_OUTPUT}"
LOG_DIR="${LAB01_V2_LOG_DIR:-/tmp/lab01_overnight_logs}"
SMOKE_STEPS="${LAB01_V2_SMOKE_STEPS:-2}"
TRAIN_STEPS="${LAB01_V2_DAGGER_TRAIN_STEPS}"
BATCH_SIZE="${LAB01_V2_BATCH_SIZE:-1}"
NUM_WORKERS="${LAB01_V2_NUM_WORKERS:-2}"
SAVE_FREQ="${LAB01_V2_SAVE_FREQ:-2000}"
SKIP_LONG="${LAB01_V2_SKIP_LONG:-0}"
TRAIN_EXPERT_ONLY="${LAB01_V2_DAGGER_TRAIN_EXPERT_ONLY}"
UPWEIGHT_LAST="${LAB01_V2_DAGGER_UPWEIGHT_LAST}"
UPWEIGHT_REPEAT="${LAB01_V2_DAGGER_UPWEIGHT_REPEAT}"

mkdir -p "$LOG_DIR"
LOCK_FILE="${LAB01_V2_LOCK:-/tmp/lab01_overnight_logs/v2_dagger_train.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another V2 dagger train holds $LOCK_FILE" >&2
  exit 75
fi

NPROC="${LAB01_V2_NPROC:-}"
if [[ -z "$NPROC" ]]; then
  NPROC=$(awk -F',' '{print NF}' <<<"$HIP_VISIBLE_DEVICES")
fi
TORCHRUN=("${VENV_PY%/python}/torchrun" --standalone --nproc_per_node="$NPROC")
if [[ ! -x "${TORCHRUN[0]}" ]]; then
  TORCHRUN=("$VENV_PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
fi

[[ -x "$VENV_PY" ]] || { echo "Missing $VENV_PY" >&2; exit 1; }
[[ -d "$POLICY_PATH" ]] || { echo "Missing policy $POLICY_PATH" >&2; exit 1; }
[[ -d "$DATASET_ROOT/meta" ]] || { echo "Missing dagger dataset $DATASET_ROOT (run collect_v2_dagger.cmd first)" >&2; exit 1; }
[[ -f "$SLOTS_JSON" ]] || { echo "Missing $SLOTS_JSON" >&2; exit 1; }
[[ -d "$LAB01_V2_SIDE_SRC/lerobot" ]] || { echo "Missing V2 side tree: $LAB01_V2_SIDE_SRC (clone PR #3967)" >&2; exit 1; }

eval "$("$VENV_PY" - "$SLOTS_JSON" <<'PY'
import json, shlex, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
state = json.dumps(cfg["state_slots"], separators=(",", ":"))
action = json.dumps(cfg["action_slots"], separators=(",", ":"))
rename = {"observation.images.camera_wrist": "observation.images.camera_wrist_left"}
for src, dst in (cfg.get("camera_map") or {}).items():
    if not src.startswith("observation.images."):
        continue
    dst_full = dst if dst.startswith("observation.images.") else f"observation.images.{dst}"
    if src != dst_full:
        rename[src] = dst_full
print(f"STATE_SLOTS={shlex.quote(state)}")
print(f"ACTION_SLOTS={shlex.quote(action)}")
print(f"RENAME_MAP={shlex.quote(json.dumps(rename, separators=(',', ':')))}")
PY
)"

TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_LOG="${LOG_DIR}/v2_dagger_smoke_${TS}.log"
TRAIN_LOG="${LOG_DIR}/v2_dagger_train_${TS}.log"
_log() { echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/campaign.log"; }

SKIP_PREFIX_JSON="${LAB01_V2_DAGGER_SKIP_PREFIX_JSON}"
if [[ "$UPWEIGHT_LAST" -gt 0 || -n "$SKIP_PREFIX_JSON" ]]; then
  TRAIN_ENTRY=("$_LAB01_DIR/train_upweight_episodes.py"
    --upweight-last "$UPWEIGHT_LAST" --upweight-repeat "$UPWEIGHT_REPEAT")
  if [[ -n "$SKIP_PREFIX_JSON" ]]; then
    [[ -f "$SKIP_PREFIX_JSON" ]] || { echo "Missing $SKIP_PREFIX_JSON" >&2; exit 1; }
    TRAIN_ENTRY+=(--skip-prefix-json "$SKIP_PREFIX_JSON")
  fi
else
  TRAIN_ENTRY=(-m lerobot.scripts.lerobot_train)
fi

run_train() {
  local steps="$1" out="$2" log="$3" job="$4"
  "${TORCHRUN[@]}" "${TRAIN_ENTRY[@]}" \
    --policy.path="$POLICY_PATH" \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
    --policy.freeze_vision_encoder=false \
    --policy.gradient_checkpointing=true \
    --policy.align_params='{}' \
    --policy.state_slots="$STATE_SLOTS" \
    --policy.action_slots="$ACTION_SLOTS" \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --rename_map="$RENAME_MAP" \
    --output_dir="$out" \
    --job_name="$job" \
    --batch_size="$BATCH_SIZE" \
    --steps="$steps" \
    --num_workers="$NUM_WORKERS" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --log_freq=1 \
    --parallelism.dp_shard=-1 \
    --accelerator.mixed_precision=bf16 \
    2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

_log "V2 dagger smoke begin expert_only=${TRAIN_EXPERT_ONLY} upweight_last=${UPWEIGHT_LAST}x${UPWEIGHT_REPEAT} from=${POLICY_PATH} data=${DATASET_ROOT}"
SMOKE_OUT="${OUTPUT_DIR}_smoke"
rm -rf "$SMOKE_OUT"
if ! run_train "$SMOKE_STEPS" "$SMOKE_OUT" "$SMOKE_LOG" "lab01_v2_dagger_smoke"; then
  _log "V2 dagger smoke FAILED log=${SMOKE_LOG}"
  exit 1
fi
_log "V2 dagger smoke OK log=${SMOKE_LOG}"

if [[ "$SKIP_LONG" == "1" ]]; then
  _log "V2 dagger long skipped"
  exit 0
fi

_log "V2 dagger train begin steps=${TRAIN_STEPS} out=${OUTPUT_DIR}"
if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null || true)" && -z "${LAB01_V2_KEEP_OUTPUT:-}" ]]; then
  echo "Output exists: $OUTPUT_DIR (set LAB01_V2_KEEP_OUTPUT=1 to reuse)" >&2
  exit 1
fi
rm -rf "$OUTPUT_DIR"
if ! run_train "$TRAIN_STEPS" "$OUTPUT_DIR" "$TRAIN_LOG" "lab01_v2_dagger10k"; then
  _log "V2 dagger train FAILED log=${TRAIN_LOG}"
  exit 1
fi
_log "V2 dagger train OK out=${OUTPUT_DIR}"
