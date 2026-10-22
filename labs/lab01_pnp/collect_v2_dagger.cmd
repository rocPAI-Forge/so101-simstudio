#!/bin/bash
# Lab 01 — LingBot-VLA-v2 full DAgger (policy → close-without-lift → scripted expert).
#
# Copies the 193 follow2 demos into .tmp/ (full copy, ~6 GB), then appends
# recovery episodes via collect_policy_recovery.py --mode full --register-v2.
# Does **not** modify the Hub follow2 tree.
#
#   ./labs/lab01_pnp/collect_v2_dagger.cmd              # default target 30
#   LAB01_V2_DAGGER_TARGET=1 ./labs/lab01_pnp/collect_v2_dagger.cmd   # smoke
#   LAB01_V2_DAGGER_TARGET=40 LAB01_V2_DAGGER_MAX_ATTEMPTS=160 ./labs/lab01_pnp/collect_v2_dagger.cmd
#
# Needs: MI210/ROCm .venv-rocm, PYTHONPATH side tree (PR #3967), V2 20K ckpt.
# Headless: MUJOCO_GL=egl (default) or osmesa.
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
  exit 0
fi

export MUJOCO_GL="${MUJOCO_GL:-${LAB01_MUJOCO_GL:-egl}}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
unset CUDA_VISIBLE_DEVICES || true
unset DISPLAY || true

POLICY="${LAB01_V2_POLICY_PATH}"
SRC="${LAB01_V2_BASE_DATASET_ROOT}"
DST="${LAB01_V2_DAGGER_DATASET_ROOT}"
SIDE="${LAB01_V2_SIDE_SRC}"
TARGET="${LAB01_V2_DAGGER_TARGET}"
MAX_ATTEMPTS="${LAB01_V2_DAGGER_MAX_ATTEMPTS}"

[[ -d "$POLICY" ]] || { echo "Missing policy: $POLICY" >&2; exit 1; }
[[ -d "$SRC/meta" ]] || { echo "Missing base dataset: $SRC" >&2; exit 1; }
[[ -d "$SIDE/lerobot" ]] || { echo "Missing V2 side tree: $SIDE (clone PR #3967)" >&2; exit 1; }

_clone_base() {
  if [[ -d "$DST/meta" ]]; then
    echo "dagger dataset exists: $DST (will append)"
    return 0
  fi
  mkdir -p "$(dirname "$DST")"
  # No hardlinks: the recorder rewrites meta/, data/ and spawns.jsonl in place,
  # which would leak DAgger episodes and stats into the Hub base tree.
  echo "Copying base demos → $DST"
  cp -a --reflink=auto "$SRC" "$DST"
}

_clone_base

export PYTHONPATH="${SIDE}:${PYTHONPATH:-}"
echo "[$(date -Is)] V2 DAgger begin target=${TARGET} policy=${POLICY} dataset=${DST}"

# Unbuffered so attempt lines show up under nohup.
export PYTHONUNBUFFERED=1
"$PYTHON" -u "$_LAB01_DIR/collect_policy_recovery.py" \
  --mode full \
  --register-v2 \
  --n-action-steps 0 \
  --target "$TARGET" \
  --max-attempts "$MAX_ATTEMPTS" \
  --closed-hold-ticks "${LAB01_V2_DAGGER_CLOSED_HOLD_TICKS}" \
  --policy-path "$POLICY" \
  --policy-tag "${LAB01_V2_DAGGER_POLICY_TAG}" \
  --rollout-config "$_LAB01_DIR/configs/rollout_lingbot_vla_v2.yaml" \
  --dataset-root "$DST" \
  --dataset-repo-id "${LAB01_V2_DAGGER_DATASET_REPO_ID}" \
  --stats-root "$SRC" \
  --stats-repo-id "${LAB01_V2_BASE_DATASET_REPO_ID}"

echo "[$(date -Is)] V2 DAgger done dataset=${DST}"
