#!/bin/bash
# Lab 01 — time-boxed LingBot-VA + FastWAM train then 20-ep EGL eval.
#
# 20K @ batch 32 is VRAM-ok on MI300X (~155 GB) but ~200 s/step (~weeks).
# This campaign fits a ~10 h window: smaller batch, frequent checkpoints,
# wall-clock cap per phase, then headless eval of the latest ckpt.
#
# Intended to run on DORobot:
#   nohup ./labs/lab01_pnp/campaign_lingbot_fastwam_10h.cmd \
#     > /mnt/doscratch/outputs/lab01_ext_campaign.outer.log 2>&1 &
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

STATUS="${LAB01_CAMPAIGN_STATUS:-/mnt/doscratch/outputs/lab01_ext_campaign.status}"
OUT_ROOT="${LAB01_CAMPAIGN_ROOT:-/mnt/doscratch/outputs}"
mkdir -p "$OUT_ROOT" "$(dirname "$STATUS")"

# Force 10h recipe *after* _env.sh (which defaults to 20K @ 32).
LINGBOT_HOURS="${LAB01_CAMPAIGN_LINGBOT_HOURS:-4}"
FASTWAM_HOURS="${LAB01_CAMPAIGN_FASTWAM_HOURS:-3.5}"
LINGBOT_STEPS="${LAB01_CAMPAIGN_LINGBOT_STEPS:-8000}"
LINGBOT_BATCH="${LAB01_CAMPAIGN_LINGBOT_BATCH:-4}"
LINGBOT_SAVE="${LAB01_CAMPAIGN_LINGBOT_SAVE:-200}"
FASTWAM_STEPS="${LAB01_CAMPAIGN_FASTWAM_STEPS:-4000}"
FASTWAM_BATCH="${LAB01_CAMPAIGN_FASTWAM_BATCH:-4}"
FASTWAM_SAVE="${LAB01_CAMPAIGN_FASTWAM_SAVE:-200}"
EVAL_EPS="${LAB01_CAMPAIGN_EVAL_EPISODES:-20}"

write_status() {
    printf '%s\n' "$1" > "$STATUS"
    echo "[campaign $(date -Is)] $1"
}

hours_to_sec() {
    python3 - <<PY
h = float("${1}")
print(int(h * 3600))
PY
}

latest_ckpt() {
    local root="$1"
    ls -td "$root"/checkpoints/*/pretrained_model 2>/dev/null | head -1 || true
}

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv-rocm/bin/activate

write_status "phase=lingbot_train started=$(date -Is)"

export LAB01_LINGBOT_OUTPUT="$OUT_ROOT/train/lab01_pnp_lingbot_va"
export LAB01_LINGBOT_LOG="$OUT_ROOT/train_lingbot_va.log"
export LAB01_LINGBOT_STEPS="$LINGBOT_STEPS"
export LAB01_LINGBOT_BATCH_SIZE="$LINGBOT_BATCH"
export LAB01_LINGBOT_SAVE_FREQ="$LINGBOT_SAVE"
export LAB01_LINGBOT_NUM_WORKERS=4

rm -rf "$LAB01_LINGBOT_OUTPUT"
set +e
timeout --signal=TERM --kill-after=120 "$(hours_to_sec "$LINGBOT_HOURS")s" \
    "$_LAB01_DIR/train_lingbot_va.cmd"
lingbot_rc=$?
set -e
LINGBOT_CKPT="$(latest_ckpt "$LAB01_LINGBOT_OUTPUT")"
write_status "phase=lingbot_done rc=$lingbot_rc ckpt=$LINGBOT_CKPT ended=$(date -Is)"
if [[ -z "$LINGBOT_CKPT" ]]; then
    write_status "phase=failed reason=no_lingbot_ckpt rc=$lingbot_rc"
    exit 1
fi

write_status "phase=fastwam_train started=$(date -Is)"
export LAB01_FASTWAM_OUTPUT="$OUT_ROOT/train/lab01_pnp_fastwam"
export LAB01_FASTWAM_LOG="$OUT_ROOT/train_fastwam.log"
export LAB01_FASTWAM_STEPS="$FASTWAM_STEPS"
export LAB01_FASTWAM_BATCH_SIZE="$FASTWAM_BATCH"
export LAB01_FASTWAM_SAVE_FREQ="$FASTWAM_SAVE"
export LAB01_FASTWAM_NUM_WORKERS=4

rm -rf "$LAB01_FASTWAM_OUTPUT"
set +e
timeout --signal=TERM --kill-after=120 "$(hours_to_sec "$FASTWAM_HOURS")s" \
    "$_LAB01_DIR/train_fastwam.cmd"
fastwam_rc=$?
set -e
FASTWAM_CKPT="$(latest_ckpt "$LAB01_FASTWAM_OUTPUT")"
write_status "phase=fastwam_done rc=$fastwam_rc ckpt=$FASTWAM_CKPT ended=$(date -Is)"
if [[ -z "$FASTWAM_CKPT" ]]; then
    write_status "phase=failed reason=no_fastwam_ckpt rc=$fastwam_rc"
    exit 1
fi

write_status "phase=eval started=$(date -Is)"
export MUJOCO_GL="${LAB01_MUJOCO_GL:-egl}"
unset DISPLAY
export LAB01_RENDER_WINDOW=false
export LAB01_EVAL_EPISODES="$EVAL_EPS"

export LAB01_POLICY_PATH="$LINGBOT_CKPT"
export LAB01_EVAL_CONFIG="labs/lab01_pnp/configs/rollout_lingbot_va.yaml"
export LAB01_EVAL_LOG="$OUT_ROOT/eval/lingbot_va_egl20.log"
export LAB01_N_ACTION_STEPS=""
mkdir -p "$(dirname "$LAB01_EVAL_LOG")"
set +e
"$_LAB01_DIR/eval.cmd"
lingbot_eval_rc=$?
set -e

export LAB01_POLICY_PATH="$FASTWAM_CKPT"
export LAB01_EVAL_CONFIG="labs/lab01_pnp/configs/rollout_fastwam.yaml"
export LAB01_EVAL_LOG="$OUT_ROOT/eval/fastwam_egl20.log"
export LAB01_N_ACTION_STEPS=32
set +e
"$_LAB01_DIR/eval.cmd"
fastwam_eval_rc=$?
set -e

write_status "phase=complete lingbot_eval=$lingbot_eval_rc fastwam_eval=$fastwam_eval_rc ended=$(date -Is) lingbot_ckpt=$LINGBOT_CKPT fastwam_ckpt=$FASTWAM_CKPT"
echo "LingBot eval log: $OUT_ROOT/eval/lingbot_va_egl20.log"
echo "FastWAM eval log: $OUT_ROOT/eval/fastwam_egl20.log"
tail -20 "$OUT_ROOT/eval/lingbot_va_egl20.log" || true
tail -20 "$OUT_ROOT/eval/fastwam_egl20.log" || true
