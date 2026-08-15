#!/bin/bash
# Lab 01 — ACT sim2sim policy eval in MuJoCo.
#
# Defaults: labs/lab01_pnp/_env.sh (override via LAB01_* env vars).
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/eval_act.cmd
#
# Backend (MUJOCO_GL / LAB01_MUJOCO_GL):
#   egl   — headless GPU (recommended for batch eval)
#   glfw  — MuJoCo window (needs DISPLAY)
#   osmesa — CPU headless (slow fallback)
#
# Examples:
#   LAB01_MUJOCO_GL=egl LAB01_ACT_EVAL_EPISODES=50 ./labs/lab01_pnp/eval_act.cmd
#   LAB01_ACT_N_ACTION_STEPS=50 ./labs/lab01_pnp/eval_act.cmd
#   LAB01_ACT_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp/eval_act.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

# Resolve MuJoCo GL backend
if [[ -n "${LAB01_MUJOCO_GL:-}" ]]; then
    export MUJOCO_GL="$LAB01_MUJOCO_GL"
elif [[ -z "${MUJOCO_GL:-}" ]]; then
    if [[ -n "${DISPLAY:-}" ]]; then
        export MUJOCO_GL=glfw
    else
        export MUJOCO_GL=egl
    fi
fi
if [[ "$MUJOCO_GL" == "egl" || "$MUJOCO_GL" == "osmesa" ]]; then
    unset DISPLAY || true
    RENDER_WINDOW=false
else
    RENDER_WINDOW=true
fi

CKPT_STEP="${LAB01_ACT_CKPT_STEP:-050000}"
POLICY_PATH="${LAB01_ACT_POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
    for candidate in \
        "$LAB01_ACT_OUTPUT/checkpoints/${CKPT_STEP}/pretrained_model" \
        "$LAB01_ACT_OUTPUT/checkpoints/last/pretrained_model"; do
        if [[ -d "$candidate" ]]; then
            POLICY_PATH="$candidate"
            break
        fi
    done
fi

if [[ -z "$POLICY_PATH" || ! -d "$POLICY_PATH" ]]; then
    echo "ACT checkpoint not found under $LAB01_ACT_OUTPUT" >&2
    echo "Expected: $LAB01_ACT_OUTPUT/checkpoints/${CKPT_STEP}/pretrained_model" >&2
    echo "Train first: ./labs/lab01_pnp/train_act.cmd" >&2
    echo "Or set LAB01_ACT_POLICY_PATH=./outputs/train/.../pretrained_model" >&2
    exit 1
fi

EVAL_LOG="$LAB01_ACT_OUTPUT/eval_act_${CKPT_STEP}_${MUJOCO_GL}.log"
mkdir -p "$LAB01_ACT_OUTPUT"

echo "=== Lab 01: ACT sim2sim eval ==="
echo "MUJOCO_GL: $MUJOCO_GL"
echo "DISPLAY:   ${DISPLAY:-<unset>}"
echo "Policy:    $POLICY_PATH"
echo "Dataset:   $LAB01_DATASET_ROOT  (normalizer stats)"
echo "Config:    configs/so101_mujoco_rollout_act.yaml"
echo "Episodes:  $LAB01_ACT_EVAL_EPISODES"
echo "n_action_steps: $LAB01_ACT_N_ACTION_STEPS"
echo "Log:       $EVAL_LOG"
echo ""

"$PYTHON" -m simstudio.scripts.eval \
    --config configs/so101_mujoco_rollout_act.yaml \
    --policy.path="$POLICY_PATH" \
    --eval.num_episodes="$LAB01_ACT_EVAL_EPISODES" \
    --eval.stats_dataset_repo_id="$LAB01_DATASET_REPO_ID" \
    --eval.stats_dataset_root="$LAB01_DATASET_ROOT" \
    --policy.n_action_steps="$LAB01_ACT_N_ACTION_STEPS" \
    --robot.render_window="$RENDER_WINDOW" \
    "$@" 2>&1 | tee "$EVAL_LOG"
exit "${PIPESTATUS[0]}"
