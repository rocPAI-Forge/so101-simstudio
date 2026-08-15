#!/bin/bash
# Lab 01 — sim2sim policy eval in MuJoCo (any LeRobot policy).
#
# All knobs: labs/lab01_pnp/_env.sh (categorized: Shared / Record / Train / Eval / Hub).
# Point LAB01_POLICY_PATH at a pretrained_model/ dir and LAB01_EVAL_CONFIG at a
# matching lab YAML under labs/lab01_pnp/configs/ (SmolVLA needs rename_map;
# ACT must use rollout_act*.yaml without that map).
#
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/eval.cmd
#
#   # SmolVLA full-range
#   LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_smolvla.yaml \
#     LAB01_EVAL_EPISODES=50 ./labs/lab01_pnp/eval.cmd
#
#   # ACT
#   LAB01_POLICY_PATH=./outputs/train/lab01_pnp_act/checkpoints/last/pretrained_model \
#   LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml \
#   LAB01_N_ACTION_STEPS=50 LAB01_EVAL_EPISODES=50 \
#   LAB01_EVAL_LOG=./outputs/eval/act_egl.log \
#     ./labs/lab01_pnp/eval.cmd
#
#   LAB01_MUJOCO_GL=glfw LAB01_RENDER_WINDOW=true ./labs/lab01_pnp/eval.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

export MUJOCO_GL="$LAB01_MUJOCO_GL"
# Headless backends must not see a DISPLAY or MuJoCo may pick GLX/GLFW instead.
[[ "$MUJOCO_GL" == "glfw" ]] || unset DISPLAY

if [[ ! -d "$LAB01_POLICY_PATH" ]]; then
    echo "Policy not found: $LAB01_POLICY_PATH" >&2
    echo "Set LAB01_POLICY_PATH to a LeRobot pretrained_model/ directory." >&2
    exit 1
fi

mkdir -p "$(dirname "$LAB01_EVAL_LOG")"

EXTRA_ARGS=()
if [[ -n "${LAB01_N_ACTION_STEPS}" ]]; then
    EXTRA_ARGS+=(--policy.n_action_steps="$LAB01_N_ACTION_STEPS")
fi

echo "=== Lab 01: sim2sim policy eval ==="
echo "MUJOCO_GL:         $MUJOCO_GL"
echo "render_window:     $LAB01_RENDER_WINDOW"
echo "DISPLAY:           ${DISPLAY:-<unset>}"
echo "Policy:            $LAB01_POLICY_PATH"
echo "Config:            $LAB01_EVAL_CONFIG"
echo "Episodes:          $LAB01_EVAL_EPISODES"
echo "n_action_steps:    ${LAB01_N_ACTION_STEPS:-<checkpoint default>}"
echo "Stats dataset:     $LAB01_DATASET_ROOT"
echo "Log:               $LAB01_EVAL_LOG"
echo ""

"$PYTHON" -m simstudio.scripts.eval \
    --config "$LAB01_EVAL_CONFIG" \
    --policy.path="$LAB01_POLICY_PATH" \
    --eval.num_episodes="$LAB01_EVAL_EPISODES" \
    --eval.stats_dataset_repo_id="$LAB01_DATASET_REPO_ID" \
    --eval.stats_dataset_root="$LAB01_DATASET_ROOT" \
    --robot.render_window="$LAB01_RENDER_WINDOW" \
    "${EXTRA_ARGS[@]}" \
    "$@" 2>&1 | tee "$LAB01_EVAL_LOG"
exit "${PIPESTATUS[0]}"
