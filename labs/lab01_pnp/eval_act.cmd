#!/bin/bash
# Lab 01 — ACT sim2sim policy eval in MuJoCo (closed-loop inference).
#
# Prereq: trained ACT checkpoint, .venv-rocm.
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/eval_act.cmd
#
# Options:
#   ./labs/lab01_pnp/eval_act.cmd --eval.num_episodes 5
#   LAB01_ACT_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp/eval_act.cmd
set -euo pipefail
source "$(dirname "$0")/../../scripts/quicktest/_common.sh"

OUTPUT_DIR="${LAB01_ACT_OUTPUT:-./outputs/train/lab01_pnp_act}"
POLICY_PATH="${LAB01_ACT_POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
    POLICY_PATH="$(ls -td "$OUTPUT_DIR"/checkpoints/*/pretrained_model 2>/dev/null | head -1 || true)"
fi
EPISODES="${LAB01_ACT_EVAL_EPISODES:-20}"

if [[ -z "$POLICY_PATH" || ! -d "$POLICY_PATH" ]]; then
    echo "ACT checkpoint not found under $OUTPUT_DIR" >&2
    echo "Train first: ./labs/lab01_pnp/train_act.cmd" >&2
    exit 1
fi

echo "=== Lab 01: ACT sim2sim policy eval ==="
echo "Policy:   $POLICY_PATH"
echo "Config:   configs/so101_mujoco_rollout_act.yaml"
echo "Episodes: $EPISODES (headless)"
echo ""

"$PYTHON" -m simstudio.scripts.eval \
    --config configs/so101_mujoco_rollout_act.yaml \
    --policy.path="$POLICY_PATH" \
    --eval.num_episodes="$EPISODES" \
    --robot.render_window=false \
    "$@"
