#!/bin/bash
# Lab 01 — ACT sim2sim policy eval in MuJoCo.
#
# Defaults: labs/lab01_pnp/_env.sh (override via LAB01_* env vars).
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/eval_act.cmd
#
# Options:
#   ./labs/lab01_pnp/eval_act.cmd --eval.num_episodes 5
#   LAB01_ACT_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp/eval_act.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

POLICY_PATH="${LAB01_ACT_POLICY_PATH:-}"
if [[ -z "$POLICY_PATH" ]]; then
    POLICY_PATH="$(ls -td "$LAB01_ACT_OUTPUT"/checkpoints/*/pretrained_model 2>/dev/null | head -1 || true)"
fi

if [[ -z "$POLICY_PATH" || ! -d "$POLICY_PATH" ]]; then
    echo "ACT checkpoint not found under $LAB01_ACT_OUTPUT" >&2
    echo "Train first: ./labs/lab01_pnp/train_act.cmd" >&2
    exit 1
fi

echo "=== Lab 01: ACT sim2sim policy eval ==="
echo "Policy:   $POLICY_PATH"
echo "Dataset:  $LAB01_DATASET_ROOT  (normalizer stats)"
echo "Config:   configs/so101_mujoco_rollout_act.yaml"
echo "Episodes: $LAB01_ACT_EVAL_EPISODES (headless)"
echo ""

"$PYTHON" -m simstudio.scripts.eval \
    --config configs/so101_mujoco_rollout_act.yaml \
    --policy.path="$POLICY_PATH" \
    --eval.num_episodes="$LAB01_ACT_EVAL_EPISODES" \
    --eval.stats_dataset_repo_id="$LAB01_DATASET_REPO_ID" \
    --eval.stats_dataset_root="$LAB01_DATASET_ROOT" \
    --robot.render_window=false \
    "$@"
