#!/bin/bash
# Lab 01 — sim2sim SmolVLA policy eval in MuJoCo (closed-loop inference).
#
# Prereq: trained checkpoint (Run 1 default below), .venv-rocm, validated dataset not required.
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/eval.cmd
#
# Options (pass through):
#   ./labs/lab01_pnp/eval.cmd --eval.num_episodes 5
#   LAB01_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp/eval.cmd
#   ./labs/lab01_pnp/eval.cmd --inference.type sync   # if RTC is unstable
set -euo pipefail
source "$(dirname "$0")/../../scripts/quicktest/_common.sh"

POLICY_PATH="${LAB01_POLICY_PATH:-./outputs/train/lab01_pnp_smolvla_bs4/checkpoints/007500/pretrained_model}"
EPISODES="${LAB01_EVAL_EPISODES:-10}"

if [[ ! -d "$POLICY_PATH" ]]; then
    echo "Checkpoint not found: $POLICY_PATH" >&2
    echo "Set LAB01_POLICY_PATH or train first: ./labs/lab01_pnp/train.cmd" >&2
    exit 1
fi

echo "=== Lab 01: sim2sim policy eval ==="
echo "Policy:   $POLICY_PATH"
echo "Config:   configs/so101_mujoco_rollout.yaml"
echo "Episodes: $EPISODES"
echo ""

"$PYTHON" -m simstudio.scripts.eval \
    --config configs/so101_mujoco_rollout.yaml \
    --policy.path="$POLICY_PATH" \
    --eval.num_episodes="$EPISODES" \
    "$@"
