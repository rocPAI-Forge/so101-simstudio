#!/bin/bash
# Lab01-PnP-Home20 for LingBot-VLA 2.0 (official 25/60 path).
# Prerequisites:
#   ./labs/lab01_pnp/setup_lingbot_vla_v2_side_tree.cmd
#   Weights at LAB01_V2_POLICY_PATH (local or hf download — see lingbot_vla_v2_adapt.md)
#
#   ./labs/lab01_pnp/eval_lingbot_vla_v2_home20.cmd
#   LAB01_V2_HOME20_SEEDS="1000" ./labs/lab01_pnp/eval_lingbot_vla_v2_home20.cmd   # smoke one seed
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/../lab01_pnp_agent/_env.sh"

SIDE_ROOT="${LAB01_V2_SIDE_ROOT:-/tmp/lerobot_vla_v2_src}"
SIDE_SRC="${LAB01_V2_SIDE_SRC:-$SIDE_ROOT/src}"
POLICY="${LAB01_V2_POLICY_PATH}"
SEEDS="${LAB01_V2_HOME20_SEEDS:-1000 1001 1002}"
TAG="${LAB01_V2_EVAL_TAG:-mi210_unfreeze20k_slotfix}"
export PYTHONPATH="${SIDE_SRC}:${PYTHONPATH:-}"
export LAB01_EPISODE_INIT="${LAB01_EPISODE_INIT:-$LAB01_HOME20_SPAWNS}"
export LAB01_EVAL_CONFIG="${LAB01_EVAL_CONFIG:-labs/lab01_pnp/configs/rollout_lingbot_vla_v2.yaml}"
export LAB01_POLICY_PATH="$POLICY"
export LAB01_EVAL_EPISODES="${LAB01_EVAL_EPISODES:-20}"
export LAB01_MUJOCO_GL="${LAB01_MUJOCO_GL:-${MUJOCO_GL:-egl}}"
export LAB01_RENDER_WINDOW="${LAB01_RENDER_WINDOW:-false}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"

[[ -d "$SIDE_SRC" ]] || {
    echo "Missing side tree src: $SIDE_SRC" >&2
    echo "Run: ./labs/lab01_pnp/setup_lingbot_vla_v2_side_tree.cmd" >&2
    exit 1
}
[[ -f "$POLICY/model.safetensors" || -f "$POLICY/config.json" ]] || {
    echo "Missing policy: $POLICY" >&2
    echo "Download: hf download ${LAB01_V2_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k} --local-dir $POLICY" >&2
    exit 1
}

"$PYTHON" -c "import lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2" || {
    echo "Cannot import lingbot_vla_v2 from PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}

echo "=== LingBot-VLA 2.0 Home20 ==="
echo "Policy: $POLICY"
echo "Side:   $SIDE_SRC"
echo "Spawns: $LAB01_EPISODE_INIT"
echo "GL:     $LAB01_MUJOCO_GL"
echo "Seeds:  $SEEDS"
echo "Note:   default sampler (do not set LAB01_FLOW_* for the ranked score)"
echo ""

mkdir -p outputs/eval
for SEED in $SEEDS; do
    export LAB01_EVAL_SEED="$SEED"
    export LAB01_EVAL_LOG="${LAB01_EVAL_LOG_PREFIX:-./outputs/eval/lingbot_vla_v2_${TAG}_home20_g085_seed${SEED}.log}"
    echo "[$(date -Is)] seed=${SEED} log=${LAB01_EVAL_LOG}"
    "$_LAB01_DIR/eval.cmd" \
        --robot.reset_arm=home \
        --robot.home_gripper=0.85
done

echo ""
echo "Home20 campaign done. Quote final_success from the three logs (official = sum/60)."
