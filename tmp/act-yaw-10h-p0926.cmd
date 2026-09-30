#!/usr/bin/env bash
# Ten-hour ACT agent-yaw diagnosis on DORobot.
set -euo pipefail

REPO=/root/so101-simstudio
cd "$REPO"
source .venv-rocm/bin/activate

export LAB01_DATASET_NAME=so101-simstudio-lab01-pnp-agent-yaw
export LAB01_DATASET_ROOT=/mnt/doscratch/datasets/so101-simstudio-lab01-pnp-agent-yaw
export LAB01_DATASET_REPO_ID=alexhegit/so101-simstudio-lab01-pnp-agent-yaw
export LAB01_POLICY_PATH=/mnt/doscratch/outputs/train/lab01_pnp_act_agent_yaw/checkpoints/080000/pretrained_model
export LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_act.yaml
export LAB01_MUJOCO_GL=osmesa
export LAB01_RENDER_WINDOW=false
unset DISPLAY || true

OUT=/mnt/doscratch/outputs/eval/act_yaw_ablation_p0926
mkdir -p "$OUT"

run() {
    local name="$1" arm="$2" nsteps="$3" episodes="$4"
    local log="$OUT/${name}.log"
    if grep -q "Eval complete:" "$log" 2>/dev/null; then
        echo "SKIP completed: $name"
        return
    fi
    LAB01_N_ACTION_STEPS="$nsteps" \
    LAB01_EVAL_EPISODES="$episodes" \
    LAB01_EVAL_LOG="$log" \
        ./labs/lab01_pnp/eval.cmd --robot.reset_arm="$arm"
}

# Existing reference: follow / n_action_steps=50 / n=50 = 9/50.
# Isolate record-vs-eval arm-start mismatch first.
run home_a50_n20 home 50 20
# Then test whether shorter closed-loop replanning helps under the matched start.
run home_a10_n20 home 10 20
# Small cross-check separates chunk effect from the home reset effect if time permits.
run follow_a10_n10 follow 10 10

{
    date -Is
    grep -hE "Eval complete:|Stage funnel" "$OUT"/*.log
} > "$OUT/SUMMARY.txt"
