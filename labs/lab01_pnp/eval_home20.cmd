#!/bin/bash
# Lab01-PnP-Home20 — three seeds on the frozen 20 poses.
#
# Protocol: labs/lab01_pnp_agent/home20.md. Arm home, gripper start 0.85
# (the pose file's qpos), 20 episodes, 60 s, stats from the 193-episode set.
# Seeds 1000, 1001, 1002. Quote final_success.
#
# xVLA and DM0.5 need LAB01_LEROBOT_SRC (see _env.sh section 5f). The published
# evals used .venv-cuda and MUJOCO_GL=egl on one NVIDIA GPU:
#
#   LAB01_LEROBOT_SRC=/path/to/lerobot \
#   LAB01_PYTHON=.venv-cuda/bin/python \
#   LAB01_POLICY_PATH=./outputs/train/lab01_xvla_agent193_10k/checkpoints/010000/pretrained_model \
#   LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_xvla.yaml \
#   LAB01_N_ACTION_STEPS=30 \
#   LAB01_HOME20_TAG=xvla_agent193_10k \
#     ./labs/lab01_pnp/eval_home20.cmd
#
#   LAB01_POLICY_PATH=./outputs/train/lab01_dm05_agent193_20k/checkpoints/020000/pretrained_model \
#   LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_dm05.yaml \
#   LAB01_N_ACTION_STEPS=10 \
#   LAB01_HOME20_TAG=dm05_base_agent193_20k \
#     ./labs/lab01_pnp/eval_home20.cmd
#
# Existing logs are kept. Set LAB01_HOME20_OVERWRITE=1 to replace them.
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
_user_policy="${LAB01_POLICY_PATH:-}"
_user_config="${LAB01_EVAL_CONFIG:-}"
_user_steps="${LAB01_N_ACTION_STEPS:-}"
_user_tag="${LAB01_HOME20_TAG:-}"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
# shellcheck disable=SC1091
source "$_LAB01_DIR/../lab01_pnp_agent/_env.sh"
if [[ -n "${LAB01_PYTHON:-}" ]]; then
    PYTHON="$LAB01_PYTHON"
fi
LAB01_POLICY_PATH="$_user_policy"
LAB01_EVAL_CONFIG="$_user_config"
if [[ -n "$_user_steps" ]]; then
    LAB01_N_ACTION_STEPS="$_user_steps"
fi
LAB01_HOME20_TAG="$_user_tag"

if [[ -z "$LAB01_POLICY_PATH" || ! -d "$LAB01_POLICY_PATH" ]]; then
    echo "Set LAB01_POLICY_PATH to a pretrained_model/ directory." >&2
    exit 1
fi
if [[ -z "$LAB01_EVAL_CONFIG" || ! -f "$LAB01_EVAL_CONFIG" ]]; then
    echo "Set LAB01_EVAL_CONFIG to a rollout YAML (rollout_xvla.yaml or rollout_dm05.yaml)." >&2
    exit 1
fi
if [[ -z "${LAB01_HOME20_TAG}" ]]; then
    echo "Set LAB01_HOME20_TAG so logs do not overwrite an older run." >&2
    echo "Published tags: xvla_agent193_10k, dm05_base_agent193_20k." >&2
    exit 1
fi

export LAB01_POLICY_PATH LAB01_EVAL_CONFIG
export LAB01_DATASET_ROOT="$LAB01_AGENT193_DATASET_ROOT"
export LAB01_DATASET_REPO_ID="$LAB01_AGENT193_DATASET_REPO_ID"
export LAB01_EPISODE_INIT="${LAB01_EPISODE_INIT:-$LAB01_HOME20_SPAWNS}"
export LAB01_EVAL_EPISODES="${LAB01_EVAL_EPISODES:-20}"
export LAB01_RENDER_WINDOW="${LAB01_RENDER_WINDOW:-false}"
export LAB01_MUJOCO_GL="${LAB01_MUJOCO_GL:-egl}"

if [[ -z "${LAB01_N_ACTION_STEPS}" ]]; then
    case "$LAB01_EVAL_CONFIG" in
        *xvla*) export LAB01_N_ACTION_STEPS="$LAB01_XVLA_N_ACTION_STEPS" ;;
        *dm05*) export LAB01_N_ACTION_STEPS="$LAB01_DM05_N_ACTION_STEPS" ;;
        *)
            echo "Set LAB01_N_ACTION_STEPS for $LAB01_EVAL_CONFIG." >&2
            exit 1
            ;;
    esac
fi
export LAB01_N_ACTION_STEPS

if [[ ! -f "$LAB01_EPISODE_INIT" ]]; then
    echo "Home20 pose file not found: $LAB01_EPISODE_INIT" >&2
    exit 1
fi

echo "=== Lab01-PnP-Home20 ==="
echo "Policy:   $LAB01_POLICY_PATH"
echo "Config:   $LAB01_EVAL_CONFIG"
echo "Poses:    $LAB01_EPISODE_INIT"
echo "Dataset:  $LAB01_DATASET_ROOT"
echo "Seeds:    $LAB01_HOME20_SEEDS"
echo "n_action: $LAB01_N_ACTION_STEPS"
echo ""

for seed in $LAB01_HOME20_SEEDS; do
    elog="$REPO_ROOT/outputs/eval/${LAB01_HOME20_TAG}_lab01_pnp_home20_g085_seed${seed}.log"
    if [[ -e "$elog" && "${LAB01_HOME20_OVERWRITE:-}" != "1" ]]; then
        echo "Refusing to overwrite $elog" >&2
        echo "Set LAB01_HOME20_OVERWRITE=1, or choose another LAB01_HOME20_TAG." >&2
        exit 1
    fi
    export LAB01_EVAL_SEED="$seed"
    export LAB01_EVAL_LOG="$elog"
    export LAB01_EVAL_LOG_APPEND=1
    mkdir -p "$(dirname "$elog")"
    {
        echo "benchmark: Lab01-PnP-Home20"
        echo "gripper_qpos: 0.85"
        echo "seed: $seed"
        echo "checkpoint: $LAB01_POLICY_PATH"
        echo "started: $(date -Iseconds)"
    } > "$elog"
    set +e
    "$_LAB01_DIR/eval.cmd"
    status=$?
    set -e
    echo "exit: $status finished: $(date -Iseconds)" >> "$elog"
    echo "seed $seed exit $status  log $elog"
    if [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
done
