#!/bin/bash
# Lab 01 — headless closed-loop Hz sweep (8060S vs MI300X).
#
# Wraps eval.cmd: fixed-cube YAMLs, 4 episodes, render_window=false.
# Logs (gitignore): ${LAB01_HZ_BENCH_LOG:-./outputs/eval/hz_bench}/
# Tables / FAQ: labs/lab01_pnp/hz_bench/COMPARISON.md
#
#   HOST=halo GL=egl ./labs/lab01_pnp/hz_bench.cmd
#   HOST=dorobot GL=osmesa ./labs/lab01_pnp/hz_bench.cmd
#   python3 labs/lab01_pnp/hz_bench/summarize.py
#
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/_env.sh"
cd "$_LAB01_DIR/../.."

HOST="${HOST:-halo}"
GL="${GL:-$LAB01_MUJOCO_GL}"
OUT="${LAB01_HZ_BENCH_LOG:-./outputs/eval/hz_bench}"
mkdir -p "$OUT"
SUMMARY="$OUT/${HOST}_${GL}.tsv"
echo -e "host\tpolicy\tckpt\tconfig\tn_action_steps\tgl\tlog" > "$SUMMARY"

run_one() {
    local name="$1" ckpt="$2" cfg="$3" nsteps="${4:-}"
    local log="$OUT/${HOST}_${GL}_${name}.log"
    if [[ ! -f "$ckpt/model.safetensors" ]]; then
        echo "SKIP $name missing $ckpt" | tee -a "$OUT/${HOST}_skip.txt"
        return 0
    fi
    echo "=== $HOST $name $(date -Is) ===" | tee -a "$OUT/${HOST}_progress.txt"
    env -u LAB01_POLICY_PATH \
      LAB01_MUJOCO_GL="$GL" \
      LAB01_RENDER_WINDOW=false \
      LAB01_EVAL_EPISODES=4 \
      LAB01_POLICY_PATH="$ckpt" \
      LAB01_EVAL_CONFIG="$cfg" \
      LAB01_EVAL_LOG="$log" \
      LAB01_N_ACTION_STEPS="${nsteps}" \
      ./labs/lab01_pnp/eval.cmd
    echo -e "${HOST}\t${name}\t${ckpt}\t${cfg}\t${nsteps:-yaml}\t${GL}\t${log}" >> "$SUMMARY"
}

# Native inference (sync vs RTC) stays in each YAML.
run_one act6 \
  "${ACT6:-./outputs/train/lab01_pnp_act_state6/checkpoints/050000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_act_demo_fixed.yaml 50

run_one act15 \
  "${ACT15:-./outputs/train/lab01_pnp_act/checkpoints/050000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_act_demo_fixed.yaml 50

run_one smolvla \
  "${SMOL:-./outputs/train/lab01_pnp_smolvla/checkpoints/050000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_smolvla_demo_fixed.yaml ""

run_one molmo6 \
  "${MOLMO6:-./outputs/train/lab01_pnp_molmoact2_state6/checkpoints/010000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_molmoact2_demo_fixed.yaml 30

run_one molmo15 \
  "${MOLMO15:-./outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_molmoact2_demo_fixed.yaml 30

run_one jepa \
  "${JEPA:-./outputs/train/lab01_pnp_vla_jepa_state6_bc/checkpoints/010000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_vla_jepa_demo_start.yaml 30

run_one groot \
  "${GROOT:-./outputs/train/lab01_pnp_groot_n17/checkpoints/010000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_groot_demo_fixed.yaml ""

run_one fastwam \
  "${FASTWAM:-./outputs/train/lab01_pnp_fastwam_pad7_unfreeze4/checkpoints/008000/pretrained_model}" \
  labs/lab01_pnp/configs/rollout_fastwam_demo_start.yaml 32

echo "=== DONE $HOST $(date -Is) ===" | tee -a "$OUT/${HOST}_progress.txt"
echo "Summarize: python3 labs/lab01_pnp/hz_bench/summarize.py"
echo "Read:      labs/lab01_pnp/hz_bench/COMPARISON.md"
