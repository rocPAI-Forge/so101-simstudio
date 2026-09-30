#!/usr/bin/env bash
# Sequential Lab 01 agent-yaw campaign on DORobot. Idempotent via tmp/p0923_done/*.
# Source of truth: tmp/lab01-agent-p0923.md
set -euo pipefail

REPO="${REPO:-/root/so101-simstudio}"
cd "$REPO"
# shellcheck disable=SC1091
source "$REPO/.venv-rocm/bin/activate"

DS_NAME="so101-simstudio-lab01-pnp-agent-yaw"
DS_ROOT="/mnt/doscratch/datasets/${DS_NAME}"
# Set repo_id *before* any lab cmd sources _env.sh (otherwise the :- default sticks).
export LAB01_DATASET_NAME="$DS_NAME"
export LAB01_DATASET_ROOT="$DS_ROOT"
export LAB01_DATASET_REPO_ID="alexhegit/${DS_NAME}"
EVAL_DIR="/mnt/doscratch/outputs/eval"
TRAIN_DIR="/mnt/doscratch/outputs/train"
MARK="$REPO/tmp/p0923_done"
PLAN="$REPO/tmp/lab01-agent-p0923.md"
HF_EP="https://huggingface.co"
HF_USER="${LAB01_HF_USER:-alexhegit}"
LOG="$REPO/tmp/lab01-agent-p0923.driver.log"

mkdir -p "$MARK" "$EVAL_DIR" "$TRAIN_DIR" "$REPO/tmp"

stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

note() {
  echo "[$(stamp)] $*" | tee -a "$LOG"
}

mark_plan() {
  local msg="$1"
  python3 - "$PLAN" "$msg" <<'PY'
import pathlib, sys, datetime
p = pathlib.Path(sys.argv[1])
msg = sys.argv[2]
text = p.read_text()
old = None
for line in text.splitlines():
    if line.startswith("| Last update |"):
        old = line
        break
if old:
    now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    new = f"| Last update | {now} — {msg} |"
    p.write_text(text.replace(old, new, 1))
PY
}

need_dataset() {
  if [[ ! -f "$DS_ROOT/meta/info.json" ]]; then
    note "FATAL: dataset missing at $DS_ROOT"
    exit 1
  fi
}

export LAB01_RENDER_WINDOW=false
export LAB01_MUJOCO_GL="${LAB01_MUJOCO_GL:-egl}"
unset DISPLAY || true

run_eval() {
  local policy="$1" cfg="$2" nsteps="$3" n_ep="$4" logfile="$5"
  shift 5 || true
  note "eval policy=$policy config=$cfg n_action_steps=$nsteps n=$n_ep gl=$LAB01_MUJOCO_GL extra=$*"
  set +e
  LAB01_POLICY_PATH="$policy" \
  LAB01_EVAL_CONFIG="$cfg" \
  LAB01_N_ACTION_STEPS="$nsteps" \
  LAB01_EVAL_EPISODES="$n_ep" \
  LAB01_EVAL_LOG="$logfile" \
    "$REPO/labs/lab01_pnp/eval.cmd" "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    if [[ "${LAB01_MUJOCO_GL}" == "egl" ]]; then
      note "EGL eval failed rc=$rc — retrying OSMesa (tick-budgeted)"
      export LAB01_MUJOCO_GL=osmesa
      LAB01_POLICY_PATH="$policy" \
      LAB01_EVAL_CONFIG="$cfg" \
      LAB01_N_ACTION_STEPS="$nsteps" \
      LAB01_EVAL_EPISODES="$n_ep" \
      LAB01_EVAL_LOG="${logfile%.log}_osmesa.log" \
        "$REPO/labs/lab01_pnp/eval.cmd" "$@"
    else
      note "eval failed rc=$rc"
      return $rc
    fi
  fi
}

hf_upload_dir() {
  local repo_id="$1" local_dir="$2" msg="$3"
  export LAB01_HF_UPLOAD_ENDPOINT="$HF_EP"
  note "hf repo create $repo_id"
  hf repo create "$repo_id" --type model --exist-ok || true
  note "hf upload $local_dir -> $repo_id"
  hf upload "$repo_id" "$local_dir" . --repo-type model --commit-message "$msg"
}

need_dataset
note "===== p0923 driver start cwd=$REPO gl=$LAB01_MUJOCO_GL ====="

# --- 1a ACT eval ---
ACT_CKPT="$TRAIN_DIR/lab01_pnp_act_agent_yaw/checkpoints/080000/pretrained_model"
if [[ ! -f "$MARK/1a_act_eval" ]]; then
  if [[ ! -d "$ACT_CKPT" ]]; then
    note "FATAL: ACT 080000 missing: $ACT_CKPT"
    exit 1
  fi
  mark_plan "stage 1a ACT n=50 eval running"
  run_eval "$ACT_CKPT" "labs/lab01_pnp/configs/rollout_act.yaml" 50 50 \
    "$EVAL_DIR/act_yaw_n100_080000_n50.log"
  grep -E 'Eval complete:|Stage funnel|INFO.*eval' "$EVAL_DIR/act_yaw_n100_080000_n50.log" \
    "$EVAL_DIR/act_yaw_n100_080000_n50_osmesa.log" 2>/dev/null | tee -a "$LOG" || true
  touch "$MARK/1a_act_eval"
  mark_plan "stage 1a ACT eval DONE"
fi

# --- 1b ACT HF ---
if [[ ! -f "$MARK/1b_act_hf" ]]; then
  mark_plan "stage 1b ACT HF upload"
  hf_upload_dir "${HF_USER}/so101-simstudio-lab01-pnp-act-agent-yaw" "$ACT_CKPT" \
    "ACT 15-D 80K bs64 on agent-yaw 100 ep"
  touch "$MARK/1b_act_hf"
  mark_plan "stage 1b ACT HF DONE"
fi

# --- 2 SmolVLA ---
SMOL_OUT="$TRAIN_DIR/lab01_pnp_smolvla_agent_yaw"
SMOL_CKPT="$SMOL_OUT/checkpoints/050000/pretrained_model"
if [[ ! -f "$MARK/2a_smolvla_train" ]]; then
  mark_plan "stage 2a SmolVLA 50K train"
  LAB01_TRAIN_OUTPUT="$SMOL_OUT" \
  LAB01_TRAIN_BATCH_SIZE=64 LAB01_TRAIN_STEPS=50000 \
  LAB01_TRAIN_SAVE_FREQ=10000 LAB01_TRAIN_NUM_WORKERS=8 \
    "$REPO/labs/lab01_pnp/train.cmd"
  cp -f "$REPO/train.log" "$TRAIN_DIR/../train_smolvla_agent_yaw.log" 2>/dev/null || true
  cp -f "$REPO/train.log" /mnt/doscratch/outputs/train_smolvla_agent_yaw.log 2>/dev/null || true
  touch "$MARK/2a_smolvla_train"
  mark_plan "stage 2a SmolVLA train DONE"
fi
if [[ ! -f "$MARK/2b_smolvla_eval" ]]; then
  mark_plan "stage 2b SmolVLA n=50 eval"
  run_eval "$SMOL_CKPT" "labs/lab01_pnp/configs/rollout_smolvla.yaml" 50 50 \
    "$EVAL_DIR/smolvla_yaw_n100_050000_n50.log"
  touch "$MARK/2b_smolvla_eval"
  mark_plan "stage 2b SmolVLA eval DONE"
fi
if [[ ! -f "$MARK/2c_smolvla_hf" ]]; then
  mark_plan "stage 2c SmolVLA HF upload"
  export LAB01_HF_UPLOAD_ENDPOINT="$HF_EP"
  LAB01_SMOLVLA_HF_REPO_ID="${HF_USER}/so101-simstudio-lab01-pnp-smolvla-agent-yaw" \
  LAB01_POLICY_PATH="$SMOL_CKPT" \
    "$REPO/labs/lab01_pnp/push_smolvla_model.cmd" || \
    hf_upload_dir "${HF_USER}/so101-simstudio-lab01-pnp-smolvla-agent-yaw" "$SMOL_CKPT" \
      "SmolVLA 50K bs64 on agent-yaw 100 ep"
  touch "$MARK/2c_smolvla_hf"
  mark_plan "stage 2c SmolVLA HF DONE"
fi

# --- 3 JEPA ---
JEPA_OUT="$TRAIN_DIR/lab01_pnp_vla_jepa_agent_yaw"
JEPA_CKPT="$JEPA_OUT/checkpoints/010000/pretrained_model"
if [[ ! -f "$MARK/3a_jepa_train" ]]; then
  mark_plan "stage 3a JEPA 10K freeze-Qwen train"
  LAB01_JEPA_OUTPUT="$JEPA_OUT" \
  LAB01_STATE_DIM=6 \
  LAB01_JEPA_STEPS=10000 LAB01_JEPA_BATCH_SIZE=16 LAB01_JEPA_SAVE_FREQ=2000 \
  LAB01_JEPA_CHUNK_SIZE=30 LAB01_JEPA_N_ACTION_STEPS=30 \
  LAB01_JEPA_FREEZE_QWEN=true LAB01_JEPA_ENABLE_WORLD_MODEL=false \
    "$REPO/labs/lab01_pnp/train_vla_jepa.cmd"
  touch "$MARK/3a_jepa_train"
  mark_plan "stage 3a JEPA train DONE"
fi
if [[ ! -f "$MARK/3b_jepa_eval" ]]; then
  mark_plan "stage 3b JEPA n=50 eval"
  export LAB01_STATE_DIM=6
  run_eval "$JEPA_CKPT" "labs/lab01_pnp/configs/rollout_vla_jepa_demo_start.yaml" 30 50 \
    "$EVAL_DIR/jepa_yaw_n100_010000_n50.log" \
    --robot.reset_cube=random \
    --robot.cube_random_x_range='[0.26,0.34]' \
    --robot.cube_random_y_range='[0.165,0.235]' \
    --robot.cube_random_z=0.0125 \
    --robot.cube_random_yaw_range='[-45.0,45.0]'
  touch "$MARK/3b_jepa_eval"
  mark_plan "stage 3b JEPA eval DONE"
fi
if [[ ! -f "$MARK/3c_jepa_hf" ]]; then
  mark_plan "stage 3c JEPA HF upload"
  hf_upload_dir "${HF_USER}/so101-simstudio-lab01-pnp-vla-jepa-agent-yaw" "$JEPA_CKPT" \
    "VLA-JEPA freeze-Qwen WM-off chunk30 10K on agent-yaw 100 ep"
  touch "$MARK/3c_jepa_hf"
  mark_plan "stage 3c JEPA HF DONE"
fi

# --- 4 GR00T ---
GROOT_OUT="$TRAIN_DIR/lab01_pnp_groot_n17_agent_yaw"
GROOT_CKPT="$GROOT_OUT/checkpoints/010000/pretrained_model"
if [[ ! -f "$MARK/4a_groot_train" ]]; then
  mark_plan "stage 4a GR00T N1.7 10K train"
  "$REPO/scripts/install-groot-deps.sh" || true
  LAB01_GROOT_OUTPUT="$GROOT_OUT" \
  LAB01_GROOT_STEPS=10000 LAB01_GROOT_BATCH_SIZE=4 LAB01_GROOT_SAVE_FREQ=2000 \
  LAB01_GROOT_NUM_WORKERS=4 \
    "$REPO/labs/lab01_pnp/train_groot.cmd"
  touch "$MARK/4a_groot_train"
  mark_plan "stage 4a GR00T train DONE"
fi
if [[ ! -f "$MARK/4b_groot_eval" ]]; then
  mark_plan "stage 4b GR00T n=50 eval"
  run_eval "$GROOT_CKPT" "labs/lab01_pnp/configs/rollout_groot.yaml" 16 50 \
    "$EVAL_DIR/groot_yaw_n100_010000_n50.log"
  touch "$MARK/4b_groot_eval"
  mark_plan "stage 4b GR00T eval DONE"
fi
if [[ ! -f "$MARK/4c_groot_hf" ]]; then
  mark_plan "stage 4c GR00T HF upload"
  hf_upload_dir "${HF_USER}/so101-simstudio-lab01-pnp-groot-n17-agent-yaw" "$GROOT_CKPT" \
    "GR00T N1.7-3B new_embodiment 10K bs4 on agent-yaw 100 ep"
  touch "$MARK/4c_groot_hf"
  mark_plan "stage 4c GR00T HF DONE"
fi

# --- 5 analysis stub: extract success lines into PLAN ---
if [[ ! -f "$MARK/5_analysis" ]]; then
  mark_plan "stage 5 extracting eval lines"
  {
    echo
    echo "## Auto-extracted eval headlines ($(stamp))"
    echo '```'
    grep -hE 'Eval complete:|Stage funnel' \
      "$EVAL_DIR/act_yaw_n100_080000_n50.log" \
      "$EVAL_DIR/act_yaw_n100_080000_n50_osmesa.log" \
      "$EVAL_DIR/smolvla_yaw_n100_050000_n50.log" \
      "$EVAL_DIR/smolvla_yaw_n100_050000_n50_osmesa.log" \
      "$EVAL_DIR/jepa_yaw_n100_010000_n50.log" \
      "$EVAL_DIR/jepa_yaw_n100_010000_n50_osmesa.log" \
      "$EVAL_DIR/groot_yaw_n100_010000_n50.log" \
      "$EVAL_DIR/groot_yaw_n100_010000_n50_osmesa.log" \
      2>/dev/null || true
    echo '```'
    echo
    echo "Fill the agent-yaw Success / Funnel cells in the table above from these lines. Compare to teleop 32/50 ACT, 11/50 SmolVLA, 1/10 JEPA (fixed), 0/10 GR00T."
  } >> "$PLAN"
  touch "$MARK/5_analysis"
  mark_plan "stage 5 analysis appendix written — human/agent should fill table"
fi

note "===== p0923 driver ALL STAGES COMPLETE ====="
