#!/bin/bash
# On halo: wait for DORobot ACT tmux to finish, rsync checkpoints, eval on EGL.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
source "$REPO/labs/rax_drawer/vla/_env.sh"
REMOTE="${RAX_REMOTE:-DORobot}"
REMOTE_REPO="${RAX_REMOTE_REPO:-/root/so101-simstudio-rax}"
SESSION="${RAX_TMUX_SESSION:-rax_act}"
OUT_LOCAL="$REPO/outputs/train/rax_drawer_act_state6"
LOG="$REPO/outputs/logs/rax_act_eval.log"
mkdir -p "$(dirname "$LOG")" "$OUT_LOCAL"

echo "Waiting for tmux $SESSION on $REMOTE to end"
while ssh -o ConnectTimeout=20 "$REMOTE" "tmux has-session -t $SESSION 2>/dev/null"; do
  echo "$(date -Is) training still running"
  sleep 120
done
echo "tmux $SESSION gone — rsync checkpoints"
rsync -a --info=progress2 \
  "$REMOTE:$REMOTE_REPO/outputs/train/rax_drawer_act_state6/" \
  "$OUT_LOCAL/"

CKPT=$(ls -td "$OUT_LOCAL"/checkpoints/*/pretrained_model 2>/dev/null | head -1 || true)
if [[ -z "${CKPT}" ]]; then
  echo "No checkpoint under $OUT_LOCAL" >&2
  exit 1
fi
echo "Eval $CKPT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$REPO/.venv-rocm/bin/python" -m labs.rax_drawer.vla.eval_policy \
  --policy "$CKPT" \
  --episodes "${RAX_EVAL_EPISODES:-20}" \
  --split "${RAX_EVAL_SPLIT:-test}" \
  --out "$REPO/outputs/eval/rax_drawer_act_state6.json" \
  2>&1 | tee "$LOG"

echo "ACT eval done. Starting SmolVLA train on DORobot..."
ssh "$REMOTE" "tmux has-session -t rax_smol 2>/dev/null && tmux kill-session -t rax_smol; \
  tmux new-session -d -s rax_smol \"bash -lc 'cd $REMOTE_REPO && \
    export PYTHONPATH=$REMOTE_REPO:\\\$PYTHONPATH && \
    source .venv-rocm/bin/activate && \
    RAX_SMOL_BATCH_SIZE=${RAX_SMOL_BATCH_SIZE:-32} RAX_SMOL_STEPS=${RAX_SMOL_STEPS:-20000} \
    ./labs/rax_drawer/vla/train_smolvla.cmd 2>&1 | tee outputs/train/rax_drawer_smolvla_mi300x.log'\""
echo "tmux rax_smol started"
