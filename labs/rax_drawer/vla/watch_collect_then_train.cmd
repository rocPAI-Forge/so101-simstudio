#!/bin/bash
# On halo: wait until place100 has TARGET successes, rsync to DORobot, start ACT train in tmux.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
TARGET="${RAX_TARGET:-100}"
ROOT="${RAX_DATASET_ROOT:-./datasets/so101-simstudio-rax-drawer-place100}"
REMOTE="${RAX_REMOTE:-DORobot}"
REMOTE_REPO="${RAX_REMOTE_REPO:-/root/so101-simstudio-rax}"
SPAWN="$ROOT/spawns.jsonl"

count_ok() {
  python3 - <<PY
import json, pathlib
p=pathlib.Path("$SPAWN")
if not p.exists():
  print(0); raise SystemExit
print(sum(1 for l in p.read_text().splitlines() if l.strip() and json.loads(l).get("success")))
PY
}

echo "Waiting for $TARGET successes in $SPAWN"
while true; do
  n=$(count_ok)
  echo "$(date -Is) successes=$n/$TARGET"
  if [[ "$n" -ge "$TARGET" ]]; then
    break
  fi
  if ! pgrep -f 'labs.rax_drawer.collect_rax_place' >/dev/null; then
    echo "collector exited early with $n successes" >&2
    if [[ "$n" -lt "$TARGET" ]]; then
      exit 1
    fi
    break
  fi
  sleep 60
done

echo "Rsync dataset → $REMOTE:$REMOTE_REPO"
rsync -a --info=progress2 "$ROOT/" "$REMOTE:$REMOTE_REPO/datasets/$(basename "$ROOT")/"
rsync -a "$REPO/labs/rax_drawer/" "$REMOTE:$REMOTE_REPO/labs/rax_drawer/"

echo "Start ACT train on $REMOTE (tmux rax_act)"
ssh "$REMOTE" "tmux has-session -t rax_act 2>/dev/null && tmux kill-session -t rax_act; \
  tmux new-session -d -s rax_act \"bash -lc 'cd $REMOTE_REPO && \
    export PYTHONPATH=$REMOTE_REPO:\\\$PYTHONPATH && \
    source .venv-rocm/bin/activate && \
    RAX_ACT_BATCH_SIZE=${RAX_ACT_BATCH_SIZE:-64} RAX_ACT_STEPS=${RAX_ACT_STEPS:-30000} \
    ./labs/rax_drawer/vla/train_act.cmd 2>&1 | tee outputs/train/rax_drawer_act_state6_mi300x.log'\""
echo "tmux rax_act started. Attach: ssh $REMOTE -t 'tmux attach -t rax_act'"
