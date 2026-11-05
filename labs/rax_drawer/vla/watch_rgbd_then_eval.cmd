#!/bin/bash
# Wait for MI300X ACT RGB-D 30k, pull ckpt to .13, eval RGBD (explore10 + test20).
# Skips RGB eval if place100d RGB JSONs already exist (run those in parallel earlier).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv-rocm/bin/python}"
LOG="$ROOT/outputs/logs/watch_rgbd_then_eval.log"
mkdir -p "$ROOT/outputs/logs" "$ROOT/outputs/eval"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] watch_rgbd_then_eval start"

RGB_CKPT="$ROOT/outputs/train/rax_drawer_act_rgb/checkpoints/030000/pretrained_model"
RGBD_REMOTE="/root/so101-simstudio-rax/outputs/train/rax_drawer_act_rgbd"
RGBD_LOCAL="$ROOT/outputs/train/rax_drawer_act_rgbd"
RGBD_CKPT="$RGBD_LOCAL/checkpoints/030000/pretrained_model"

if [[ ! -d "$RGB_CKPT" ]]; then
  echo "ERROR: local RGB 030000 missing: $RGB_CKPT"
  exit 1
fi

echo "Waiting for DORobot RGBD train to finish (030000 + process gone)..."
while true; do
  # Avoid pgrep self-match on the ssh command line: look for End of training + 030000.
  remote_state=$(ssh -o BatchMode=yes -o ConnectTimeout=20 DORobot \
    "step=\$(grep -oE '[0-9]+/30000' $RGBD_REMOTE.log 2>/dev/null | tail -1)
     done=0; grep -q 'End of training' $RGBD_REMOTE.log 2>/dev/null && done=1
     has=0; test -d $RGBD_REMOTE/checkpoints/030000/pretrained_model && has=1
     echo \"\$step done=\$done has030000=\$has\"" \
    2>/dev/null || echo "ssh_fail")
  echo "[$(date -Is)] $remote_state"
  if [[ "$remote_state" == *"done=1"* && "$remote_state" == *"has030000=1"* ]]; then
    break
  fi
  sleep 120
done

echo "[$(date -Is)] rsync RGBD checkpoints from DORobot"
mkdir -p "$RGBD_LOCAL/checkpoints"
rsync -aH --info=stats2 \
  "DORobot:$RGBD_REMOTE/checkpoints/" \
  "$RGBD_LOCAL/checkpoints/"
test -d "$RGBD_CKPT"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

run_eval() {
  local name="$1" policy="$2" split="$3" n="$4"
  local out="$ROOT/outputs/eval/${name}_${split}${n}.json"
  echo "[$(date -Is)] eval $name $split $n -> $out"
  "$PYTHON" -m labs.rax_drawer.vla.eval_policy \
    --policy "$policy" \
    --episodes "$n" \
    --split "$split" \
    --out "$out"
}

# RGB may already be evaluated in parallel while RGBD trains.
if [[ ! -f "$ROOT/outputs/eval/rax_drawer_act_rgb_place100d_explore10.json" ]]; then
  run_eval rax_drawer_act_rgb_place100d "$RGB_CKPT" explore 10
fi
if [[ ! -f "$ROOT/outputs/eval/rax_drawer_act_rgb_place100d_test20.json" ]]; then
  run_eval rax_drawer_act_rgb_place100d "$RGB_CKPT" test 20
fi
run_eval rax_drawer_act_rgbd_place100d "$RGBD_CKPT" explore 10
run_eval rax_drawer_act_rgbd_place100d "$RGBD_CKPT" test 20

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("outputs/eval")
rows = []
for name, label in [
    ("rax_drawer_act_rgb_place100d_explore10.json", "RGB explore10"),
    ("rax_drawer_act_rgb_place100d_test20.json", "RGB test20"),
    ("rax_drawer_act_rgbd_place100d_explore10.json", "RGBD explore10"),
    ("rax_drawer_act_rgbd_place100d_test20.json", "RGBD test20"),
]:
    p = root / name
    if not p.is_file():
        continue
    d = json.loads(p.read_text())
    rows.append({"label": label, "successes": d["successes"], "n": d["n"], "rate": d["rate"]})
    print(f"{label}: {d['successes']}/{d['n']} ({d['rate']:.0%})")

out = Path("outputs/eval/rax_drawer_act_rgb_vs_rgbd_summary.json")
out.write_text(json.dumps({"rows": rows}, indent=2))
print("wrote", out)
PY

echo "[$(date -Is)] watch_rgbd_then_eval done"
