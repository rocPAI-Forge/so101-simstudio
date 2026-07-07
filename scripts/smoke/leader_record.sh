#!/bin/bash
# Leader arm recording smoke test.
# Usage: leader_record.sh [episodes] [resume] [view_mode]

set -e
source "$(dirname "$0")/_common.sh"

EPISODES=${1:-3}
RESUME=${2:-false}
VIEW_MODE=${3:-mujoco}
ROOT="./datasets/leader-record-test"
REPO_ID="alexhegit/so101_mujoco_leader_record_test"

echo "=== Leader Arm Record Smoke Test ==="
echo "Episodes: $EPISODES"
echo "Resume: $RESUME"
echo "View mode: $VIEW_MODE"
echo "Dataset: $ROOT"
echo ""
echo "Controls:"
echo "  Move leader arm to control robot"
echo "  n / Right arrow: Save episode (when terminal focused)"
echo "  r: Rerecord episode"
echo "  q / ESC: Stop recording"
echo ""

"$PYTHON" -m simstudio.scripts.record \
    --config configs/so101_mujoco_leader.yaml \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME" \
    --view_mode "$VIEW_MODE"
