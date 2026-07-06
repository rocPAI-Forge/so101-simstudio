#!/bin/bash
# Leader arm 录制测试脚本
# 用法: ./test_leader_record.sh [episodes] [resume]

set -e

EPISODES=${1:-3}
RESUME=${2:-false}
ROOT="./datasets/leader-record-test"
REPO_ID="alexhegit/so101_mujoco_leader_record_test"

echo "=== Leader Arm Record Test ==="
echo "Episodes: $EPISODES"
echo "Resume: $RESUME"
echo "Dataset: $ROOT"
echo ""
echo "Controls:"
echo "  Move leader arm to control robot"
echo "  Ctrl+C: Stop recording"
echo ""

.venv-rocm/bin/python -m simstudio.scripts.record \
    --config configs/so101_mujoco_leader.yaml \
    --dataset.root "$ROOT" \
    --dataset.repo_id "$REPO_ID" \
    --dataset.num_episodes "$EPISODES" \
    --resume "$RESUME"
