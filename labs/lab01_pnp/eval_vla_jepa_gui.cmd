#!/bin/bash
# Lab 01 — one-click MuJoCo GLFW eval for local VLA-JEPA checkpoints.
#
# Opens the robot viewer (not headless). Default: 6-D BC 10K (best measured
# JEPA, 1/10) from leader-demo mean start, chunk 30, 3 episodes.
#
# Usage (graphical session, repo root):
#   source .venv-rocm/bin/activate
#   ./labs/lab01_pnp/eval_vla_jepa_gui.cmd
#
#   LAB01_JEPA_EVAL_VARIANT=acthead ./labs/lab01_pnp/eval_vla_jepa_gui.cmd
#   LAB01_JEPA_EVAL_START=home LAB01_EVAL_EPISODES=10 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd
#
# Variants: bc10k (default) | bc20k | acthead | wm | 15d
# Start:    demo (demo-mean home_joints) | home (keyboard home)
#
# Gripper snap (diagnostic for "hovers over cube, never closes"):
#   LAB01_GRIPPER_SNAP=1 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd
#   Tuning: LAB01_GRIPPER_SNAP_THRESHOLD (0.45) / _CLOSED (-0.1) / _OPEN (predicted)
#           LAB01_GRIPPER_SNAP_LATCH=1 keeps the jaws closed once they close.
#   Report snap on/off next to any success rate — it changes execution, not the policy.
#
# Proximity oracle (close when gripperframe is within 5 cm of the cube; open over the box):
#   LAB01_GRIPPER_ORACLE=1 ./labs/lab01_pnp/eval_vla_jepa_gui.cmd
#   Tuning: LAB01_GRIPPER_ORACLE_RADIUS (0.04 m)
#   This is an ablation of the gripper channel, not a policy score.
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "$_LAB01_DIR/../.." && pwd)"
cd "$_REPO_ROOT"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is unset — GLFW needs a graphical session (local desktop or ssh -X)." >&2
    exit 1
fi

VARIANT="${LAB01_JEPA_EVAL_VARIANT:-bc10k}"
START="${LAB01_JEPA_EVAL_START:-demo}"

case "$START" in
    demo)
        export LAB01_EVAL_CONFIG="${LAB01_EVAL_CONFIG:-labs/lab01_pnp/configs/rollout_vla_jepa_demo_start.yaml}"
        ;;
    home)
        export LAB01_EVAL_CONFIG="${LAB01_EVAL_CONFIG:-labs/lab01_pnp/configs/rollout_vla_jepa_demo_fixed.yaml}"
        ;;
    *)
        echo "Unknown LAB01_JEPA_EVAL_START=$START (use demo or home)" >&2
        exit 1
        ;;
esac

case "$VARIANT" in
    bc10k|bc)
        _ckpt=./outputs/train/lab01_pnp_vla_jepa_state6_bc/checkpoints/010000/pretrained_model
        _n=30
        ;;
    bc20k)
        _ckpt=./outputs/train/lab01_pnp_vla_jepa_state6_bc/checkpoints/020000/pretrained_model
        _n=30
        ;;
    acthead)
        _ckpt=./outputs/train/lab01_pnp_vla_jepa_state6_bc_acthead/checkpoints/003000/pretrained_model
        _n=30
        ;;
    wm|state6)
        _ckpt=./outputs/train/lab01_pnp_vla_jepa_state6/checkpoints/020000/pretrained_model
        _n=7
        ;;
    15d)
        _ckpt=./outputs/train/lab01_pnp_vla_jepa/checkpoints/010000/pretrained_model
        _n=7
        ;;
    *)
        echo "Unknown LAB01_JEPA_EVAL_VARIANT=$VARIANT (bc10k|bc20k|acthead|wm|15d)" >&2
        exit 1
        ;;
esac

export LAB01_POLICY_PATH="${LAB01_POLICY_PATH:-$_ckpt}"
export LAB01_N_ACTION_STEPS="${LAB01_N_ACTION_STEPS:-$_n}"
export LAB01_MUJOCO_GL=glfw
export LAB01_RENDER_WINDOW=true
export LAB01_EVAL_EPISODES="${LAB01_EVAL_EPISODES:-3}"
_diag_tag=""
[[ "${LAB01_GRIPPER_SNAP:-}" == "1" || "${LAB01_GRIPPER_SNAP:-}" == "true" ]] && _diag_tag="${_diag_tag}_snap"
[[ "${LAB01_GRIPPER_ORACLE:-}" == "1" || "${LAB01_GRIPPER_ORACLE:-}" == "true" ]] && _diag_tag="${_diag_tag}_oracle"
export LAB01_EVAL_LOG="${LAB01_EVAL_LOG:-./outputs/eval/vla_jepa_gui_${VARIANT}_${START}${_diag_tag}.log}"

echo "=== Lab 01: VLA-JEPA GLFW eval ==="
echo "variant:  $VARIANT"
echo "start:    $START  ($LAB01_EVAL_CONFIG)"
echo "policy:   $LAB01_POLICY_PATH"
echo "n_action: $LAB01_N_ACTION_STEPS  episodes: $LAB01_EVAL_EPISODES"
echo "snap:     ${LAB01_GRIPPER_SNAP:-off}  (threshold ${LAB01_GRIPPER_SNAP_THRESHOLD:-0.45}, latch ${LAB01_GRIPPER_SNAP_LATCH:-off})"
echo "oracle:   ${LAB01_GRIPPER_ORACLE:-off}  (radius ${LAB01_GRIPPER_ORACLE_RADIUS:-0.04} m)"
echo "log:      $LAB01_EVAL_LOG"
echo ""

exec "$_LAB01_DIR/eval.cmd" "$@"
