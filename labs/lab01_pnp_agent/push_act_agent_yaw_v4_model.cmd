#!/bin/bash
# Create/upload Lab 01 ACT agent-yaw v4 Hub repo (MI300X 80K pretrained_model).
#
# Usage (from repo root, after hf auth login):
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp_agent/push_act_agent_yaw_v4_model.cmd
#   ./labs/lab01_pnp_agent/push_act_agent_yaw_v4_model.cmd --card-only
#
# Override:
#   LAB01_ACT_YAW_V4_HF_REPO_ID=your-user/your-repo ./labs/lab01_pnp_agent/push_act_agent_yaw_v4_model.cmd
#   LAB01_ACT_YAW_V4_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp_agent/push_act_agent_yaw_v4_model.cmd
set -euo pipefail
_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_AGENT_DIR/../../scripts/quicktest/_common.sh"
source "$_AGENT_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

CARD="$_AGENT_DIR/hf_model_card_act_agent_yaw_v4.md"
REPO_ID="${LAB01_ACT_YAW_V4_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act-agent-yaw-v4}"
CKPT_STEP="${LAB01_ACT_YAW_V4_CKPT_STEP:-080000}"
POLICY_PATH="${LAB01_ACT_YAW_V4_POLICY_PATH:-./outputs/train/lab01_pnp_act_agent_yaw_v4/checkpoints/${CKPT_STEP}/pretrained_model}"

CARD_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --card-only) CARD_ONLY=true ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--card-only]" >&2
            exit 1
            ;;
    esac
done

HF="${HF_BIN:-$(command -v hf || true)}"
if [[ -z "$HF" && -x "$(dirname "$PYTHON")/hf" ]]; then
    HF="$(dirname "$PYTHON")/hf"
fi
if [[ -z "$HF" ]]; then
    echo "hf CLI not found. Install huggingface_hub or activate .venv-rocm." >&2
    exit 1
fi

if [[ ! -f "$CARD" ]]; then
    echo "Model card not found: $CARD" >&2
    exit 1
fi

echo "=== Lab 01: push ACT agent-yaw v4 model ==="
echo "Endpoint: ${HF_ENDPOINT:-https://huggingface.co}"
echo "Hub:      https://huggingface.co/$REPO_ID"
echo "Card:     $CARD"
echo "Weights:  $POLICY_PATH"
echo ""

"$HF" repo create "$REPO_ID" --type model --exist-ok

if [[ "$CARD_ONLY" != true ]]; then
    if [[ ! -f "$POLICY_PATH/model.safetensors" ]]; then
        echo "Checkpoint not found: $POLICY_PATH/model.safetensors" >&2
        exit 1
    fi
    echo "=== Upload pretrained_model/ ==="
    "$HF" upload "$REPO_ID" "$POLICY_PATH" . \
        --repo-type model \
        --commit-message "Add ACT agent-yaw v4 MI300X 80K pretrained_model"
fi

echo "=== Upload model card ==="
"$HF" upload "$REPO_ID" "$CARD" README.md \
    --repo-type model \
    --commit-message "Update ACT agent-yaw v4 model card (SimStudio Lab 01)"

echo ""
echo "Done: https://huggingface.co/$REPO_ID"
