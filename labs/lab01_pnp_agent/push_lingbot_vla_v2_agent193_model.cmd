#!/bin/bash
# Upload LingBot-VLA 2.0 MI210 20K (Home20 25/60) to the Hub.
#
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co \
#     ./labs/lab01_pnp_agent/push_lingbot_vla_v2_agent193_model.cmd
#   ./labs/lab01_pnp_agent/push_lingbot_vla_v2_agent193_model.cmd --card-only
set -euo pipefail
_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_AGENT_DIR/../../scripts/quicktest/_common.sh"
source "$_AGENT_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

CARD="$_AGENT_DIR/hf_model_card_lingbot_vla_v2_agent193.md"
REPO_ID="${LAB01_V2_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k}"
POLICY_PATH="${LAB01_V2_POLICY_PATH:-./outputs/train/lab01_lingbot_vla_v2_agent193_mi210_unfreeze20k/checkpoints/020000/pretrained_model}"

CARD_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --card-only) CARD_ONLY=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

HF="${HF_BIN:-$(command -v hf || true)}"
if [[ -z "$HF" && -x "$(dirname "$PYTHON")/hf" ]]; then
    HF="$(dirname "$PYTHON")/hf"
fi
if [[ -z "$HF" ]]; then
    echo "hf CLI not found." >&2
    exit 1
fi

echo "=== Push LingBot-VLA 2.0 agent193 20K (Home20 25/60) ==="
echo "Endpoint: ${HF_ENDPOINT:-https://huggingface.co}"
echo "Hub:      https://huggingface.co/$REPO_ID"
echo "Weights:  $POLICY_PATH"
"$HF" repo create "$REPO_ID" --type model --exist-ok
if [[ "$CARD_ONLY" != true ]]; then
    [[ -f "$POLICY_PATH/model.safetensors" ]] || { echo "Missing $POLICY_PATH/model.safetensors" >&2; exit 1; }
    "$HF" upload "$REPO_ID" "$POLICY_PATH" . --repo-type model \
        --commit-message "Add LingBot-VLA 2.0 MI210 20K pretrained_model (Home20 25/60)"
fi
"$HF" upload "$REPO_ID" "$CARD" README.md --repo-type model \
    --commit-message "Add LingBot-VLA 2.0 model card (Home20 25/60)"
echo "Done: https://huggingface.co/$REPO_ID"
