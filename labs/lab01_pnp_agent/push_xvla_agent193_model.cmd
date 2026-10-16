#!/bin/bash
# Upload the xVLA 10K checkpoint (pure-model Home20 47/60).
#
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp_agent/push_xvla_agent193_model.cmd
#   ./labs/lab01_pnp_agent/push_xvla_agent193_model.cmd --card-only
set -euo pipefail
_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_AGENT_DIR/../../scripts/quicktest/_common.sh"
source "$_AGENT_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

CARD="$_AGENT_DIR/hf_model_card_xvla_agent193.md"
REPO_ID="${LAB01_XVLA_AGENT193_HF_REPO_ID}"
POLICY_PATH="${LAB01_XVLA_AGENT193_POLICY_PATH}"

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

echo "=== Push xVLA agent193 10K (Home20 47/60) ==="
echo "Hub: https://huggingface.co/$REPO_ID"
"$HF" repo create "$REPO_ID" --type model --exist-ok
if [[ "$CARD_ONLY" != true ]]; then
    [[ -f "$POLICY_PATH/model.safetensors" ]] || { echo "Missing $POLICY_PATH/model.safetensors" >&2; exit 1; }
    "$HF" upload "$REPO_ID" "$POLICY_PATH" . --repo-type model \
        --commit-message "Add xVLA 10K pretrained_model (Home20 47/60)"
fi
"$HF" upload "$REPO_ID" "$CARD" README.md --repo-type model \
    --commit-message "Add xVLA agent193 model card (Home20 47/60)"
echo "Done: https://huggingface.co/$REPO_ID"
