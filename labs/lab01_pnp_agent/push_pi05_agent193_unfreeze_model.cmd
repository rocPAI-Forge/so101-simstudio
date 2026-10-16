#!/bin/bash
# Upload the pi05 full-unfreeze checkpoint (pure-model Home20 34/60).
# Does not touch alexhegit/so101-simstudio-lab01-pnp-pi05-agent193 (the 12/60 run).
#
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp_agent/push_pi05_agent193_unfreeze_model.cmd
#   ./labs/lab01_pnp_agent/push_pi05_agent193_unfreeze_model.cmd --card-only
set -euo pipefail
_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_AGENT_DIR/../../scripts/quicktest/_common.sh"
source "$_AGENT_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

CARD="$_AGENT_DIR/hf_model_card_pi05_agent193_unfreeze.md"
REPO_ID="${LAB01_PI05_UNFREEZE_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze}"
POLICY_PATH="${LAB01_PI05_UNFREEZE_POLICY_PATH:-/tmp/lab01_pi05_front_unfreezeA2_15k/checkpoints/005000/pretrained_model}"

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

echo "=== Push pi05 agent193 full unfreeze (Home20 34/60) ==="
echo "Hub: https://huggingface.co/$REPO_ID"
"$HF" repo create "$REPO_ID" --type model --exist-ok
if [[ "$CARD_ONLY" != true ]]; then
    [[ -f "$POLICY_PATH/model.safetensors" ]] || { echo "Missing $POLICY_PATH/model.safetensors" >&2; exit 1; }
    "$HF" upload "$REPO_ID" "$POLICY_PATH" . --repo-type model \
        --commit-message "Add pi05 full-unfreeze pretrained_model (Home20 34/60)"
fi
"$HF" upload "$REPO_ID" "$CARD" README.md --repo-type model \
    --commit-message "Add pi05 full-unfreeze model card (Home20 34/60)"
echo "Done: https://huggingface.co/$REPO_ID"
