#!/bin/bash
# Upload GR00T N1.7 absolute-joint 10K (Home20 14/60).
# Does not replace alexhegit/so101-simstudio-lab01-pnp-groot-n17 (50-ep relative negative result).
#
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp_agent/push_groot_agent193_abs_model.cmd
#   ./labs/lab01_pnp_agent/push_groot_agent193_abs_model.cmd --card-only
set -euo pipefail
_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_AGENT_DIR/../../scripts/quicktest/_common.sh"
source "$_AGENT_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

CARD="$_AGENT_DIR/hf_model_card_groot_agent193_abs.md"
REPO_ID="${LAB01_GROOT_AGENT193_ABS_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k}"
POLICY_PATH="${LAB01_GROOT_AGENT193_ABS_POLICY_PATH:-./outputs/train/lab01_pnp_groot_n17_state6_agent193_abs10k/checkpoints/010000/pretrained_model}"

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

echo "=== Push GR00T N1.7 absolute 10K ==="
echo "Hub: https://huggingface.co/$REPO_ID"
"$HF" repo create "$REPO_ID" --type model --exist-ok
if [[ "$CARD_ONLY" != true ]]; then
    [[ -f "$POLICY_PATH/model.safetensors" ]] || { echo "Missing $POLICY_PATH/model.safetensors" >&2; exit 1; }
    "$HF" upload "$REPO_ID" "$POLICY_PATH" . --repo-type model \
        --commit-message "Add GR00T N1.7 absolute-joint 10K pretrained_model (Home20 14/60)"
fi
"$HF" upload "$REPO_ID" "$CARD" README.md --repo-type model \
    --commit-message "Update GR00T absolute-joint agent193 model card"
echo "Done: https://huggingface.co/$REPO_ID"
