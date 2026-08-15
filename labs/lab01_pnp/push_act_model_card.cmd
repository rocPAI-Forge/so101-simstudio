#!/bin/bash
# Upload / overwrite the Hugging Face model card (README.md) for the Lab 01 ACT checkpoint.
#
# Usage (from repo root, after hf auth login):
#   ./labs/lab01_pnp/push_act_model_card.cmd
#
# Override hub repo:
#   LAB01_ACT_HF_REPO_ID=your-user/your-act-repo ./labs/lab01_pnp/push_act_model_card.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

CARD="$_LAB01_DIR/hf_model_card_act.md"
REPO_ID="${LAB01_ACT_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-act}"

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

echo "=== Upload model card ==="
echo "Source:  $CARD"
echo "Hub repo: $REPO_ID"
echo ""

"$HF" upload "$REPO_ID" "$CARD" README.md \
    --repo-type model \
    --commit-message "Update model card (SimStudio Lab 01 pointer)"

echo ""
echo "Done: https://huggingface.co/$REPO_ID"
