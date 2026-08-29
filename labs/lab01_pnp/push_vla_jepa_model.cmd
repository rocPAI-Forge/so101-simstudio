#!/bin/bash
# Create/upload the Lab 01 VLA-JEPA Hub repo (MI300X 20K pretrained_model).
#
# Usage (from repo root, after hf auth login):
#   ./labs/lab01_pnp/push_vla_jepa_model.cmd
#   ./labs/lab01_pnp/push_vla_jepa_model.cmd --card-only
#
# Force official Hub (recommended if HF_ENDPOINT is a mirror):
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp/push_vla_jepa_model.cmd
#
# Override:
#   LAB01_JEPA_HF_REPO_ID=your-user/your-repo ./labs/lab01_pnp/push_vla_jepa_model.cmd
#   LAB01_POLICY_PATH=./outputs/train/.../pretrained_model ./labs/lab01_pnp/push_vla_jepa_model.cmd
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

CARD="$_LAB01_DIR/hf_model_card_vla_jepa.md"
REPO_ID="${LAB01_JEPA_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-vla-jepa}"
CKPT_STEP="${LAB01_JEPA_CKPT_STEP:-020000}"
POLICY_PATH="${LAB01_POLICY_PATH:-$LAB01_JEPA_OUTPUT/checkpoints/${CKPT_STEP}/pretrained_model}"

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

echo "=== Lab 01: push VLA-JEPA model ==="
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
        --commit-message "Add VLA-JEPA MI300X 20K pretrained_model"
fi

echo "=== Upload model card ==="
"$HF" upload "$REPO_ID" "$CARD" README.md \
    --repo-type model \
    --commit-message "Update VLA-JEPA model card (SimStudio Lab 01)"

echo ""
echo "Done: https://huggingface.co/$REPO_ID"
