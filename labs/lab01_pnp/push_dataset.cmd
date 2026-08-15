#!/bin/bash
# Validate and upload Lab 01 dataset to Hugging Face Hub.
#
# Usage (from repo root, after hf auth login):
#   ./labs/lab01_pnp/push_dataset.cmd
#
# Options:
#   ./labs/lab01_pnp/push_dataset.cmd --private          # private dataset repo
#   ./labs/lab01_pnp/push_dataset.cmd --card-only        # refresh README only
#   ./labs/lab01_pnp/push_dataset.cmd --skip-validate    # skip integrity check
set -euo pipefail
_LAB01_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_LAB01_DIR/../../scripts/quicktest/_common.sh"
source "$_LAB01_DIR/_env.sh"

CARD="$_LAB01_DIR/hf_dataset_card.md"
PRIVATE=false
CARD_ONLY=false
SKIP_VALIDATE=false

for arg in "$@"; do
    case "$arg" in
        --private) PRIVATE=true ;;
        --card-only) CARD_ONLY=true ;;
        --skip-validate) SKIP_VALIDATE=true ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--private] [--card-only] [--skip-validate]" >&2
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
    echo "Dataset card not found: $CARD" >&2
    exit 1
fi

_push_card() {
    echo ""
    echo "=== Upload dataset card (README.md) ==="
    "$HF" upload "$LAB01_DATASET_REPO_ID" "$CARD" README.md \
        --repo-type dataset \
        --commit-message "Update dataset card (SimStudio Lab 01 pointer)"
    echo "Dataset page: https://huggingface.co/datasets/$LAB01_DATASET_REPO_ID"
}

_ensure_v3_tag() {
    echo ""
    echo "=== Ensure LeRobot v3.0 tag ==="
    "$PYTHON" <<PY
from huggingface_hub import HfApi

api = HfApi()
repo = "${LAB01_DATASET_REPO_ID}"
refs = api.list_repo_refs(repo, repo_type="dataset")
if any(t.name == "v3.0" for t in refs.tags):
    print("Tag v3.0 already exists")
else:
    api.create_tag(repo, tag="v3.0", repo_type="dataset", revision="main")
    print("Created tag v3.0")
PY
}

if [[ "$CARD_ONLY" == true ]]; then
    echo "=== Lab 01: push dataset card only ==="
    _push_card
    _ensure_v3_tag
    exit 0
fi

if [[ ! -d "$LAB01_DATASET_ROOT/meta" ]]; then
    echo "Dataset not found: $LAB01_DATASET_ROOT" >&2
    echo "Record first: ./labs/lab01_pnp/record.cmd" >&2
    exit 1
fi

echo "=== Lab 01: push dataset to Hugging Face ==="
echo "Local:  $LAB01_DATASET_ROOT"
echo "Hub:    $LAB01_DATASET_REPO_ID"
echo "Private: $PRIVATE"
echo ""

if [[ "$SKIP_VALIDATE" != true ]]; then
    echo "=== Validate dataset ==="
    "$PYTHON" -m simstudio.scripts.validate_dataset --root "$LAB01_DATASET_ROOT"
    echo ""
fi

echo "=== Upload dataset files (LeRobot push_to_hub) ==="
PY_PRIVATE="False"
if [[ "$PRIVATE" == true ]]; then
    PY_PRIVATE="True"
fi
set +e
"$PYTHON" <<PY
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="${LAB01_DATASET_REPO_ID}",
    root="${LAB01_DATASET_ROOT}",
)
dataset.push_to_hub(
    tags=["so101", "simstudio", "mujoco", "pick-and-place", "lerobot"],
    license="apache-2.0",
    private=${PY_PRIVATE},
    push_videos=True,
)
print(f"Uploaded: https://huggingface.co/datasets/${LAB01_DATASET_REPO_ID}")
PY
upload_status=$?
set -e
if [[ "$upload_status" -ne 0 ]]; then
    echo ""
    echo "WARNING: push_to_hub returned $upload_status (often the auto dataset-card step)."
    echo "Checking whether data files are already on the Hub..."
    "$PYTHON" <<PY
from huggingface_hub import HfApi
api = HfApi()
repo = "${LAB01_DATASET_REPO_ID}"
if not api.repo_exists(repo, repo_type="dataset"):
    raise SystemExit("Hub repo missing — re-run after fixing network/auth.")
files = api.list_repo_files(repo, repo_type="dataset")
need = {"meta/info.json", "data/chunk-000/file-000.parquet"}
missing = need - set(files)
if missing:
    raise SystemExit(f"Hub repo incomplete, missing: {sorted(missing)}")
print(f"Hub repo looks complete ({len(files)} files). Continuing with README + v3.0 tag.")
PY
fi

_push_card
_ensure_v3_tag

echo ""
echo "Done."
