#!/bin/bash
# Upload the 193-episode agent-yaw follow2 dataset.
#
#   LAB01_HF_UPLOAD_ENDPOINT=https://huggingface.co ./labs/lab01_pnp_agent/push_agent_yaw_follow2_dataset.cmd
#   ./labs/lab01_pnp_agent/push_agent_yaw_follow2_dataset.cmd --card-only
#   ./labs/lab01_pnp_agent/push_agent_yaw_follow2_dataset.cmd --skip-validate
set -euo pipefail
_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$_AGENT_DIR/../../scripts/quicktest/_common.sh"
source "$_AGENT_DIR/_env.sh"

if [[ -n "${LAB01_HF_UPLOAD_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT="$LAB01_HF_UPLOAD_ENDPOINT"
fi

ROOT="${LAB01_FOLLOW2_DATASET_ROOT:-$_AGENT_DIR/../../datasets/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2}"
REPO_ID="${LAB01_FOLLOW2_DATASET_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2}"
CARD="$_AGENT_DIR/hf_dataset_card_agent_yaw_v4_follow2.md"

CARD_ONLY=false
SKIP_VALIDATE=false
for arg in "$@"; do
    case "$arg" in
        --card-only) CARD_ONLY=true ;;
        --skip-validate) SKIP_VALIDATE=true ;;
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

_push_card() {
    "$HF" upload "$REPO_ID" "$CARD" README.md --repo-type dataset \
        --commit-message "Add agent-yaw v4 follow2 dataset card (193 episodes)"
}

_ensure_v3_tag() {
    "$PYTHON" - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
refs = api.list_repo_refs(repo, repo_type="dataset")
if any(t.name == "v3.0" for t in refs.tags):
    print("Tag v3.0 already exists")
else:
    api.create_tag(repo, tag="v3.0", repo_type="dataset", revision="main")
    print("Created tag v3.0")
PY
}

echo "=== Push agent-yaw follow2 dataset ==="
echo "Local: $ROOT"
echo "Hub:   https://huggingface.co/datasets/$REPO_ID"
"$HF" repo create "$REPO_ID" --type dataset --exist-ok

if [[ "$CARD_ONLY" == true ]]; then
    _push_card
    _ensure_v3_tag
    exit 0
fi

if [[ "$SKIP_VALIDATE" != true ]]; then
    "$PYTHON" -m simstudio.scripts.validate_dataset --root "$ROOT"
fi

set +e
"$PYTHON" - "$REPO_ID" "$ROOT" <<'PY'
import sys
from lerobot.datasets import LeRobotDataset

repo, root = sys.argv[1], sys.argv[2]
dataset = LeRobotDataset(repo_id=repo, root=root)
dataset.push_to_hub(
    tags=["so101", "simstudio", "mujoco", "pick-and-place", "lerobot", "yaw-aware"],
    license="apache-2.0",
    private=False,
    push_videos=True,
)
print(f"Uploaded: https://huggingface.co/datasets/{repo}")
PY
upload_status=$?
set -e
if [[ "$upload_status" -ne 0 ]]; then
    echo "WARNING: push_to_hub returned $upload_status. Checking Hub files..."
    "$PYTHON" - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
api = HfApi()
if not api.repo_exists(repo, repo_type="dataset"):
    raise SystemExit("Hub repo missing")
files = set(api.list_repo_files(repo, repo_type="dataset"))
need = {"meta/info.json"}
missing = need - files
if missing:
    raise SystemExit(f"Hub repo incomplete, missing: {sorted(missing)}")
print(f"Hub repo has {len(files)} files. Continuing with README + v3.0 tag.")
PY
fi

_push_card
_ensure_v3_tag
echo "Done: https://huggingface.co/datasets/$REPO_ID"
