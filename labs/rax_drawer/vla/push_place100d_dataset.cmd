#!/bin/bash
# Push place100d LeRobot dataset to Hugging Face (official endpoint).
# Prefer running on DORobot when halo cannot reach huggingface.co.
# Do not publish via hf-mirror.com.
set -euo pipefail
_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$_DIR/_env.sh"
PYTHON="${PYTHON:-$REPO_ROOT/.venv-rocm/bin/python}"
PRIVATE="${PRIVATE:-false}"
# Force official Hub unless the caller already set a non-mirror endpoint.
if [[ "${HF_ENDPOINT:-}" == *"hf-mirror.com"* ]]; then
  echo "Refusing HF_ENDPOINT=$HF_ENDPOINT — use https://huggingface.co (e.g. from DORobot)" >&2
  exit 1
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "Local: $RAX_DATASET_ROOT"
echo "Hub:   $RAX_DATASET_REPO_ID"
echo "HF_ENDPOINT=$HF_ENDPOINT"

PY_PRIVATE=False
[[ "$PRIVATE" == true ]] && PY_PRIVATE=True

"$PYTHON" <<PY
from lerobot.datasets import LeRobotDataset
from pathlib import Path
import shutil

repo = "${RAX_DATASET_REPO_ID}"
root = "${RAX_DATASET_ROOT}"
card = Path("labs/rax_drawer/vla/hf_dataset_card_place100d.md")
readme = Path(root) / "README.md"
if card.is_file():
    shutil.copyfile(card, readme)

ds = LeRobotDataset(repo_id=repo, root=root)
ds.push_to_hub(
    tags=["so101", "simstudio", "mujoco", "drawer", "lerobot", "rgb-d"],
    license="apache-2.0",
    private=${PY_PRIVATE},
    push_videos=True,
)
print(f"Uploaded: https://huggingface.co/datasets/{repo}")
PY
