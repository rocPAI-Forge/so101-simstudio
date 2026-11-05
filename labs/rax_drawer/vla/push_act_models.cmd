#!/bin/bash
# Upload ACT RGB + RGB-D place100d checkpoints to Hugging Face (official endpoint).
# Prefer running on DORobot when halo cannot reach huggingface.co.
# Do not publish via hf-mirror.com.
set -euo pipefail
_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source "$_DIR/_env.sh"
PYTHON="${PYTHON:-$REPO_ROOT/.venv-rocm/bin/python}"
if [[ "${HF_ENDPOINT:-}" == *"hf-mirror.com"* ]]; then
  echo "Refusing HF_ENDPOINT=$HF_ENDPOINT — use https://huggingface.co (e.g. from DORobot)" >&2
  exit 1
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

RGB_LOCAL="${RAX_ACT_RGB_OUTPUT}/checkpoints/030000/pretrained_model"
RGBD_LOCAL="${RAX_ACT_DEPTH_OUTPUT}/checkpoints/030000/pretrained_model"
RGB_REPO="${RAX_ACT_RGB_HF_REPO_ID}"
RGBD_REPO="${RAX_ACT_RGBD_HF_REPO_ID}"

for p in "$RGB_LOCAL" "$RGBD_LOCAL"; do
  [[ -f "$p/model.safetensors" ]] || { echo "missing $p"; exit 1; }
done

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "RGB  $RGB_LOCAL -> $RGB_REPO"
echo "RGBD $RGBD_LOCAL -> $RGBD_REPO"

"$PYTHON" <<PY
from pathlib import Path
import shutil
from huggingface_hub import HfApi, create_repo

api = HfApi()
uploads = [
    ("${RGB_REPO}", Path("${RGB_LOCAL}"), Path("labs/rax_drawer/vla/hf_model_card_act_rgb.md")),
    ("${RGBD_REPO}", Path("${RGBD_LOCAL}"), Path("labs/rax_drawer/vla/hf_model_card_act_rgbd.md")),
]
for repo, local, card in uploads:
    create_repo(repo, repo_type="model", exist_ok=True, private=False)
    readme = local / "README.md"
    shutil.copyfile(card, readme)
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo,
        repo_type="model",
        commit_message="Upload ACT place100d checkpoint + model card",
    )
    print(f"Uploaded: https://huggingface.co/{repo}")
PY
