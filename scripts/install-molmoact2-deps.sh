#!/usr/bin/env bash
# Install MolmoAct2 Python extras into .venv-rocm without replacing ROCm torch.
#
# Do NOT run:  uv pip install 'lerobot[molmoact2]'
#   That extra is defined in the LeRobot submodule, but installing it via uv
#   re-resolves the PyPI/lerobot graph and pulls CUDA torch + nvidia-* wheels.
#   Policy code is already in the editable `lerobot/` checkout.
#
# The extra is only: transformers>=5.4,<5.6  peft>=0.18,<1  scipy>=1.14
#
# Usage (repo root):
#   ./scripts/install-molmoact2-deps.sh
#   ./scripts/install-molmoact2-deps.sh --repair-torch   # CUDA already leaked in
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
ROCM_VENV="${ROCM_VENV:-.venv-rocm}"
PY="$ROCM_VENV/bin/python"

info() { printf '\033[1;34m%s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -x "$PY" ]] || die "$PY not found. Run: make rocm-sync"

REPAIR=false
[[ "${1:-}" == "--repair-torch" ]] && REPAIR=true

if "$REPAIR"; then
    info "Reinstalling ROCm torch (repair) ..."
    uv pip install --python "$PY" --torch-backend rocm7.2 --force-reinstall torch torchvision
fi

TORCH_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$TORCH_CONSTRAINTS"' EXIT
"$PY" - <<'PY' > "$TORCH_CONSTRAINTS"
import torch
import torchvision

print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY

info "Pinning $(tr '\n' ' ' < "$TORCH_CONSTRAINTS")"
info "Installing MolmoAct2 extras (transformers / peft / scipy) ..."
uv pip install --python "$PY" --constraints "$TORCH_CONSTRAINTS" \
    "transformers>=5.4.0,<5.6.0" \
    "peft>=0.18.0,<1.0.0" \
    "scipy>=1.14.0,<2.0.0"

"$PY" - <<'PY'
import torch
v = torch.__version__
hip = getattr(torch.version, "hip", None)
ok = hip is not None and "+rocm" in v
print(f"torch {v}  HIP={hip}")
if not ok:
    raise SystemExit(
        "ROCm torch was replaced. Re-run: ./scripts/install-molmoact2-deps.sh --repair-torch\n"
        "Never use: uv pip install 'lerobot[molmoact2]'"
    )
import peft
import transformers
import scipy
print(f"peft {peft.__version__}  transformers {transformers.__version__}  scipy {scipy.__version__}")
PY

info "MolmoAct2 extras OK. Do not install lerobot[molmoact2] on this venv."
