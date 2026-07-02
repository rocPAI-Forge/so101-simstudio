#!/usr/bin/env bash
# setup-rocm.sh — create .venv-rocm with ROCm 7.2 PyTorch for Python 3.12
#
# Usage:
#   ./scripts/setup-rocm.sh          # create venv and install all deps
#   ./scripts/setup-rocm.sh --sync   # re-sync only (venv must exist)
set -euo pipefail

ROCM_VENV=".venv-rocm"
PYTHON="${PYTHON:-python3.12}"

info()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
die()   { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found — https://docs.astral.sh/uv/getting-started/installation/"
[[ "$($PYTHON --version 2>&1)" == Python<3.12* ]] && die "$PYTHON must be >=3.12"

SYNC_ONLY=false
[[ "${1:-}" == "--sync" ]] && SYNC_ONLY=true

# ── create venv if needed ───────────────────────────────────────────────
if [[ ! -d "$ROCM_VENV" ]]; then
    info "Creating $ROCM_VENV ($PYTHON) ..."
    uv venv "$ROCM_VENV" --python "$PYTHON"
fi

PY="$ROCM_VENV/bin/python"

# ── install project (no-deps first to avoid pulling CUDA torch) ────────
info "Installing so101-mujoco-teleop (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e ".[smolvla]"

# ── install lerobot (local editable, no-deps) ──────────────────────────
info "Installing lerobot (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e "lerobot[kinematics,smolvla]"

# ── install remaining deps (excluding torch) ────────────────────────────
info "Installing remaining dependencies ..."
uv pip install --python "$PY" \
    "mujoco>=3.0.0,<4.0.0" \
    "scipy>=1.10.0" \
    "matplotlib>=3.10.6" \
    "datasets>=4.0.0" \
    "diffusers>=0.27.2" \
    "huggingface-hub[hf-transfer,cli]>=0.34.2" \
    "cmake>=3.29.0.1" \
    "einops>=0.8.0" \
    "opencv-python-headless>=4.9.0" \
    "av>=14.2.0" \
    "jsonlines>=4.0.0" \
    "packaging>=24.2" \
    "pynput>=1.7.7" \
    "pyserial>=3.5" \
    "wandb>=0.20.0" \
    "torchcodec>=0.2.1,<0.6.0" \
    "draccus==0.10.0" \
    "gymnasium>=0.29.1,<1.0.0" \
    "rerun-sdk>=0.21.0,<0.23.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "imageio[ffmpeg]>=2.34.0,<3.0.0" \
    "termcolor>=2.4.0,<4.0.0" \
    "placo>=0.9.6" \
    "transformers>=4.53.0" \
    "num2words>=0.5.14" \
    "accelerate>=1.7.0" \
    "safetensors>=0.4.3"

# ── install torch (ROCm 7.2) LAST, force-reinstall to override any CUDA ─
info "Installing torch via --torch-backend rocm7.2 ..."
uv pip install --python "$PY" --torch-backend rocm7.2 --force-reinstall torch torchvision

# ── install dev/test deps ──────────────────────────────────────────────
info "Installing dev/test deps ..."
uv pip install --python "$PY" pytest black ruff

# ── verify ──────────────────────────────────────────────────────────────
info "Verifying torch ..."
"$PY" -c "
import torch
v = torch.__version__
hip = getattr(torch.version, 'hip', None)
print(f'torch {v}' + (f'  ROCm/HIP {hip}' if hip else '  WARNING: HIP not detected'))
"

info "Done.  Activate:  source $ROCM_VENV/bin/activate"
