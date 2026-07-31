#!/usr/bin/env bash
# setup-rocm.sh — create .venv-rocm with ROCm 7.2 PyTorch for Python 3.12
#
# This is the ONLY supported install path for SO-101 SimStudio.
# CUDA (Linux) and macOS are not supported in the current release.
#
# Installs: ROCm torch → MuJoCo/LeRobot/SmolVLA deps → editable project + lerobot.
#
# Usage:
#   ./scripts/setup-rocm.sh          # create venv and install all deps
#   ./scripts/setup-rocm.sh --sync   # re-sync only (venv must exist)
#
# Do NOT run bare `uv sync` in .venv-rocm — it can replace ROCm torch with CUDA.
set -euo pipefail

ROCM_VENV=".venv-rocm"
PYTHON="${PYTHON:-python3.12}"

info()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
die()   { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found — https://docs.astral.sh/uv/getting-started/installation/"
py_version="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$py_version" < "3.12" ]] && die "$PYTHON must be >=3.12"

SYNC_ONLY=false
[[ "${1:-}" == "--sync" ]] && SYNC_ONLY=true

# ── create venv if needed ───────────────────────────────────────────────
if [[ ! -d "$ROCM_VENV" ]]; then
    info "Creating $ROCM_VENV ($PYTHON) ..."
    uv venv "$ROCM_VENV" --python "$PYTHON"
fi

PY="$ROCM_VENV/bin/python"

# ── install torch (ROCm 7.2) first, before any deps that may pull torch ──
info "Installing torch via --torch-backend rocm7.2 ..."
uv pip install --python "$PY" --torch-backend rocm7.2 --force-reinstall torch torchvision

TORCH_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$TORCH_CONSTRAINTS"' EXIT
"$PY" - <<'PY' > "$TORCH_CONSTRAINTS"
import torch
import torchvision

print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY

# ── install remaining deps, constraining torch/torchvision to ROCm builds ─
info "Installing remaining dependencies ..."
uv pip install --python "$PY" --constraints "$TORCH_CONSTRAINTS" \
    "mujoco>=3.0.0,<4.0.0" \
    "scipy>=1.10.0" \
    "matplotlib>=3.10.6" \
    "datasets>=4.0.0" \
    "diffusers>=0.27.2" \
    "huggingface-hub[hf-transfer,cli]>=0.34.2" \
    "cmake>=3.29.0.1" \
    "einops>=0.8.0" \
    "opencv-python-headless>=4.9.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0" \
    "packaging>=24.2" \
    "pynput>=1.7.7" \
    "pyserial>=3.5" \
    "wandb>=0.20.0" \
    "torchcodec>=0.2.1,<0.6.0" \
    "draccus==0.10.0" \
    "gymnasium>=0.29.1,<1.0.0" \
    "rerun-sdk>=0.24.0,<0.34.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "imageio[ffmpeg]>=2.34.0,<3.0.0" \
    "termcolor>=2.4.0,<4.0.0" \
    "placo>=0.9.6,<0.9.16" \
    "cmeel-urdfdom>=4,<5" \
    "cmeel-tinyxml2<11" \
    "transformers>=4.53.0" \
    "num2words>=0.5.14" \
    "accelerate>=1.7.0" \
    "safetensors>=0.4.3"

# ── install local editable packages last, without deps ───────────────────
info "Installing so101-simstudio (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e ".[smolvla]"

info "Installing lerobot (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e "lerobot[kinematics,smolvla]"

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
