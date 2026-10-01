#!/usr/bin/env bash
# setup-cuda.sh — create .venv-cuda with CUDA PyTorch for Python 3.12
#
# Parallel to scripts/setup-rocm.sh. Use this on NVIDIA GPUs (A30 is sm_80).
# Do not share this venv with ROCm torch. Do NOT run bare `uv sync` here —
# it can replace the pinned CUDA wheel.
#
# Usage:
#   ./scripts/setup-cuda.sh
#   CUDA_TORCH_BACKEND=cu128 CUDA_EXPECT_DEVICE_COUNT=4 ./scripts/setup-cuda.sh
#
# CUDA_EXPECT_DEVICE_COUNT defaults to 4 (this host's A30s). Set it to the
# number of NVIDIA GPUs you expect, or to 0 to skip the count check.
set -euo pipefail

CUDA_VENV=".venv-cuda"
PYTHON="${PYTHON:-python3.12}"
CUDA_TORCH_BACKEND="${CUDA_TORCH_BACKEND:-cu128}"
CUDA_EXPECT_DEVICE_COUNT="${CUDA_EXPECT_DEVICE_COUNT:-4}"

info()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
die()   { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found — https://docs.astral.sh/uv/getting-started/installation/"
py_version="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$py_version" < "3.12" ]] && die "$PYTHON must be >=3.12"

if [[ ! -d "$CUDA_VENV" ]]; then
    info "Creating $CUDA_VENV ($PYTHON) ..."
    uv venv "$CUDA_VENV" --python "$PYTHON"
fi

PY="$CUDA_VENV/bin/python"

info "Installing torch via --torch-backend $CUDA_TORCH_BACKEND ..."
uv pip install --python "$PY" --torch-backend "$CUDA_TORCH_BACKEND" --force-reinstall torch torchvision

TORCH_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$TORCH_CONSTRAINTS"' EXIT
"$PY" - <<'PY' > "$TORCH_CONSTRAINTS"
import torch
import torchvision

print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY

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
    "transformers>=5.4.0,<5.6.0" \
    "peft>=0.18.0,<1.0.0" \
    "num2words>=0.5.14" \
    "accelerate>=1.7.0" \
    "safetensors>=0.4.3"

info "Installing so101-simstudio (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e ".[smolvla]"

info "Installing lerobot (editable, no-deps) ..."
uv pip install --python "$PY" --no-deps -e "lerobot[kinematics,smolvla]"

info "Installing dev/test deps ..."
uv pip install --python "$PY" pytest black ruff

info "Verifying CUDA torch ..."
"$PY" - "$CUDA_EXPECT_DEVICE_COUNT" <<'PY'
import sys

import torch

expected = int(sys.argv[1])
version = torch.__version__
hip = getattr(torch.version, "hip", None)
print(f"torch {version}")
print(f"hip {hip}")
print(f"arch {torch.cuda.get_arch_list()}")

if "+cu" not in version:
    raise SystemExit(f"expected a CUDA wheel (+cu), got {version}")
if hip is not None:
    raise SystemExit(f"expected NVIDIA torch (hip is None), got hip={hip}")
if "sm_80" not in torch.cuda.get_arch_list():
    raise SystemExit(
        "this torch build has no sm_80; A30 needs Ampere. "
        "Re-run with a CUDA_TORCH_BACKEND that still includes sm_80."
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "torch.cuda.is_available() is False. "
        "The CUDA wheel is installed, but the NVIDIA driver is not usable "
        "(often a driver/library version mismatch — reboot after a driver upgrade)."
    )

count = torch.cuda.device_count()
print(f"device_count {count}")
for index in range(count):
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    print(f"  [{index}] {name}  cc={props.major}.{props.minor}  {props.total_memory / 1024**3:.1f} GiB")
    ones = torch.ones(64, 64, device=f"cuda:{index}")
    result = ones @ ones
    torch.cuda.synchronize(index)
    if result.shape != (64, 64):
        raise SystemExit(f"matmul on cuda:{index} returned shape {tuple(result.shape)}")
    print(f"  [{index}] matmul ok")

if expected and count != expected:
    raise SystemExit(f"expected {expected} CUDA devices, found {count}")
PY

info "Done.  Activate:  source $CUDA_VENV/bin/activate"
info "Do NOT run bare uv sync inside $CUDA_VENV."
