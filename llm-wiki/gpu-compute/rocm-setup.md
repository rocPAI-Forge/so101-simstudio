---
tags: [rocm, pytorch, gpu, amd]
platform: [linux]
update-check: 2026-07
---

# ROCm Environment Setup

For AMD GPUs only. If you have NVIDIA GPU, use CUDA instead.

## Quick start

```bash
# Create dedicated venv
uv venv .venv-rocm --python python3.12
source .venv-rocm/bin/activate

# Install torch with ROCm backend FIRST
uv pip install --torch-backend rocm7.2 --force-reinstall torch torchvision

# Then install remaining deps
uv pip install mujoco scipy matplotlib ...
```

## Critical: install order

1. Create venv
2. Install torch+torchvision with `--torch-backend rocm7.2` FIRST
3. Freeze torch version in constraints file
4. Install all other deps with `--constraints`
5. Install local editable packages LAST with `--no-deps`

**Why**: If any dep pulls torch before you pin the ROCm build, pip/uv may install CUDA torch instead.

## Pinning torch from resolver

```python
# Generate constraints from installed torch
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
```

Save output to `torch-constraints.txt`, pass via `--constraints`.

## Verify

```bash
python -c "import torch; print(torch.__version__, getattr(torch.version, 'hip', None))"
```

Expected: version string + ROCm/HIP version, not `None`.

## When NOT to use

- NVIDIA GPU available → use CUDA
- No GPU → use CPU-only torch
- macOS → use MPS backend

## Common issues

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns False | Wrong torch build; reinstall with `--torch-backend rocm7.2` |
| `torch.version.hip` is None | Same as above |
| CUDA packages installed alongside ROCm | Remove all `nvidia-*` packages, reinstall torch |
