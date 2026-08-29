---
tags: [rocm, pytorch, gpu, amd]
platform: [linux]
update-check: 2026-07
---

# ROCm Environment Setup

> **SO-101 SimStudio:** The **only supported** release target is **Ubuntu 24.04 + AMD ROCm**
> (`make rocm-sync` → `.venv-rocm`). **NVIDIA CUDA and macOS are not supported** in the
> current release (unverified; planned). See [ROADMAP.md](../../ROADMAP.md).

For AMD GPUs on Ubuntu only.

## Quick start (recommended)

Use the project script — do not hand-install unless debugging:

```bash
cd so101-simstudio
make rocm-sync
source .venv-rocm/bin/activate
```

This runs `scripts/setup-rocm.sh`, which:

1. Creates `.venv-rocm` (Python 3.12)
2. Installs ROCm PyTorch **first** (`uv pip install --torch-backend rocm7.2 torch torchvision`)
3. Installs MuJoCo, LeRobot, SmolVLA deps, and dev tools under torch constraints
4. Editable-installs the project and `lerobot` submodule
5. Prints a HIP verification line

First run may take 1–3 hours (ROCm torch ~6 GB download).

## Verify

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('HIP:', getattr(torch.version, 'hip', None))
print('GPU:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"
```

**Pass criteria:**

| Check | Expected |
|-------|----------|
| Version | Contains `+rocm`, not `+cu128` |
| `torch.version.hip` | Version string, not `None` |
| `torch.cuda.is_available()` | `True` on a working ROCm GPU |

On ROCm, PyTorch exposes the GPU through the `cuda` API namespace (`torch.cuda.*`).

## Critical: install order (manual installs only)

If you must install by hand instead of `make rocm-sync`:

1. Create venv
2. Install torch+torchvision with `--torch-backend rocm7.2` **FIRST**
3. Freeze torch version in a constraints file
4. Install all other deps with `--constraints`
5. Install local editable packages **LAST** with `--no-deps`

**Why:** If any dep pulls torch before you pin the ROCm build, uv/pip may install CUDA torch instead.

```bash
uv venv .venv-rocm --python python3.12
uv pip install --python .venv-rocm/bin/python --torch-backend rocm7.2 --force-reinstall torch torchvision
# generate constraints, then install remaining deps — see scripts/setup-rocm.sh
```

## Do not use `uv sync`

`uv sync` (without `--torch-backend rocm7.2`) resolves LeRobot deps against **CUDA** PyTorch wheels.
That path is **not supported** for SimStudio. Always use `make rocm-sync` to create or refresh `.venv-rocm`.

The same trap applies to **`uv pip install 'lerobot[molmoact2]'`** (and other `lerobot[extra]` installs that are not `--no-deps` on the local submodule). The extra only needs `transformers` / `peft` / `scipy`; installing it as a package extra re-resolves torch onto CUDA. Use:

```bash
./scripts/install-molmoact2-deps.sh
# if nvidia-* / +cu already leaked in:
./scripts/install-molmoact2-deps.sh --repair-torch
```

## When NOT to use this guide

- **NVIDIA GPU** → CUDA is not supported by SimStudio yet (planned)
- **macOS / Apple Silicon** → MPS is not supported yet (planned)
- **No GPU** → CPU-only torch is not the supported workflow for teleop/VLA training

## Common issues

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns False | Wrong torch build; delete `.venv-rocm`, re-run `make rocm-sync` |
| `torch.version.hip` is None | CUDA torch installed; delete venv, re-run `make rocm-sync`; never run `uv sync` after |
| Version shows `+cu128` | Same as above |
| `nvidia-*` packages in venv | CUDA torch pulled in; recreate venv with `make rocm-sync`, or `./scripts/install-molmoact2-deps.sh --repair-torch` if only extras leaked |
| Install very slow | Normal on first run (~6 GB torch wheel) |

## Pinning torch from resolver

```python
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
```

Save output to a constraints file and pass via `--constraints` when installing other packages.
