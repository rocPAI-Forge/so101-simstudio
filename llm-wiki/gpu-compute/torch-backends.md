---
tags: [pytorch, cuda, rocm, mps, backend]
platform: [linux, macos, windows]
update-check: 2026-07
---

# PyTorch Backend Selection

> **SO-101 SimStudio:** Validated path is **ROCm on Ubuntu 24.04** (`make rocm-sync` → `.venv-rocm`).
> Branch `cuda` adds an unverified NVIDIA path: `make cuda-sync` → `.venv-cuda` (default `--torch-backend cu128`, A30 / sm_80).
> macOS is not supported in the current release.

PyTorch supports three compute backends: CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon). Only one can be active per install. On a machine with both NVIDIA and AMD GPUs, use two venvs. Do not install both torch builds in the same venv.

## Decision logic

```
Is there an NVIDIA GPU?
├── Yes → CUDA (default, just pip install torch)
├── No → Is there an AMD GPU?
│   ├── Yes → ROCm (--torch-backend rocm7.2)
│   └── No → CPU (--torch-backend cpu)
└── Is it macOS with Apple Silicon?
    └── Yes → MPS (default torch install, automatic)
```

## Install commands

| Backend | Command |
|---------|---------|
| CUDA (branch `cuda`, A30) | `make cuda-sync` (`uv pip install --torch-backend cu128 torch torchvision` inside `.venv-cuda`) |
| ROCm 7.2 | `make rocm-sync` (`uv pip install --torch-backend rocm7.2 torch`) |
| CPU | `uv pip install --torch-backend cpu torch` |
| MPS | Same as CUDA install; auto-detected at runtime |

## Verify

```python
import torch
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ROCm/HIP: {getattr(torch.version, 'hip', None)}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Common mistake

Installing CUDA torch on AMD machine or vice versa. The package installs fine but `torch.cuda.is_available()` returns `False` with no error.

## Multi-GPU machines

If both NVIDIA and AMD GPUs are present, pick one backend per venv (`.venv-cuda` or `.venv-rocm`). Launchers use `.venv-cuda` when its `torch.cuda.is_available()` is true; set `SIMSTUDIO_VENV=.venv-rocm` to force ROCm.

A30 (24 GB, sm_80) training in this repo is one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./labs/lab01_pnp/train.cmd
CUDA_VISIBLE_DEVICES=1 ./labs/lab01_pnp/eval.cmd
```

Start from Lab 01's `batch_size=4`. Do not reuse an MI300X `batch_size=64` recipe on a 24 GB A30.

On a host that also has AMD compute GPUs (for example MI210), headless MuJoCo EGL can bind Mesa/`radeonsi` and fail with `can't create a graphics context on a compute chip`. Lab launchers that select `.venv-cuda` set `__EGL_VENDOR_LIBRARY_FILENAMES` to `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` when that file exists. If NVIDIA EGL then cannot create a device display, the kernel module and userspace driver are probably mismatched (`nvidia-smi` fails the same way). Reboot after the driver upgrade. Until EGL works, `MUJOCO_GL=osmesa` still renders on the CPU.
