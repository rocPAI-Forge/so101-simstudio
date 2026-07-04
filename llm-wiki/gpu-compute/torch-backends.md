---
tags: [pytorch, cuda, rocm, mps, backend]
platform: [linux, macos, windows]
update-check: 2026-07
---

# PyTorch Backend Selection

PyTorch supports three compute backends: CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon). Only one can be active per install.

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
| CUDA | `uv pip install torch` |
| ROCm 7.2 | `uv pip install --torch-backend rocm7.2 torch` |
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

If both NVIDIA and AMD GPUs are present, explicitly choose one. Do not install both torch builds in the same venv.
