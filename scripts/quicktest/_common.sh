# Shared helpers for fixed quick-test launchers (collaboration / smoke-style runs).
QUICKTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$QUICKTEST_DIR/../.." && pwd)"

# 1) SIMSTUDIO_VENV (e.g. .venv-rocm on the MI210s)
# 2) .venv-cuda when that interpreter can see a CUDA device
#    (nvidia-smi can fail on an NVML mismatch while torch still works)
# 3) .venv-rocm, then .venv
if [[ -n "${SIMSTUDIO_VENV:-}" ]]; then
    if [[ -x "$REPO_ROOT/${SIMSTUDIO_VENV}/bin/python" ]]; then
        PYTHON="$REPO_ROOT/${SIMSTUDIO_VENV}/bin/python"
    else
        echo "SIMSTUDIO_VENV=${SIMSTUDIO_VENV} has no python. Expected $REPO_ROOT/${SIMSTUDIO_VENV}/bin/python" >&2
        exit 1
    fi
elif [[ -x "$REPO_ROOT/.venv-cuda/bin/python" ]] && "$REPO_ROOT/.venv-cuda/bin/python" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    PYTHON="$REPO_ROOT/.venv-cuda/bin/python"
elif [[ -x "$REPO_ROOT/.venv-rocm/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv-rocm/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    echo "No project venv found. Run: make cuda-sync  or  make rocm-sync" >&2
    exit 1
fi

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"

# Mixed NVIDIA+AMD hosts: Mesa's EGL ICD binds MI210 (compute-only) and
# headless MuJoCo dies in radeonsi. Prefer the NVIDIA ICD when using .venv-cuda.
if [[ "$PYTHON" == "$REPO_ROOT/.venv-cuda/bin/python" ]]; then
    _nvidia_egl="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" && -f "$_nvidia_egl" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$_nvidia_egl"
    fi
    unset _nvidia_egl
fi
