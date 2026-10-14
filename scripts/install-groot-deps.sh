#!/usr/bin/env bash
# Wrapper: GR00T N1.7 extras into .venv-rocm without replacing ROCm torch.
# Never: uv pip install 'lerobot[groot]'
exec "$(cd "$(dirname "$0")" && pwd)/install-lerobot-extra-deps.sh" groot "$@"
