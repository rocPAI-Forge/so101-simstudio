#!/usr/bin/env bash
# Wrapper: LingBot-VA extras into .venv-rocm without replacing ROCm torch.
exec "$(cd "$(dirname "$0")" && pwd)/install-lerobot-extra-deps.sh" lingbot_va "$@"
