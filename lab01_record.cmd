#!/bin/bash
# Temporary alias — same as labs/lab01_pnp/record.cmd (lab01-pnp dataset).
#
# Usage (from repo root):
#   source .venv-rocm/bin/activate
#   ./lab01_record.cmd
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$REPO_ROOT/labs/lab01_pnp/record.cmd" "$@"
