#!/bin/bash
# Leader arm pick-and-place recording — delegates to lab01 defaults (_env.sh).
#
# Run from an existing terminal (do not double-click):
#   cd ~/Repo/so101-simstudio
#   source .venv-rocm/bin/activate
#   ./scripts/quicktest/record.cmd
set -euo pipefail
exec "$(cd "$(dirname "$0")/../.." && pwd)/labs/lab01_pnp/record.cmd" "$@"
