# Shared helpers for fixed quick-test launchers (collaboration / smoke-style runs).
QUICKTEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$QUICKTEST_DIR/../.." && pwd)"

if [[ -x "$REPO_ROOT/.venv-rocm/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv-rocm/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    echo "No project venv found. Run: make rocm-sync  (or uv sync)" >&2
    exit 1
fi

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
