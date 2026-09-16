#!/usr/bin/env bash
# Install LeRobot policy extras into .venv-rocm without replacing ROCm torch.
#
# Do NOT run:  uv pip install -e '.[lingbot_va]'  or  '.[fastwam]'
#   Those extras re-resolve the PyPI graph and typically pull CUDA torch + nvidia-*.
# Policy code is already in the editable `lerobot/` checkout; this script only
# installs the Python packages listed in lerobot/pyproject.toml extras.
#
# Usage (repo root):
#   ./scripts/install-lerobot-extra-deps.sh lingbot_va
#   ./scripts/install-lerobot-extra-deps.sh fastwam
#   ./scripts/install-lerobot-extra-deps.sh lingbot_va fastwam
#   ./scripts/install-lerobot-extra-deps.sh --repair-torch lingbot_va
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
ROCM_VENV="${ROCM_VENV:-.venv-rocm}"
PY="$ROCM_VENV/bin/python"

info() { printf '\033[1;34m%s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -x "$PY" ]] || die "$PY not found. Run: make rocm-sync"

REPAIR=false
EXTRAS=()
for arg in "$@"; do
    case "$arg" in
        --repair-torch) REPAIR=true ;;
        lingbot_va|fastwam|molmoact2) EXTRAS+=("$arg") ;;
        *) die "Unknown extra '$arg'. Use: lingbot_va | fastwam | molmoact2 [--repair-torch]" ;;
    esac
done
[[ ${#EXTRAS[@]} -gt 0 ]] || die "Pass at least one extra: lingbot_va fastwam molmoact2"

if "$REPAIR"; then
    info "Reinstalling ROCm torch (repair) ..."
    uv pip install --python "$PY" --torch-backend rocm7.2 --force-reinstall torch torchvision
fi

TORCH_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$TORCH_CONSTRAINTS"' EXIT
"$PY" - <<'PY' > "$TORCH_CONSTRAINTS"
import importlib

for name in ("torch", "torchvision", "torchaudio"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}=={mod.__version__}")
    except Exception:
        pass
PY

info "Pinning $(tr '\n' ' ' < "$TORCH_CONSTRAINTS")"

# Match lerobot/pyproject.toml extras, plus peft for LingBot-VA LoRA.
# Do not pin an upper bound on diffusers here: this venv may already have a
# newer working build (e.g. 0.39) from other policies; forcing <0.36 would
# churn the stack without helping ROCm torch.
PKGS=()
NEED_PEFT=false
NEED_SCIPY=false
for extra in "${EXTRAS[@]}"; do
    case "$extra" in
        lingbot_va)
            PKGS+=("transformers>=5.4.0,<5.6.0" "diffusers>=0.27.2" "accelerate>=1.14.0,<2.0.0")
            NEED_PEFT=true
            ;;
        fastwam)
            PKGS+=("transformers>=5.4.0,<5.6.0" "diffusers>=0.27.2" "ftfy" "regex")
            ;;
        molmoact2)
            PKGS+=("transformers>=5.4.0,<5.6.0")
            NEED_PEFT=true
            NEED_SCIPY=true
            ;;
    esac
done
if "$NEED_PEFT"; then
    PKGS+=("peft>=0.18.0,<1.0.0")
fi
if "$NEED_SCIPY"; then
    PKGS+=("scipy>=1.14.0,<2.0.0")
fi

# Dedup while keeping order
DEDUPED=()
seen=""
for p in "${PKGS[@]}"; do
    case " $seen " in
        *" $p "*) ;;
        *) DEDUPED+=("$p"); seen+=" $p" ;;
    esac
done

info "Installing extras (${EXTRAS[*]}): ${DEDUPED[*]}"
uv pip install --python "$PY" --constraints "$TORCH_CONSTRAINTS" "${DEDUPED[@]}"

"$PY" - <<'PY'
import torch

v = torch.__version__
hip = getattr(torch.version, "hip", None)
ok = hip is not None and "+rocm" in v
print(f"torch {v}  HIP={hip}")
if not ok:
    raise SystemExit(
        "ROCm torch was replaced. Re-run with --repair-torch.\n"
        "Never use: uv pip install -e '.[lingbot_va]' / '.[fastwam]' / 'lerobot[…]'"
    )
print("ROCm torch OK")
PY

info "Extras OK. Do not install lerobot[lingbot_va] / lerobot[fastwam] on this venv."
