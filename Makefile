.PHONY: format lint test rocm-sync rocm-format rocm-lint rocm-test rocm-smoke-record joycon-sync \
	smoke-keyboard-record smoke-keyboard-teleop smoke-joycon-record smoke-leader-record smoke-leader-teleop

SMOKE := scripts/smoke
EPISODES ?= 1
RESUME ?= false
VIEW_MODE ?= mujoco
SIDE ?= right

# ---------------------------------------------------------------------------
# CUDA / CPU (default) — uses .venv
# ---------------------------------------------------------------------------

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff check

test:
	uv run pytest

# ---------------------------------------------------------------------------
# ROCm — uses .venv-rocm (created by scripts/setup-rocm.sh)
# ---------------------------------------------------------------------------

ROCM_PY := .venv-rocm/bin/python

rocm-sync:
	scripts/setup-rocm.sh

rocm-test:
	$(ROCM_PY) -m pytest

rocm-smoke-record:
	rm -rf datasets/keyboard-smoke-rocm
	$(ROCM_PY) -m simstudio.scripts.record --config configs/so101_mujoco_keyboard_smoke.yaml

rocm-format:
	.venv-rocm/bin/ruff format
	.venv-rocm/bin/ruff check --fix

rocm-lint:
	.venv-rocm/bin/ruff check

# ---------------------------------------------------------------------------
# Joy-Con — install joycon-robotics with patches
# ---------------------------------------------------------------------------

joycon-sync:
	scripts/setup-joycon.sh

# ---------------------------------------------------------------------------
# Manual smoke tests (interactive; see scripts/smoke/README.md)
# ---------------------------------------------------------------------------

smoke-keyboard-record:
	$(SMOKE)/keyboard_record.sh $(EPISODES) $(RESUME) $(VIEW_MODE)

smoke-keyboard-teleop:
	$(SMOKE)/keyboard_teleop.sh

smoke-joycon-record:
	$(SMOKE)/joycon_record.sh $(EPISODES) $(RESUME) $(SIDE) $(VIEW_MODE)

smoke-leader-record:
	$(SMOKE)/leader_record.sh $(EPISODES) $(RESUME) $(VIEW_MODE)

smoke-leader-teleop:
	$(SMOKE)/leader_teleop.sh
