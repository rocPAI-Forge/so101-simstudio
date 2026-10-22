# Manual smoke tests

Interactive launchers for local hardware/teleop checks. These are **not** run by `make test` (pytest).

| Script | Purpose |
|--------|---------|
| `keyboard_record.sh [episodes] [resume] [view_mode]` | Record with keyboard |
| `keyboard_replay.sh [episode\|all]` | Replay keyboard smoke dataset |
| `keyboard_teleop.sh` | Keyboard teleop only (`simple_pick`) |
| `keyboard_teleop_kitchen.sh` | Keyboard teleop on `kitchen_counter` |
| `keyboard_teleop_smart_drawer.sh` | Keyboard teleop on `smart_drawer` (T2M 02) |
| `joycon_record.sh [episodes] [resume] [side] [view_mode]` | Record with Joy-Con |
| `leader_record.sh [episodes] [resume] [view_mode]` | Record with leader arm |
| `leader_teleop.sh` | Leader teleop only |

`view_mode` is `mujoco` (default) or `rerun`. Requires `.venv-rocm` (see `make rocm-sync`).

Makefile shortcuts: `make smoke-keyboard-record`, `make smoke-keyboard-replay`, `make smoke-keyboard-teleop`, etc.

For fixed 2-episode collaboration runs with `test.log` output, see `scripts/quicktest/`.
