# Quick-test launchers

Fixed one-command record runs for collaboration debugging. Output is always teed to
`test.log` at the repo root.

| Script | Teleop | Episodes | Notes |
|--------|--------|----------|-------|
| `joycon.cmd` | Joy-Con (right) | 2 | `SO101_JOYCON_DEBUG=1` for stick debug lines |
| `keyboard.cmd` | Keyboard | 2 | World-frame velocity teleop |
| `leader.cmd` | Real leader arm | 2 | Pass extra args, e.g. `--teleop.port /dev/ttyACM1` |

```bash
./scripts/quicktest/keyboard.cmd
./scripts/quicktest/joycon.cmd
./scripts/quicktest/leader.cmd
```

For parameterized smoke tests (episode count, resume, view mode), use `make smoke-*`
or `scripts/smoke/` — see `scripts/smoke/README.md`.
