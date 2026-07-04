---
tags: [uv, dependency, constraints, packaging]
platform: [linux, macos, windows]
update-check: 2026-07
---

# uv Constraints

Use constraints files to pin transitive dependency versions without declaring them as direct dependencies. Critical for ABI-sensitive packages (PyTorch, MuJoCo, robotics stacks).

## Problem

Some packages have implicit version ranges enforced at build time but not in metadata. Example: `placo` needs specific `cmeel-urdfdom` versions, but pip/uv resolver doesn't know this.

## Solution: constraints file

Create `constraints.txt`:

```
cmeel-urdfdom>=4,<5
cmeel-tinyxml2<11
```

Apply during install:

```bash
uv sync --constraint constraints.txt
```

## Rules

1. Constraints only restrict — they cannot add new packages
2. Use `>=X,<Y` not `==X` for transitive deps (allow patch updates)
3. Place constraints file in repo root, not inside `.venv`
4. Document WHY each constraint exists in a comment in the constraints file

## Common pitfalls

- **PyTorch + constraints**: Install torch FIRST with `--torch-backend`, then apply constraints for remaining deps
- **Editable installs + constraints**: Local editable packages (`-e`) are not affected by constraints
- **uv lock**: Constraints are NOT reflected in `uv.lock`; they are a runtime install hint only

## Verify

```bash
uv pip show <package>  # check installed version
uv pip list | grep <package>  # check if constraint applied
```
