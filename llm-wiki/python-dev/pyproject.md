---
tags: [pyproject, packaging, python]
platform: [linux, macos, windows]
update-check: 2026-07
---

# pyproject.toml Best Practices

## Dependency groups

Use `[dependency-groups]` for dev-only tools:

```toml
[dependency-groups]
dev = ["ruff", "pytest"]
```

## Editable local packages

When a monorepo contains multiple packages, list local packages as path dependencies:

```toml
[project]
dependencies = [
    "lerobot @ file:///${PROJECT_ROOT}/lerobot",
]
```

## Avoid mixing

Do NOT mix `uv sync` with raw `pip install` in the same environment. Pick one package manager per venv.

## Version pinning strategy

- Direct dependencies: loose (`>=X.Y`)
- Transitive ABI-sensitive deps: tight (constraints file)
- Dev tools: exact (`==X.Y.Z`)

## Verify

```bash
uv pip show <package>  # check source (PyPI vs local)
```
