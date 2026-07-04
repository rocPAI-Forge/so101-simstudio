---
tags: [pypi, mirror, network]
platform: [linux, macos, windows]
update-check: 2026-07
---

# PyPI Mirror Configuration

Use when default PyPI is slow or blocked. Not a permanent setting — remove when network is fine.

## uv

```bash
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync
```

## pip

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

## Common mirrors

| Region | URL |
|--------|-----|
| Tsinghua (China) | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| Alibaba (China) | `https://mirrors.aliyun.com/pypi/simple` |
| USTC (China) | `https://pypi.mirrors.ustc.edu.cn/simple` |

## When NOT to use

- CI/CD pipelines: use cached wheels or private registry instead
- Dependency resolution: mirrors can lag behind PyPI; if resolution fails, try default first
- Private packages: mirrors only serve public PyPI; use `--extra-index-url` for private registries

## Verify

```bash
uv pip config get index-url  # check active mirror
```
