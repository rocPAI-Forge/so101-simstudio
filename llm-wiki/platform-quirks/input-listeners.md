---
tags: [pynput, keyboard, input, listener]
platform: [linux, macos, windows]
update-check: 2026-07
---

# Keyboard Input Listeners

Different Python keyboard libraries have platform-specific pitfalls.

## pynput vs TerminalKeyListener

**pynput**: Global keyboard listener, works outside terminal focus. Uses OS-level hooks. Preferred for teleoperation.

**TerminalKeyListener** (from `pynput.keyboard._util`): Terminal-only listener. Falls back to this on some Linux systems when X11/Wayland hooks fail.

## The conflict

If both pynput and TerminalKeyListener run simultaneously, they compete for key events. One of them misses keys.

## Solution

When using pynput-based teleoperator, monkey-patch the upstream listener factory to skip TerminalKeyListener:

```python
import lerobot.utils.keyboard_input as _ki

def _patched_init_listener(*a, **kw):
    """Skip TerminalKeyListener; use only pynput."""
    return None

_ki.init_keyboard_listener = _patched_init_listener
```

**When to apply**: In project wrapper scripts that call LeRobot's record/replay, BEFORE importing LeRobot's record module.

## Platform notes

| Platform | pynput backend | Notes |
|----------|---------------|-------|
| Linux X11 | `Xlib` | Works with root or `XINPUT` permissions |
| Linux Wayland | Limited | May need `evdev` backend or `root` |
| macOS | `Quartz` | Works out of box |
| Windows | `win32api` | Works out of box |

## Verify

```python
from pynput import keyboard
# If this succeeds, pynput is available
```
