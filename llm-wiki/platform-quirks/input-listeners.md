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

## Import-time hang on GNOME Xwayland

`import pynput` builds an `Xlib.display.Display()` at module scope. Under GNOME
Xwayland this can block **forever** (no timeout, no error) even when `xset q`
and GLFW windows work on the same `DISPLAY`. Any module that transitively
imports pynput then hangs before printing a single log line — for LeRobot that
is `lerobot.rollout` (→ DAgger strategy → `keyboard_input` → pynput).

Symptom: the process sits at ~1.4 GB RSS with the log file empty, threads in
`futex_do_wait`, and no GPU activity.

Workaround: hide `DISPLAY` for the duration of the import, then restore it, so
pynput falls back to its no-X path while GLFW still gets a real display later.

```python
_display = os.environ.pop("DISPLAY", None)
try:
    from lerobot.rollout import RolloutConfig  # pulls in pynput
finally:
    if _display is not None:
        os.environ["DISPLAY"] = _display
```

Applied in `src/simstudio/scripts/eval.py` and `src/simstudio/scripts/rollout.py`.
Required for GLFW GUI eval/rollout on Ubuntu 24.04 / GNOME Wayland; harmless
headless.

## Platform notes

| Platform | pynput backend | Notes |
|----------|---------------|-------|
| Linux X11 | `Xlib` | Works with root or `XINPUT` permissions |
| Linux Wayland | Limited | May need `evdev` backend or `root`; see Xwayland import hang above |
| macOS | `Quartz` | Works out of box |
| Windows | `win32api` | Works out of box |

## Verify

```python
from pynput import keyboard
# If this succeeds, pynput is available
```
