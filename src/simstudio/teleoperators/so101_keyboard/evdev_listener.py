"""evdev-based keyboard listener for Wayland.

Reads key events directly from /dev/input/event* devices, which works
regardless of window focus on Wayland. Falls back to pynput if evdev
is unavailable or no keyboard device is found.
"""

import logging
import re
import threading
from pathlib import Path
from queue import Queue
from typing import Callable

logger = logging.getLogger(__name__)

# evdev key code to logical key name mapping (subset used by keyboard teleop)
_EVDEV_KEY_MAP = {
    # Letters (KEY_A=30 .. KEY_Z=52)
    **{30 + i: chr(ord('a') + i) for i in range(26)},
    # Arrow keys
    103: "up",    # KEY_UP
    108: "down",  # KEY_DOWN
    105: "left",  # KEY_LEFT
    106: "right", # KEY_RIGHT
    # Special keys
    1: "esc",     # KEY_ESC
    57: "space",  # KEY_SPACE
    28: "enter",  # KEY_ENTER
    15: "tab",    # KEY_TAB
    14: "backspace",  # KEY_BACKSPACE
    # Brackets (KEY_LEFTBRACE=26, KEY_RIGHTBRACE=27)
    26: "[",
    27: "]",
}

# Event types
EV_KEY = 0x01
KEY_DOWN = 0
KEY_UP = 1
KEY_REPEAT = 2


def _find_keyboard_device() -> str | None:
    """Find the best keyboard device from /proc/bus/input/devices."""
    devices_info = Path("/proc/bus/input/devices").read_text()
    best_device = None
    best_priority = -1

    blocks = devices_info.split("\n\n")
    for block in blocks:
        name_match = re.search(r'N: Name="([^"]+)"', block)
        handler_match = re.search(r'H: Handlers=(.+)', block)
        if not name_match or not handler_match:
            continue

        name = name_match.group(1)
        handlers = handler_match.group(1)

        # Extract event device from handlers
        event_match = re.search(r'event(\d+)', handlers)
        if not event_match:
            continue

        event_num = int(event_match.group(1))
        device_path = f"/dev/input/event{event_num}"

        # Priority: AT Translated keyboard > USB keyboards > others
        priority = 0
        if "AT Translated" in name:
            priority = 100  # Built-in laptop keyboard
        elif "USB" in name and "kbd" in handlers.lower():
            priority = 50
        elif "kbd" in handlers.lower():
            priority = 10

        if priority > best_priority:
            best_priority = priority
            best_device = device_path

    return best_device


class EvdevKeyListener:
    """Keyboard listener using Linux evdev.

    Reads directly from /dev/input/event* devices, which works on Wayland
    regardless of window focus.

    Args:
        on_key: Callback invoked with (key_name, is_pressed) for each event.
    """

    def __init__(self, on_key: Callable[[str, bool], None]):
        self._on_key = on_key
        self._running = False
        self._thread: threading.Thread | None = None
        self._device_path: str | None = None

    def start(self) -> bool:
        """Start listening for keyboard events.

        Returns True if a keyboard device was found and listening started.
        """
        self._device_path = _find_keyboard_device()
        if self._device_path is None:
            logger.warning("No keyboard device found in /proc/bus/input/devices")
            return False

        try:
            import fcntl
            with open(self._device_path, "rb") as f:
                # EVIOCGRAB = 0x4008 4561 — not needed, just test read access
                pass
        except PermissionError:
            logger.warning(
                f"Permission denied reading {self._device_path}. "
                "Add user to 'input' group: sudo usermod -aG input $USER"
            )
            return False
        except Exception as e:
            logger.warning(f"Cannot access {self._device_path}: {e}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"evdev keyboard listener started on {self._device_path}")
        return True

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        """Read events from the keyboard device in a loop."""
        with open(self._device_path, "rb") as f:
            while self._running:
                # Read one input_event (24 bytes on 64-bit Linux)
                data = f.read(24)
                if data is None or len(data) < 24:
                    continue

                # struct input_event { time_t sec, usec; __u16 type, code; __s32 value; }
                import struct
                _time_sec, _time_usec, ev_type, ev_code, ev_value = struct.unpack("qqHHi", data)

                if ev_type != EV_KEY:
                    continue

                # KEY_DOWN=0, KEY_UP=1, KEY_REPEAT=2
                if ev_value == KEY_REPEAT:
                    continue

                is_pressed = (ev_value == KEY_DOWN)
                key_name = _EVDEV_KEY_MAP.get(ev_code)
                if key_name is None:
                    continue

                try:
                    self._on_key(key_name, is_pressed)
                except Exception as e:
                    logger.debug("evdev key handler error: %s", e)
