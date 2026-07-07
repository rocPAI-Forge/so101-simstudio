"""evdev-based keyboard listener for Wayland.

Reads key events directly from /dev/input/event* devices, which works
regardless of window focus on Wayland. Falls back to pynput if evdev
is unavailable or no keyboard device is found.
"""

import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_evdev_key_map() -> dict[int, str]:
    """Build Linux evdev scancode -> logical key name map for teleop."""
    try:
        from evdev import ecodes

        key_map = {getattr(ecodes, f"KEY_{letter.upper()}"): letter for letter in "abcdefghijklmnopqrstuvwxyz"}
        key_map.update(
            {
                ecodes.KEY_UP: "up",
                ecodes.KEY_DOWN: "down",
                ecodes.KEY_LEFT: "left",
                ecodes.KEY_RIGHT: "right",
                ecodes.KEY_ESC: "esc",
                ecodes.KEY_SPACE: "space",
                ecodes.KEY_ENTER: "enter",
                ecodes.KEY_TAB: "tab",
                ecodes.KEY_BACKSPACE: "backspace",
                ecodes.KEY_LEFTBRACE: "[",
                ecodes.KEY_RIGHTBRACE: "]",
            }
        )
        return key_map
    except ImportError:
        # Fallback when evdev is unavailable (must match linux/input-event-codes.h).
        return {
            30: "a",
            48: "b",
            46: "c",
            32: "d",
            18: "e",
            33: "f",
            34: "g",
            35: "h",
            23: "i",
            36: "j",
            37: "k",
            38: "l",
            50: "m",
            49: "n",
            24: "o",
            25: "p",
            16: "q",
            19: "r",
            31: "s",
            20: "t",
            22: "u",
            47: "v",
            17: "w",
            45: "x",
            21: "y",
            44: "z",
            103: "up",
            108: "down",
            105: "left",
            106: "right",
            1: "esc",
            57: "space",
            28: "enter",
            15: "tab",
            14: "backspace",
            26: "[",
            27: "]",
        }


# evdev key code to logical key name mapping (subset used by keyboard teleop)
_EVDEV_KEY_MAP = _build_evdev_key_map()

# Event types (linux/input-event-codes.h)
EV_KEY = 0x01
# EV_KEY value: 0 = release, 1 = press, 2 = repeat (NOT 0=press!)
KEY_RELEASE = 0
KEY_PRESS = 1
KEY_REPEAT = 2


def ev_key_is_pressed(ev_value: int) -> bool | None:
    """Map Linux EV_KEY value to pressed state. Returns None for repeat/unknown."""
    if ev_value == KEY_PRESS:
        return True
    if ev_value == KEY_RELEASE:
        return False
    return None


def _find_keyboard_device() -> str | None:
    """Find the best keyboard device from /proc/bus/input/devices."""
    devices_info = Path("/proc/bus/input/devices").read_text()
    best_device = None
    best_priority = -1

    blocks = devices_info.split("\n\n")
    for block in blocks:
        name_match = re.search(r'N: Name="([^"]+)"', block)
        handler_match = re.search(r"H: Handlers=(.+)", block)
        if not name_match or not handler_match:
            continue

        name = name_match.group(1)
        handlers = handler_match.group(1)

        # Extract event device from handlers
        event_match = re.search(r"event(\d+)", handlers)
        if not event_match:
            continue

        event_num = int(event_match.group(1))
        device_path = f"/dev/input/event{event_num}"

        # Prefer USB keyboards — users often teleop on an external keyboard while
        # the built-in AT Translated device stays idle or receives ghost events.
        priority = 0
        if "USB" in name and "kbd" in handlers.lower():
            priority = 100
        elif "AT Translated" in name:
            priority = 50
        elif "kbd" in handlers.lower():
            priority = 10

        if priority > best_priority:
            best_priority = priority
            best_device = device_path
            logger.debug("evdev candidate: %s -> %s (priority=%d)", name, device_path, priority)

    if best_device is not None:
        logger.info("Selected evdev keyboard device: %s", best_device)

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

            with open(self._device_path, "rb") as _:
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

                is_pressed = ev_key_is_pressed(ev_value)
                if is_pressed is None:
                    continue

                key_name = _EVDEV_KEY_MAP.get(ev_code)
                if key_name is None:
                    continue

                try:
                    self._on_key(key_name, is_pressed)
                except Exception as e:
                    logger.debug("evdev key handler error: %s", e)
