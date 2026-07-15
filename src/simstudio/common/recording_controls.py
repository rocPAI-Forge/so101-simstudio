"""Shared keyboard recording control helpers for SO-101 record sessions."""

from __future__ import annotations

import time

from lerobot.utils.keyboard_input import apply_recording_control

from simstudio.common.constants import (
    KEYBOARD_RECORDING_CANCEL_KEYS,
    KEYBOARD_RECORDING_SAVE_KEYS,
    KEYBOARD_RECORDING_STOP_KEYS,
)

# Coalesce rapid repeats of the same recording control. Without this, mashing
# "save & next" (Right/N) lands the extra presses on the *next* episode and
# skips it with an empty buffer, silently eating episodes. A real episode always
# takes far longer than this window to record, so distinct presses are unaffected.
_RECORDING_KEY_DEBOUNCE_S = 0.8
_last_control_ts: dict[str, float] = {}


def _debounced(control: str) -> bool:
    """Return True if *control* fired too soon after the previous same control.

    Every call updates the timestamp, so a sustained burst stays fully coalesced.
    """
    now = time.monotonic()
    last = _last_control_ts.get(control)
    _last_control_ts[control] = now
    return last is not None and (now - last) < _RECORDING_KEY_DEBOUNCE_S


def reset_recording_debounce() -> None:
    """Clear debounce timestamps (call at the start of a new session or in tests)."""
    _last_control_ts.clear()


def apply_keyboard_recording_key(key_name: str, events: dict[str, bool]) -> bool:
    """Apply a recording control key to the shared LeRobot ``events`` dict.

    Save/cancel keys clear any spurious ``stop_recording`` flag first so a parallel
    terminal listener cannot end the whole session right after ``exit_early``.
    Rapid repeats of the same control are debounced so a burst of presses cannot
    skip subsequent episodes. Returns True when *key_name* was consumed as a
    recording control (even when the press was debounced/ignored).
    """
    key = key_name.lower()
    if key in KEYBOARD_RECORDING_STOP_KEYS:
        if not _debounced("stop"):
            apply_recording_control("esc", events)
        return True
    if key in KEYBOARD_RECORDING_SAVE_KEYS:
        if not _debounced("save"):
            events["stop_recording"] = False
            events["rerecord_episode"] = False
            apply_recording_control("right", events)
        return True
    if key in KEYBOARD_RECORDING_CANCEL_KEYS:
        if not _debounced("cancel"):
            events["stop_recording"] = False
            apply_recording_control("left", events)
        return True
    return False
