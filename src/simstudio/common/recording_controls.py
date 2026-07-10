"""Shared keyboard recording control helpers for SO-101 record sessions."""

from __future__ import annotations

from lerobot.utils.keyboard_input import apply_recording_control

from simstudio.common.constants import (
    KEYBOARD_RECORDING_CANCEL_KEYS,
    KEYBOARD_RECORDING_SAVE_KEYS,
    KEYBOARD_RECORDING_STOP_KEYS,
)


def apply_keyboard_recording_key(key_name: str, events: dict[str, bool]) -> bool:
    """Apply a recording control key to the shared LeRobot ``events`` dict.

    Save/cancel keys clear any spurious ``stop_recording`` flag first so a parallel
    terminal listener cannot end the whole session right after ``exit_early``.
    Returns True when *key_name* was consumed as a recording control.
    """
    key = key_name.lower()
    if key in KEYBOARD_RECORDING_STOP_KEYS:
        apply_recording_control("esc", events)
        return True
    if key in KEYBOARD_RECORDING_SAVE_KEYS:
        events["stop_recording"] = False
        events["rerecord_episode"] = False
        apply_recording_control("right", events)
        return True
    if key in KEYBOARD_RECORDING_CANCEL_KEYS:
        events["stop_recording"] = False
        apply_recording_control("left", events)
        return True
    return False
