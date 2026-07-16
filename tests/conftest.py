"""Shared pytest fixtures for SO-101 simstudio tests."""

import os

# Must be set before MuJoCo is imported during test collection.
os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from simstudio.common.recording_controls import reset_recording_debounce


@pytest.fixture(autouse=True)
def _reset_recording_debounce():
    """Isolate the module-level recording-control debounce state between tests.

    The debounce coalesces rapid same-key presses in a live session; tests fire
    many presses back-to-back, so reset it before each test to keep them isolated.
    """
    reset_recording_debounce()
    yield


@pytest.fixture(autouse=True)
def _enable_recording_keys_in_tests():
    """Recording hotkeys are disabled outside record_loop; enable for unit tests."""
    from simstudio.teleoperators.so101_keyboard.teleop_so101_keyboard import (
        set_recording_keys_enabled,
    )

    set_recording_keys_enabled(True)
    yield
    set_recording_keys_enabled(False)
