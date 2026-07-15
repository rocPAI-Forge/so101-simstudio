"""Shared pytest fixtures for SO-101 simstudio tests."""

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
