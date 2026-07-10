"""Unit tests for simstudio.scripts.record view_mode helpers."""

import sys

import numpy as np
import pytest

from simstudio.scripts.record import (
    DEFAULT_VIEW_MODE,
    apply_view_mode,
    detect_view_mode,
)


def test_detect_view_mode_default():
    assert detect_view_mode(["--config", "cfg.yaml"]) == DEFAULT_VIEW_MODE


def test_detect_view_mode_explicit():
    argv = ["--config", "cfg.yaml", "--view_mode", "rerun"]
    assert detect_view_mode(argv) == "rerun"


def test_detect_view_mode_equals_form():
    argv = ["--view_mode=rerun", "--config", "cfg.yaml"]
    assert detect_view_mode(argv) == "rerun"


def test_apply_view_mode_mujoco_defaults():
    argv = ["--config", "configs/so101_mujoco_keyboard.yaml"]
    result = apply_view_mode(argv)

    assert "--view_mode" not in result
    assert result == [
        "--config",
        "configs/so101_mujoco_keyboard.yaml",
        "--display_data",
        "false",
        "--robot.render_window",
        "true",
    ]


def test_apply_view_mode_rerun_defaults():
    argv = ["--config", "cfg.yaml", "--view_mode", "rerun"]
    result = apply_view_mode(argv)

    assert result == [
        "--config",
        "cfg.yaml",
        "--display_data",
        "true",
        "--display_mode",
        "rerun",
        "--robot.render_window",
        "false",
    ]


def test_apply_view_mode_respects_explicit_display_data():
    argv = ["--view_mode", "rerun", "--display_data", "false"]
    result = apply_view_mode(argv)

    assert "--display_data" in result
    assert result.count("--display_data") == 1
    idx = result.index("--display_data")
    assert result[idx + 1] == "false"
    assert "--display_mode" in result
    assert "--robot.render_window" in result
    assert result[result.index("--robot.render_window") + 1] == "false"


def test_apply_view_mode_respects_explicit_render_window():
    argv = ["--view_mode", "rerun", "--robot.render_window", "true"]
    result = apply_view_mode(argv)

    idx = result.index("--robot.render_window")
    assert result[idx + 1] == "true"
    assert result[result.index("--display_data") + 1] == "true"


def test_apply_view_mode_invalid():
    with pytest.raises(ValueError, match="Invalid --view_mode"):
        apply_view_mode(["--view_mode", "both"])


def test_patch_rerun_streaming_omits_static(monkeypatch):
    """Streaming patch must not log images with static=True (causes black panes)."""
    calls: list[tuple] = []

    class DummyImage:
        def __init__(self, arr):
            self.arr = arr

        def compress(self, *args, **kwargs):
            return self

    class DummyRR:
        @staticmethod
        def Scalars(v):
            return v

        @staticmethod
        def Image(arr):
            return DummyImage(arr)

        @staticmethod
        def DepthImage(*args, **kwargs):
            return None

        class components:
            class Colormap:
                Viridis = "v"

        @staticmethod
        def log(key, obj=None, **kwargs):
            entity = obj if obj is not None else kwargs.get("entity")
            calls.append((key, entity, kwargs))

        @staticmethod
        def set_time(*args, **kwargs):
            return None

    dummy_rr = DummyRR()
    dummy_rr.__spec__ = type("Spec", (), {"name": "rerun"})()

    monkeypatch.setitem(sys.modules, "rerun", dummy_rr)

    import lerobot.utils.rerun_visualization as rv
    from lerobot.utils import import_utils

    monkeypatch.setattr(rv, "_ensure_blueprint", lambda *a, **k: None)
    monkeypatch.setattr(import_utils, "require_package", lambda *a, **k: None)

    from simstudio.scripts import record as record_mod

    record_mod._patch_rerun_streaming()

    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[10:20, 10:30, 0] = 255
    rv.log_rerun_data(observation={"camera_front": img})

    image_calls = [c for c in calls if isinstance(c[1], DummyImage)]
    assert len(image_calls) == 1
    assert image_calls[0][2].get("static") is not True


def test_patch_rerun_streaming_updates_visualization_utils(monkeypatch):
    """Record loop calls ``visualization_utils.log_rerun_data``, not ``rerun_visualization``."""
    calls: list[tuple] = []

    class DummyImage:
        def __init__(self, arr):
            self.arr = arr

        def compress(self, *args, **kwargs):
            return self

    class DummyRR:
        @staticmethod
        def Scalars(v):
            return v

        @staticmethod
        def Image(arr):
            return DummyImage(arr)

        @staticmethod
        def DepthImage(*args, **kwargs):
            return None

        class components:
            class Colormap:
                Viridis = "v"

        @staticmethod
        def log(key, obj=None, **kwargs):
            entity = obj if obj is not None else kwargs.get("entity")
            calls.append((key, entity, kwargs))

        @staticmethod
        def set_time(*args, **kwargs):
            return None

    dummy_rr = DummyRR()
    dummy_rr.__spec__ = type("Spec", (), {"name": "rerun"})()

    monkeypatch.setitem(sys.modules, "rerun", dummy_rr)

    import lerobot.utils.rerun_visualization as rv
    import lerobot.utils.visualization_utils as vu
    from lerobot.utils import import_utils

    monkeypatch.setattr(rv, "_ensure_blueprint", lambda *a, **k: None)
    monkeypatch.setattr(import_utils, "require_package", lambda *a, **k: None)

    from simstudio.scripts import record as record_mod

    record_mod._patch_rerun_streaming()

    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[10:20, 10:30, 0] = 255
    vu.log_visualization_data("rerun", observation={"camera_front": img})

    image_calls = [c for c in calls if isinstance(c[1], DummyImage)]
    assert len(image_calls) == 1
    assert image_calls[0][2].get("static") is not True


def test_init_record_session_from_argv_cli_override():
    from simstudio.scripts.record import _init_record_session_from_argv, _record_session

    _init_record_session_from_argv(
        ["--config", "configs/so101_mujoco_keyboard.yaml", "--dataset.num_episodes", "5"]
    )
    assert _record_session["num_episodes"] == 5


def test_keyboard_recording_uses_evdev_for_all_view_modes(monkeypatch):
    import os

    from simstudio.scripts import record as record_mod

    monkeypatch.delenv("SO101_PREFER_EVDEV", raising=False)
    record_mod._patch_keyboard_recording._so101_patched = False
    record_mod._patch_keyboard_recording()
    assert os.environ.get("SO101_PREFER_EVDEV") == "1"


def test_keyboard_recording_skips_duplicate_listener():
    """Recording controls must come only from teleop (no terminal/pynput duplicate)."""
    import lerobot.scripts.lerobot_record as lr

    from simstudio.scripts import record as record_mod

    record_mod._patch_keyboard_recording._so101_patched = False
    record_mod._patch_keyboard_recording()
    listener, events = lr.init_keyboard_listener()
    assert listener is None
    assert events is record_mod._recording_events


def test_keyboard_recording_uses_shared_events_dict():
    import lerobot.scripts.lerobot_record as lr

    from simstudio.scripts import record as record_mod
    from simstudio.common.recording_controls import apply_keyboard_recording_key

    record_mod._patch_keyboard_recording._so101_patched = False
    record_mod._patch_keyboard_recording()
    _, events = lr.init_keyboard_listener()

    events["stop_recording"] = True
    apply_keyboard_recording_key("right", events)
    assert events["exit_early"] is True
    assert events["stop_recording"] is False
