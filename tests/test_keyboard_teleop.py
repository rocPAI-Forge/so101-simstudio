"""Tests for SO101 keyboard teleoperator."""

from types import SimpleNamespace

from simstudio.teleoperators.so101_keyboard import SO101KeyboardTeleopConfig
from simstudio.teleoperators.so101_keyboard.evdev_listener import (
    _EVDEV_KEY_MAP,
    ev_key_is_pressed,
)
from simstudio.teleoperators.so101_keyboard.teleop_so101_keyboard import SO101KeyboardTeleop


def test_evdev_press_release_values_match_linux():
    """Linux EV_KEY: 1=press, 0=release — inverted constants cause sticky keys."""
    assert ev_key_is_pressed(1) is True
    assert ev_key_is_pressed(0) is False
    assert ev_key_is_pressed(2) is None


def test_evdev_o_then_c_does_not_stick_opposite_key():
    """Reproduce user report: buggy press/release left 'o' stuck when tapping C."""
    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    teleop._use_evdev = True
    teleop._evdev_listener = SimpleNamespace(_running=True)

    # press O, release O, press C, release C, press O
    for key, is_pressed in [("o", True), ("o", False), ("c", True), ("c", False), ("o", True)]:
        teleop._on_evdev_key(key, is_pressed)

    action = teleop.get_action()
    assert action["gripper_delta"] > 0.0  # O held -> open, not close from stuck C


def test_evdev_o_and_c_gripper_signs():
    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    teleop._use_evdev = True
    teleop._evdev_listener = SimpleNamespace(_running=True)

    teleop._on_evdev_key("o", True)
    assert teleop.get_action()["gripper_delta"] > 0.0
    teleop._on_evdev_key("o", False)

    teleop._on_evdev_key("c", True)
    assert teleop.get_action()["gripper_delta"] < 0.0
    teleop._on_evdev_key("c", False)
    assert teleop.get_action()["gripper_delta"] == 0.0


def test_evdev_key_map_matches_linux_scancodes():
    """Letter scancodes are not contiguous on Linux; D/Z must not map to O/C."""
    from evdev import ecodes

    assert _EVDEV_KEY_MAP[ecodes.KEY_D] == "d"
    assert _EVDEV_KEY_MAP[ecodes.KEY_Z] == "z"
    assert _EVDEV_KEY_MAP[ecodes.KEY_O] == "o"
    assert _EVDEV_KEY_MAP[ecodes.KEY_C] == "c"
    assert _EVDEV_KEY_MAP[ecodes.KEY_W] == "w"
    assert _EVDEV_KEY_MAP[ecodes.KEY_S] == "s"


def test_evdev_d_and_z_do_not_move_gripper():
    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    teleop._use_evdev = True
    teleop._evdev_listener = SimpleNamespace(_running=True)

    teleop._on_evdev_key("d", True)
    action = teleop.get_action()
    assert action["vx"] > 0.0
    assert action["gripper_delta"] == 0.0

    teleop._on_evdev_key("d", False)
    teleop._on_evdev_key("z", True)
    action = teleop.get_action()
    assert action["vz"] > 0.0
    assert action["gripper_delta"] == 0.0


def test_evdev_key_produces_velocity_action():
    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    teleop._use_evdev = True
    teleop._evdev_listener = SimpleNamespace(_running=True)

    teleop._on_evdev_key("w", True)
    action = teleop.get_action()

    assert action["vy"] > 0.0
    assert action["vx"] == 0.0

    teleop._on_evdev_key("w", False)
    action = teleop.get_action()
    assert action["vy"] == 0.0


def test_evdev_recording_keys_update_shared_events():
    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    teleop.set_recording_events(events)

    teleop._on_evdev_key("right", True)
    assert events["exit_early"] is True

    events["exit_early"] = False
    teleop._on_evdev_key("left", True)
    assert events["rerecord_episode"] is True
    assert events["exit_early"] is True


def test_evdev_n_r_q_recording_keys():
    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    teleop.set_recording_events(events)

    teleop._on_evdev_key("n", True)
    assert events["exit_early"] is True

    events["exit_early"] = False
    teleop._on_evdev_key("r", True)
    assert events["rerecord_episode"] is True
    assert events["exit_early"] is True

    events["exit_early"] = False
    events["rerecord_episode"] = False
    teleop._on_evdev_key("q", True)
    assert events["stop_recording"] is True
    assert events["exit_early"] is True


def test_evdev_recording_keys_link_shared_events_without_connect():
    import simstudio.teleoperators.so101_keyboard.teleop_so101_keyboard as teleop_module

    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    previous = teleop_module._shared_recording_events
    teleop_module._shared_recording_events = events
    try:
        teleop._on_evdev_key("right", True)
        assert events["exit_early"] is True
    finally:
        teleop_module._shared_recording_events = previous
