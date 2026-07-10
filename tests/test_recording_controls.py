"""Tests for keyboard recording control helpers."""

from simstudio.common.recording_controls import apply_keyboard_recording_key


def test_save_key_clears_spurious_stop_recording():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": True}

    assert apply_keyboard_recording_key("n", events) is True
    assert events["exit_early"] is True
    assert events["stop_recording"] is False
    assert events["rerecord_episode"] is False


def test_right_arrow_save_does_not_stop_session():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    assert apply_keyboard_recording_key("right", events) is True
    assert events["exit_early"] is True
    assert events["stop_recording"] is False


def test_esc_stops_session():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    assert apply_keyboard_recording_key("q", events) is True
    assert events["stop_recording"] is True
    assert events["exit_early"] is True


def test_recording_keys_ignored_outside_record_loop():
    from simstudio.common.recording_controls import apply_keyboard_recording_key
    from simstudio.teleoperators.so101_keyboard.teleop_so101_keyboard import (
        SO101KeyboardTeleop,
        SO101KeyboardTeleopConfig,
        set_recording_keys_enabled,
    )

    teleop = SO101KeyboardTeleop(SO101KeyboardTeleopConfig())
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    teleop.set_recording_events(events)

    set_recording_keys_enabled(False)
    teleop._on_evdev_key("n", True)
    assert events["exit_early"] is False

    set_recording_keys_enabled(True)
    teleop._on_evdev_key("n", True)
    assert events["exit_early"] is True


def test_multi_episode_loop_continues_after_save():
    """Simulate LeRobot record() outer loop after a save key."""
    num_episodes = 3
    recorded_episodes = 0
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    while recorded_episodes < num_episodes and not events["stop_recording"]:
        apply_keyboard_recording_key("n", events)
        events["exit_early"] = False

        if not events["stop_recording"] and (
            recorded_episodes < num_episodes - 1 or events["rerecord_episode"]
        ):
            pass  # reset window

        if events["rerecord_episode"]:
            events["rerecord_episode"] = False
            events["exit_early"] = False
            continue

        recorded_episodes += 1

    assert recorded_episodes == num_episodes
    assert events["stop_recording"] is False
