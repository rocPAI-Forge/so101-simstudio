"""Project wrapper around LeRobot's record script.

This module imports project-specific robot and teleoperator plugins so they
are registered in LeRobot's config registries, then delegates to LeRobot's
standard record script.

When the SO101 keyboard teleop is active, we monkey-patch
``init_keyboard_listener`` to skip the TerminalKeyListener (which conflicts
with pynput) and instead route arrow-key / ESC recording controls through
the teleop's own ``_on_press`` callback.

Supports view_mode parameter for Rerun camera feed display during recording.
"""

import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 1. Register project plugins with LeRobot's ChoiceRegistry.
#    These imports must happen BEFORE we import lerobot_record, because
#    lerobot_record itself imports the same config classes.
# ---------------------------------------------------------------------------
from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401
from simstudio.robots.so101_real_follower import SO101RealFollowerConfig  # noqa: F401
from simstudio.teleoperators.so101_joycon import SO101JoyConTeleopConfig  # noqa: F401
from simstudio.teleoperators.so101_keyboard import (  # noqa: F401
    SO101KeyboardTeleopConfig,
)
from simstudio.teleoperators.so101_leader import (  # noqa: F401
    SO101LeaderTeleopConfig,
)


# ---------------------------------------------------------------------------
# 2. Detect teleop type and view_mode from YAML config, then monkey-patch
#    init_keyboard_listener if keyboard teleop is active.
# ---------------------------------------------------------------------------


def _detect_config_value(key_path: str) -> str | None:
    """Extract a value from the --config YAML file on the command line."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            cfg_path = Path(sys.argv[i + 1])
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                # Support nested keys like "teleop.type"
                keys = key_path.split(".")
                value = cfg
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k)
                    else:
                        return None
                return value
    return None


def _detect_view_mode() -> str:
    """Detect view_mode from command line arguments."""
    # Check command line
    for i, arg in enumerate(sys.argv):
        if arg == "--view_mode" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "mujoco"  # default


_teleop_type = _detect_config_value("teleop.type")
_view_mode = _detect_view_mode()

if _teleop_type == "so101_keyboard":
    import lerobot.utils.keyboard_input as _ki

    _original_init = _ki.init_keyboard_listener

    def _patched_init_keyboard_listener():
        import logging

        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }
        logging.info(
            "SO101 keyboard teleop active — arrow keys and ESC handled by teleop "
            "(n/r/q recording shortcuts disabled to avoid key conflicts)."
        )
        return None, events

    _ki.init_keyboard_listener = _patched_init_keyboard_listener


# ---------------------------------------------------------------------------
# 3. Entry point — register plugins, configure view_mode, then call LeRobot's record().
# ---------------------------------------------------------------------------
def main():
    import logging

    # Convert view_mode to LeRobot's display_data/display_mode format
    display_data = _view_mode in ("rerun", "both")
    display_mode = "rerun" if display_data else "rerun"

    # Remove view_mode from argv and add display_data/display_mode if needed
    argv_filtered = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--view_mode":
            skip_next = True
            continue
        if arg in ("--display_data", "--display_mode"):
            skip_next = True
            continue
        argv_filtered.append(arg)

    # Add display_data/display_mode if view_mode is not mujoco
    if _view_mode != "mujoco":
        argv_filtered.extend(["--display_data", str(display_data)])
        argv_filtered.extend(["--display_mode", display_mode])

    # Replace argv so LeRobot's draccus parser sees its own script name.
    argv = ["lerobot_record.py"] + argv_filtered[1:]  # skip script name
    sys.argv = argv

    logging.info(f"View mode: {_view_mode} -> display_data={display_data}, display_mode={display_mode}")

    # Import AFTER patching so record() sees our patched init_keyboard_listener.
    from lerobot.scripts.lerobot_record import record  # noqa: E402

    record()


if __name__ == "__main__":
    main()
