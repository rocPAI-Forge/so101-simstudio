"""Project wrapper around LeRobot's record script.

This module imports project-specific robot and teleoperator plugins so they
are registered in LeRobot's config registries, then delegates to LeRobot's
standard record script.

When the SO101 keyboard teleop is active, we monkey-patch
``init_keyboard_listener`` to skip the TerminalKeyListener (which conflicts
with pynput) and instead route arrow-key / ESC recording controls through
the teleop's own ``_on_press`` callback.
"""

import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 1. Register project plugins with LeRobot's ChoiceRegistry.
#    These imports must happen BEFORE we import lerobot_record, because
#    lerobot_record itself imports the same config classes.
# ---------------------------------------------------------------------------
from so101_mujoco_teleop.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401
from so101_mujoco_teleop.robots.so101_real_follower import SO101RealFollowerConfig  # noqa: F401
from so101_mujoco_teleop.teleoperators.so101_joycon import SO101JoyConTeleopConfig  # noqa: F401
from so101_mujoco_teleop.teleoperators.so101_keyboard import (  # noqa: F401
    SO101KeyboardTeleopConfig,
)
from so101_mujoco_teleop.teleoperators.so101_leader import (  # noqa: F401
    SO101LeaderTeleopConfig,
)

# ---------------------------------------------------------------------------
# 2. Detect teleop type from YAML config and monkey-patch init_keyboard_listener.
#    The patched version, when SO101 keyboard is active, returns a minimal
#    events dict with listener=None — arrow keys / ESC are handled by the
#    teleop's pynput listener instead.
# ---------------------------------------------------------------------------


def _detect_teleop_type() -> str | None:
    """Extract teleop.type from the --config YAML file on the command line."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            cfg_path = Path(sys.argv[i + 1])
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("teleop", {}).get("type")
    return None


_teleop_type = _detect_teleop_type()

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
# 3. Entry point — register plugins, then call LeRobot's record().
# ---------------------------------------------------------------------------
def main():
    # Replace argv so LeRobot's draccus parser sees its own script name.
    argv = ["lerobot_record.py"] + sys.argv[1:]
    sys.argv = argv

    # Import AFTER patching so record() sees our patched init_keyboard_listener.
    from lerobot.scripts.lerobot_record import record  # noqa: E402

    record()


if __name__ == "__main__":
    main()
