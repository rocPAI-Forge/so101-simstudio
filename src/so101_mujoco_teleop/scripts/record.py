"""Project wrapper around LeRobot's record script.

This module imports project-specific robot and teleoperator plugins so they
are registered in LeRobot's config registries, then delegates to LeRobot's
standard record script.
"""

import sys

# Register project plugins with LeRobot's ChoiceRegistry.
from so101_mujoco_teleop.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401
from so101_mujoco_teleop.robots.so101_real_follower import SO101RealFollowerConfig  # noqa: F401
from so101_mujoco_teleop.teleoperators.so101_joycon import SO101JoyConTeleopConfig  # noqa: F401
from so101_mujoco_teleop.teleoperators.so101_keyboard import (  # noqa: F401
    SO101KeyboardTeleopConfig,
)
from so101_mujoco_teleop.teleoperators.so101_leader import (  # noqa: F401
    SO101LeaderTeleopConfig,
)


def main():
    # Replace the first argument so LeRobot's parser sees its own script name.
    argv = ["lerobot_record.py"] + sys.argv[1:]
    sys.argv = argv

    from lerobot.scripts.lerobot_record import main as lerobot_record_main

    lerobot_record_main()


if __name__ == "__main__":
    main()
