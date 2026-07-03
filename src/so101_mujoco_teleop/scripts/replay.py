"""Project wrapper around LeRobot's replay script.

Imports SO-101 MuJoCo robot plugin so it is registered in LeRobot's
config registries, then delegates to LeRobot's standard replay script.
"""

import sys

# Register SO-101 MuJoCo robot plugin
from so101_mujoco_teleop.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401


def main():
    argv = ["lerobot_replay.py"] + sys.argv[1:]
    sys.argv = argv

    from lerobot.scripts.lerobot_replay import main as lerobot_replay_main

    lerobot_replay_main()


if __name__ == "__main__":
    main()
