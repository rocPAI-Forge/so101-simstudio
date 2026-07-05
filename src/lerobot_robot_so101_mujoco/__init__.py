"""LeRobot third-party plugin package for the SO-101 MuJoCo robot.

Importing this package registers SO101MujocoConfig with LeRobot's RobotConfig
registry and exposes the SO101Mujoco robot class.
"""

from simstudio.robots.so101_mujoco import (
    SO101Mujoco,
    SO101MujocoConfig,
    SO101MujocoRobot,
)

__all__ = ["SO101Mujoco", "SO101MujocoConfig", "SO101MujocoRobot"]
