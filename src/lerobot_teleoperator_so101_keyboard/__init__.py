"""LeRobot third-party plugin package for the SO-101 keyboard teleoperator.

Importing this package registers SO101KeyboardTeleopConfig with LeRobot's
TeleoperatorConfig registry and exposes the SO101KeyboardTeleop class.
"""

from so101_mujoco_teleop.teleoperators.so101_keyboard import (
    SO101KeyboardTeleop,
    SO101KeyboardTeleopConfig,
)

__all__ = ["SO101KeyboardTeleop", "SO101KeyboardTeleopConfig"]
