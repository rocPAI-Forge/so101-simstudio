"""SO-101 keyboard teleoperator.

Outputs normalized velocity commands for SO-101 end-effector control.
"""

from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.keyboard_input import init_keyboard_listener

from so101_mujoco_teleop.teleoperators.so101_keyboard.config import SO101KeyboardTeleopConfig


class SO101KeyboardTeleop(Teleoperator):
    """Keyboard teleoperator for SO-101.

    Maps keys to velocity commands:
      W/S : +Y / -Y
      A/D : -X / +X
      Q/E : +Z / -Z
      I/K : wrist flex up/down
      [/] : wrist roll left/right
      O/C : gripper open/close
    """

    name = "so101_keyboard"
    config_class = SO101KeyboardTeleopConfig

    def __init__(self, config: SO101KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._pressed: set[str] = set()

    def connect(self) -> None:
        # TODO: integrate with lerobot keyboard listener
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def get_action(self) -> RobotAction:
        # TODO: read keyboard state and return velocity dict
        return {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wrist_flex_rate": 0.0,
            "yaw_rate": 0.0,
            "gripper_delta": 0.0,
        }

    def send_feedback(self, action: RobotAction, **kwargs) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
