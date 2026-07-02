"""Nintendo Switch Joy-Con teleoperator (stub)."""

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from so101_mujoco_teleop.teleoperators.so101_joycon.config import SO101JoyConTeleopConfig


class SO101JoyConTeleop(Teleoperator):
    """Joy-Con teleoperator for SO-101 (planned)."""

    name = "so101_joycon"
    config_class = SO101JoyConTeleopConfig

    def __init__(self, config: SO101JoyConTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    def connect(self) -> None:
        raise NotImplementedError("Joy-Con teleop not yet implemented")

    def is_connected(self) -> bool:
        return self._connected

    def get_action(self) -> RobotAction:
        raise NotImplementedError("Joy-Con teleop not yet implemented")

    def send_feedback(self, action: RobotAction, **kwargs) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
