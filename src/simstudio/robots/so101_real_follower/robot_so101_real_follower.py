"""SO-101 real follower arm implementation (stub)."""

from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError

from simstudio.robots.so101_real_follower.config import SO101RealFollowerConfig


class SO101RealFollowerRobot(Robot):
    """Real SO-101 follower arm (placeholder).

    Receives target joint positions and drives physical motors.
    """

    config_class = SO101RealFollowerConfig
    name = "so101_real_follower"

    def __init__(self, config: SO101RealFollowerConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return False

    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError("so101_real_follower hardware support not yet implemented")

    def calibrate(self) -> None:
        raise NotImplementedError("so101_real_follower hardware support not yet implemented")

    def configure(self, **kwargs) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        raise DeviceNotConnectedError("Robot not connected")

    def send_action(self, action: RobotAction) -> RobotAction:
        raise DeviceNotConnectedError("Robot not connected")

    def disconnect(self) -> None:
        self._connected = False
